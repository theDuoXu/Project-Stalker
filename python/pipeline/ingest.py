import os
import json
import glob
import shutil
from pathlib import Path
import pandas as pd
from datetime import datetime
from cassandra.query import BatchStatement
from .database import get_cluster, initialize_db, get_max_timestamp
from .metrics import INGESTION_BATCH_TOTAL, ROWS_INGESTED_TOTAL

RAW_DATA_DIR = Path('/home/duo/Projects/proyecto-computacion-I/data/datasets/raw')
ARCHIVE_DIR = Path('/home/duo/Projects/proyecto-computacion-I/data/datasets/archive')

def process_json(filepath, batch_name):
    """Reads a JSON and returns a flattened DataFrame."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        return pd.DataFrame()

    # Extract global metadata
    resp = data.get('response', {})
    metric_name = resp.get('nombre')
    unit = resp.get('unidad')

    # Extract values
    valores = resp.get('valores', [])

    if not valores:
        return pd.DataFrame()

    # Create DataFrame
    df = pd.DataFrame(valores)

    # Cleaning & Typing
    # 1. Convert timestamp
    if 'tiempo' in df.columns:
        # Standardize SAICA format dd/MM/yyyy HH:mm
        df['timestamp'] = pd.to_datetime(df['tiempo'], format='%d/%m/%Y %H:%M', errors='coerce')
    
    # 2. Ensure value is float
    if 'valor' in df.columns:
        df['value'] = pd.to_numeric(df['valor'], errors='coerce')

    # 3. Add context columns
    df['metric'] = metric_name
    df['unit'] = unit
    df['batch'] = batch_name
    
    # Capture original metadata as JSON string for DB
    # We remove 'valores' to save space, keep the rest
    meta = data.copy()
    if 'response' in meta and 'valores' in meta['response']:
        meta['response'].pop('valores', None)
    df['metadata_json'] = json.dumps(meta)

    # Optional: Extract sensor from filename
    # C302_AMONIO.json -> C302 (Station), AMONIO (Sensor)
    # The user example: sensor_from_filename = filepath.stem.split('_')[0]
    # But usually filename is STATION_SENSOR.
    # User's code: sensor_from_filename = filepath.stem.split('_')[0] -> This gets "C302"
    # Wait, we need Station ID AND Sensor ID.
    
    filename_parts = filepath.stem.split('_')
    if len(filename_parts) >= 2:
        df['station_id'] = filename_parts[0]
        # Sensor ID is the rest? e.g. AMONIO (or O_DISUELTO if underscores?)
        # Let's rely on filename conventions or the 'metric' from JSON?
        # User prompt example used filepath.stem.split('_')[0] for `sensor_id`. 
        # But `C302` is usually the Station. `AMONIO` is the Sensor/Param.
        # Let's assume standard: STATION_SENSOR
        df['station_id'] = filename_parts[0]
        df['sensor_id'] = '_'.join(filename_parts[1:])
    else:
        # Fallback
        df['station_id'] = 'UNKNOWN'
        df['sensor_id'] = 'UNKNOWN'

    # Select useful columns
    cols_to_keep = ['timestamp', 'value', 'station_id', 'sensor_id', 'metric', 'unit', 'batch', 'metadata_json']
    
    # Filter only columns that exist
    cols = [c for c in cols_to_keep if c in df.columns]
    return df[cols]

def load_and_deduplicate():
    """Iterates all batches, loads to RAM, dedupes."""
    all_dfs = []
    
    # Ensure raw directory exists
    if not RAW_DATA_DIR.exists():
        print(f"⚠️ Raw data directory not found: {RAW_DATA_DIR}")
        return pd.DataFrame()

    batches = list(RAW_DATA_DIR.glob('batch*'))
    if not batches:
         print("⚠️ No batches found.")
         return pd.DataFrame()

    for batch_dir in batches:
        print(f"📦 Processing {batch_dir.name}...")
        
        json_files = list(batch_dir.glob('*.json'))
        if not json_files:
            continue
            
        for json_file in json_files:
            df_temp = process_json(json_file, batch_dir.name)
            if not df_temp.empty:
                all_dfs.append(df_temp)
                
    if not all_dfs:
        return pd.DataFrame()
        
    raw_df = pd.concat(all_dfs, ignore_index=True)
    
    # Sort for deterministic Keep='last'
    if 'timestamp' in raw_df.columns:
        raw_df.sort_values(by=['batch', 'timestamp'], inplace=True)
    
    # Deduplication
    # Unique Key: Station + Sensor + Timestamp
    subset_unique = ['station_id', 'sensor_id', 'timestamp']
    dupes_count = raw_df.duplicated(subset=subset_unique, keep='last').sum()
    
    print(f"🔥 Duplicates detected (batch overlap): {dupes_count:,}")
    
    clean_df = raw_df.drop_duplicates(subset=subset_unique, keep='last').copy()
    print(f"✅ Final unique records: {len(clean_df):,}")
    print(f"🗑️ Removed {len(raw_df) - len(clean_df)} redundant rows.")
    
    return clean_df

def archive_batches():
    """Moves processed batches to archive."""
    if not ARCHIVE_DIR.exists():
        ARCHIVE_DIR.mkdir(parents=True)
        
    batches = list(RAW_DATA_DIR.glob('batch*'))
    for batch_dir in batches:
        try:
            shutil.move(str(batch_dir), str(ARCHIVE_DIR / batch_dir.name))
            print(f"🗄️ Archived: {batch_dir.name}")
        except Exception as e:
            print(f"⚠️ Failed to archive {batch_dir.name}: {e}")

def run_ingestion():
    session = initialize_db()
    
    # Load and deduplicate all data from raw batches
    clean_df = load_and_deduplicate()

    if not clean_df.empty:
        print(f"🚀 Starting ingestion of {len(clean_df):,} unique records into Cassandra...")

        # Prepare Statement
        insert_stmt = session.prepare("""
            INSERT INTO raw_measurements (station_id, sensor_id, timestamp, value, metric, unit, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """)
        
        batch = BatchStatement()
        batch_count = 0
        BATCH_SIZE = 20 # Reduced from 50 to avoid "Batch too large" warnings (Target < 5KB)
        
        total_inserted = 0

        for index, row in clean_df.iterrows():
            # Ensure timestamp is a datetime object for Cassandra
            timestamp_dt = row['timestamp'].to_pydatetime() if pd.notna(row['timestamp']) else None
            
            batch.add(insert_stmt, (
                row['station_id'], 
                row['sensor_id'], 
                timestamp_dt, 
                row['value'],
                row['metric'],
                row['unit'],
                row['metadata_json']
            ))
            batch_count += 1
            
            if batch_count >= BATCH_SIZE:
                session.execute(batch)
                total_inserted += batch_count
                batch = BatchStatement()
                batch_count = 0
        print("No raw data directory found.")
        return

    subdirs = sorted([d for d in os.listdir(RAW_DATA_DIR) if d.startswith('batch')])
    
    for subdir in subdirs:
        full_path = os.path.join(RAW_DATA_DIR, subdir)
        if os.path.isdir(full_path):
            ingest_batch(full_path, session)

    session.shutdown()
