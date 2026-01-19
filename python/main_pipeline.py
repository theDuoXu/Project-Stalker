import time
import pandas as pd
from pipeline.metrics import start_metrics_server
from pipeline.cleaning import CleanerPipeline, HardLimitsCleaner, TemporalCoherenceCleaner, SmartInfillingCleaner, FlatLineCleaner
from pipeline.ingest import load_and_deduplicate, archive_batches

def run_cleaning_phase(df):
    """
    Applies cleaning logic to the provided DataFrame (In-Memory).
    """
    print("\nSTARTING CLEANING PHASE (In-Memory)...")
    
    if df.empty:
        print("No data to clean.")
        return

    # Group by Station + Sensor
    # In Pandas, we can just group
    groups = df.groupby(['station_id', 'sensor_id'])
    
    # We are NOT writing to DB yet, just verifying logic
    total_cleaned_rows = 0
    
    for (station_id, sensor_id), group_df in groups:
        # Set Index
        group_df = group_df.set_index('timestamp').sort_index()
        
        # Build Pipeline
        cleaner_stack = CleanerPipeline([
            HardLimitsCleaner(sensor_id),
            TemporalCoherenceCleaner(sensor_id),
            SmartInfillingCleaner(sensor_id),
            FlatLineCleaner(sensor_id)
        ])
        
        # Apply
        cleaned_df = cleaner_stack.clean(group_df.copy())
        total_cleaned_rows += len(cleaned_df)
        
    print(f"✅ Cleaning Phase Complete. Processed {total_cleaned_rows} rows.")

def main():
    print("Starting Pipeline...")
    
    # 1. Start Metrics Server
    start_metrics_server()
    
    # 2. Ingest & Deduplicate (Pandas)
    df = load_and_deduplicate()
    
    if df.empty:
        print("No data found. Exiting.")
        return

    # 3. Archive Batches (Move to data/datasets/archive)
    # User requested: "The python file should move all batches... It should not leave them."
    archive_batches()
    
    # 4. Cleaning (In-Memory)
    run_cleaning_phase(df)
    
    # 5. Database Write
    # User command: "Do not connect the database unless I tell you to."
    print("🔒 Database write is currently DISABLED by user request.")
    
    print("Pipeline Finished.")

if __name__ == "__main__":
    main()
