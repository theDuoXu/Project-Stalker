import os
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement

# Configuration
CASSANDRA_HOST = os.getenv('CASSANDRA_HOST', 'localhost')
CASSANDRA_PORT = int(os.getenv('CASSANDRA_PORT', 9042))
KEYSPACE = 'water_quality'

def get_cluster():
    """Establishes a connection to the Cassandra cluster."""
    return Cluster([CASSANDRA_HOST], port=CASSANDRA_PORT)

def create_keyspace(session):
    """Creates the keyspace if it doesn't exist."""
    session.execute(f"""
        CREATE KEYSPACE IF NOT EXISTS {KEYSPACE}
        WITH REPLICATION = {{ 
            'class' : 'SimpleStrategy', 
            'replication_factor' : 1 
        }};
    """)

def create_tables(session):
    """Creates the necessary tables for the pipeline."""
    session.set_keyspace(KEYSPACE)

    # Raw Measurements Table
    # Stores data exactly as received from the source.
    # Partition by station_id, Cluster by sensor_id and timestamp.
    # This allows efficient retrieval of all sensors for a station over time, 
    # or specific sensor history.
    # Deduplication handles naturally: inserting the same PK updates the record (Idempotent).
    session.execute("""
        CREATE TABLE IF NOT EXISTS raw_measurements (
            station_id text,
            sensor_id text,
            timestamp timestamp,
            value double,
            metric text,
            unit text,
            metadata_json text,
            PRIMARY KEY (station_id, sensor_id, timestamp)
        ) WITH CLUSTERING ORDER BY (sensor_id ASC, timestamp ASC);
    """)

    # Clean Measurements Table
    # Stores the processed, cleaned, and filled data.
    session.execute("""
        CREATE TABLE IF NOT EXISTS clean_measurements (
            station_id text,
            sensor_id text,
            timestamp timestamp,
            value double,
            metric text,
            unit text,
            is_imputed boolean,
            quality_flag text,
            metadata_json text,
            PRIMARY KEY (station_id, sensor_id, timestamp)
        ) WITH CLUSTERING ORDER BY (sensor_id ASC, timestamp ASC);
    """)
    
    print("✅ Schema initialized: Keyspace and Tables ready.")

def get_max_timestamp(session, station_id, sensor_id):
    """
    Retrieves the maximum timestamp for a specific station and sensor.
    Used for High Watermark Deduplication.
    """
    query = "SELECT MAX(timestamp) as max_ts FROM raw_measurements WHERE station_id=%s AND sensor_id=%s"
    # Note: select max(clustering_key) is efficient in Cassandra (single partition read)
    try:
        row = session.execute(query, (station_id, sensor_id)).one()
        return row.max_ts if row else None
    except Exception as e:
        # If table doesn't exist or other error, return None to ingest all
        return None

def initialize_db():
    """Main function to setup the DB."""
    cluster = get_cluster()
    session = cluster.connect()
    create_keyspace(session)
    create_tables(session)
    return session
