from prometheus_client import Counter, Gauge, start_http_server

# Exposure Port
METRICS_PORT = 8000

# Matrices
INGESTION_BATCH_TOTAL = Counter(
    'ingestion_batch_total', 
    'Total number of batches processed'
)

ROWS_INGESTED_TOTAL = Counter(
    'rows_ingested_total', 
    'Total number of measurement rows ingested into raw storage'
)

CLEANING_VIOLATIONS_TOTAL = Counter(
    'cleaning_violations_total',
    'Total number of values rejected or modified during cleaning',
    ['violation_type'] # label: 'hard_limit', 'temporal_coherence', etc.
)

VALUES_IMPUTED_TOTAL = Counter(
    'values_imputed_total',
    'Total number of values imputed (filled) during smart infilling',
    ['method'] # label: 'linear', 'spline'
)

DATA_GAP_SIZE = Gauge(
    'data_gap_size_hours',
    'Size of the gap found in data in hours',
    ['station_id', 'sensor_id']
)

def start_metrics_server():
    """Starts the Prometheus metrics server on the specified port."""
    try:
        start_http_server(METRICS_PORT)
        print(f"Metrics server started on port {METRICS_PORT}")
    except Exception as e:
        print(f"Could not start metrics server (might be already running): {e}")
