import json
import os

# Path to JSON
json_path = 'compute-engine/src/main/resources/saica-stations.json'
twin_id = '6bea03cf-3acb-488e-b93f-508424fcbba2'

with open(json_path, 'r') as f:
    stations = json.load(f)

sql_statements = []
for s in stations:
    code = s['codigo']
    name = s['nombre'].replace("'", "''") # Escape quotes
    url = s['url_detalles']
    
    config = {
        "url": url,
        "token": "",
        "strategy": "REAL_IoT_WEBHOOK"
    }
    config_json = json.dumps(config).replace("'", "''")
    
    sql = f"""
    INSERT INTO sensors (id, name, type, strategy_type, location_km, is_active, created_at, twin_id, configuration)
    VALUES ('{code}', '{name}', 'UNKNOWN', 'REAL_IoT_WEBHOOK', 0.0, true, NOW(), '{twin_id}', '{config_json}'::jsonb)
    ON CONFLICT (id) DO UPDATE SET 
        configuration = '{config_json}'::jsonb,
        name = '{name}';
    """
    sql_statements.append(sql.strip())

# Write to file
with open('seed_sensors.sql', 'w') as f:
    f.write('\n'.join(sql_statements))

print(f"Generated {len(sql_statements)} SQL statements in seed_sensors.sql")
