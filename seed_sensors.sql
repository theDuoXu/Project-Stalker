INSERT INTO sensors (id, name, type, strategy_type, location_km, is_active, created_at, twin_id, configuration)
    VALUES ('C322', 'SAICA CARCABOSO', 'UNKNOWN', 'REAL_IoT_WEBHOOK', 0.0, true, NOW(), '6bea03cf-3acb-488e-b93f-508424fcbba2', '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhewVO83ELxAsnz8kw3FvoHqOQ62PLmgAp%2Bd%2B0T1DPINAtkTWQ%2FEKSdOhRZDJmrPJTcQe3ajUdOMZzuxo21oa0lk%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb)
    ON CONFLICT (id) DO UPDATE SET 
        configuration = '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhewVO83ELxAsnz8kw3FvoHqOQ62PLmgAp%2Bd%2B0T1DPINAtkTWQ%2FEKSdOhRZDJmrPJTcQe3ajUdOMZzuxo21oa0lk%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb,
        name = 'SAICA CARCABOSO';
INSERT INTO sensors (id, name, type, strategy_type, location_km, is_active, created_at, twin_id, configuration)
    VALUES ('C323', 'SAICA BEJAR', 'UNKNOWN', 'REAL_IoT_WEBHOOK', 0.0, true, NOW(), '6bea03cf-3acb-488e-b93f-508424fcbba2', '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhV3RIftV%2FhjZx7UPHMHit585wNSb2thvsiymRFN0VDVh7cZeq0FZWgqW0UN8APSDV0RBcLOM4p2maA5OsDrUZjA%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb)
    ON CONFLICT (id) DO UPDATE SET 
        configuration = '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhV3RIftV%2FhjZx7UPHMHit585wNSb2thvsiymRFN0VDVh7cZeq0FZWgqW0UN8APSDV0RBcLOM4p2maA5OsDrUZjA%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb,
        name = 'SAICA BEJAR';
INSERT INTO sensors (id, name, type, strategy_type, location_km, is_active, created_at, twin_id, configuration)
    VALUES ('C313', 'SAICA CAZALEGAS', 'UNKNOWN', 'REAL_IoT_WEBHOOK', 0.0, true, NOW(), '6bea03cf-3acb-488e-b93f-508424fcbba2', '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhdqVqOP%2FWkj%2FzxQv2UeTOiXZieGVjc66HTyNAFp8A0cZaVqF1RQuBwYmCjtZxns02mM8K2P7rK8zs9lx59VLHfQ%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb)
    ON CONFLICT (id) DO UPDATE SET 
        configuration = '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhdqVqOP%2FWkj%2FzxQv2UeTOiXZieGVjc66HTyNAFp8A0cZaVqF1RQuBwYmCjtZxns02mM8K2P7rK8zs9lx59VLHfQ%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb,
        name = 'SAICA CAZALEGAS';
INSERT INTO sensors (id, name, type, strategy_type, location_km, is_active, created_at, twin_id, configuration)
    VALUES ('C326', 'SAICA ESCALONA', 'UNKNOWN', 'REAL_IoT_WEBHOOK', 0.0, true, NOW(), '6bea03cf-3acb-488e-b93f-508424fcbba2', '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhS8VBxWNndPtHh6UcpWMBbPFx2MrARPMEqeX%2BHc%2FJv3LrYTrT%2FlHXjQT%2FkHg0lBxYB%2BAF5l9z9YIqP%2BZn3gtxcU%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb)
    ON CONFLICT (id) DO UPDATE SET 
        configuration = '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhS8VBxWNndPtHh6UcpWMBbPFx2MrARPMEqeX%2BHc%2FJv3LrYTrT%2FlHXjQT%2FkHg0lBxYB%2BAF5l9z9YIqP%2BZn3gtxcU%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb,
        name = 'SAICA ESCALONA';
INSERT INTO sensors (id, name, type, strategy_type, location_km, is_active, created_at, twin_id, configuration)
    VALUES ('C342', 'SAICA PICADAS', 'UNKNOWN', 'REAL_IoT_WEBHOOK', 0.0, true, NOW(), '6bea03cf-3acb-488e-b93f-508424fcbba2', '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhYY7w5As90n7gHopIWz3mD49yWMyM9t%2BvwWCo8XDCuE3%2FrN7aReFxI8JIvGL1bibW8jcExmNZ7Igm39JgJRTdz4%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb)
    ON CONFLICT (id) DO UPDATE SET 
        configuration = '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhYY7w5As90n7gHopIWz3mD49yWMyM9t%2BvwWCo8XDCuE3%2FrN7aReFxI8JIvGL1bibW8jcExmNZ7Igm39JgJRTdz4%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb,
        name = 'SAICA PICADAS';
INSERT INTO sensors (id, name, type, strategy_type, location_km, is_active, created_at, twin_id, configuration)
    VALUES ('C317', 'SAICA ALMARAZ', 'UNKNOWN', 'REAL_IoT_WEBHOOK', 0.0, true, NOW(), '6bea03cf-3acb-488e-b93f-508424fcbba2', '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhaNk2IbwxE64W0om7V1ymnxECeb7DEKWlrmGuFpcr30jhOdxDUOpKYd9CidwuRVBzAHe%2FaN6jW4aGUYHD%2FZ0SYI%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb)
    ON CONFLICT (id) DO UPDATE SET 
        configuration = '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhaNk2IbwxE64W0om7V1ymnxECeb7DEKWlrmGuFpcr30jhOdxDUOpKYd9CidwuRVBzAHe%2FaN6jW4aGUYHD%2FZ0SYI%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb,
        name = 'SAICA ALMARAZ';
INSERT INTO sensors (id, name, type, strategy_type, location_km, is_active, created_at, twin_id, configuration)
    VALUES ('C331', 'SAICA CEDILLO', 'UNKNOWN', 'REAL_IoT_WEBHOOK', 0.0, true, NOW(), '6bea03cf-3acb-488e-b93f-508424fcbba2', '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhceuCc%2BXr1LzPSAXHQNvXsb5sveh1LozTpRc1fRvYHu0xpLCX%2FLwpoOPBV%2BByrDIkdZ3JRmQZL0nVK0yMQCyVl0%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb)
    ON CONFLICT (id) DO UPDATE SET 
        configuration = '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhceuCc%2BXr1LzPSAXHQNvXsb5sveh1LozTpRc1fRvYHu0xpLCX%2FLwpoOPBV%2BByrDIkdZ3JRmQZL0nVK0yMQCyVl0%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb,
        name = 'SAICA CEDILLO';
INSERT INTO sensors (id, name, type, strategy_type, location_km, is_active, created_at, twin_id, configuration)
    VALUES ('C343', 'SAICA VALDECAÑAS', 'UNKNOWN', 'REAL_IoT_WEBHOOK', 0.0, true, NOW(), '6bea03cf-3acb-488e-b93f-508424fcbba2', '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhdkW8EvY5ZUHu6GNnGGc%2FhTVUxcCyRt%2BOHIX4yxJKuTs50B0vT25%2BIwHLXB0IStGaq9zyNo2DFftcDZeE9OU6gI%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb)
    ON CONFLICT (id) DO UPDATE SET 
        configuration = '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhdkW8EvY5ZUHu6GNnGGc%2FhTVUxcCyRt%2BOHIX4yxJKuTs50B0vT25%2BIwHLXB0IStGaq9zyNo2DFftcDZeE9OU6gI%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb,
        name = 'SAICA VALDECAÑAS';
INSERT INTO sensors (id, name, type, strategy_type, location_km, is_active, created_at, twin_id, configuration)
    VALUES ('C345', 'SAICA TORREJON-TAJO', 'UNKNOWN', 'REAL_IoT_WEBHOOK', 0.0, true, NOW(), '6bea03cf-3acb-488e-b93f-508424fcbba2', '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2Fyxpbhc%2BGjsF6zFnufOYinWO2jYRhMEp%2FEZlAmUlA55Og6TwO9hnA1iMfYvATsxfc9P7mvUb%2BN5OexcWo5hw52bpnWqk%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb)
    ON CONFLICT (id) DO UPDATE SET 
        configuration = '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2Fyxpbhc%2BGjsF6zFnufOYinWO2jYRhMEp%2FEZlAmUlA55Og6TwO9hnA1iMfYvATsxfc9P7mvUb%2BN5OexcWo5hw52bpnWqk%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb,
        name = 'SAICA TORREJON-TAJO';
INSERT INTO sensors (id, name, type, strategy_type, location_km, is_active, created_at, twin_id, configuration)
    VALUES ('C302', 'SAICA ARANJUEZ', 'UNKNOWN', 'REAL_IoT_WEBHOOK', 0.0, true, NOW(), '6bea03cf-3acb-488e-b93f-508424fcbba2', '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhZDfgASxhWlaox8EvBWYvNkEDCxrAhGJXlgnFq8ROwp7AJEbIB6erho%2BHgiyGNl4RHVYmypFJVoD%2BJqSACQ8u7Y%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb)
    ON CONFLICT (id) DO UPDATE SET 
        configuration = '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhZDfgASxhWlaox8EvBWYvNkEDCxrAhGJXlgnFq8ROwp7AJEbIB6erho%2BHgiyGNl4RHVYmypFJVoD%2BJqSACQ8u7Y%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb,
        name = 'SAICA ARANJUEZ';
INSERT INTO sensors (id, name, type, strategy_type, location_km, is_active, created_at, twin_id, configuration)
    VALUES ('C312', 'SAICA VILLARRUBIA', 'UNKNOWN', 'REAL_IoT_WEBHOOK', 0.0, true, NOW(), '6bea03cf-3acb-488e-b93f-508424fcbba2', '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhWMhzvOqw9mo2ZpV9WW9c9j%2BO6musN1xCGd0LjaULqxYeEkTY%2F5CFWlTL%2BvHekzEZ7k6gcZHnaJaTZ66N0q1iYM%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb)
    ON CONFLICT (id) DO UPDATE SET 
        configuration = '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhWMhzvOqw9mo2ZpV9WW9c9j%2BO6musN1xCGd0LjaULqxYeEkTY%2F5CFWlTL%2BvHekzEZ7k6gcZHnaJaTZ66N0q1iYM%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb,
        name = 'SAICA VILLARRUBIA';
INSERT INTO sensors (id, name, type, strategy_type, location_km, is_active, created_at, twin_id, configuration)
    VALUES ('C316', 'SAICA TRILLO', 'UNKNOWN', 'REAL_IoT_WEBHOOK', 0.0, true, NOW(), '6bea03cf-3acb-488e-b93f-508424fcbba2', '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhXYhH0hSJErEvEovW%2F%2BfAVl8cIYESIiCnUCZieD9YaKCPQWDhA%2BsSZu%2FhrXK%2FQXsE9K8QaKUgqCca%2Fa2JSTT65g%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb)
    ON CONFLICT (id) DO UPDATE SET 
        configuration = '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhXYhH0hSJErEvEovW%2F%2BfAVl8cIYESIiCnUCZieD9YaKCPQWDhA%2BsSZu%2FhrXK%2FQXsE9K8QaKUgqCca%2Fa2JSTT65g%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb,
        name = 'SAICA TRILLO';
INSERT INTO sensors (id, name, type, strategy_type, location_km, is_active, created_at, twin_id, configuration)
    VALUES ('C320', 'SAICA ZORITA', 'UNKNOWN', 'REAL_IoT_WEBHOOK', 0.0, true, NOW(), '6bea03cf-3acb-488e-b93f-508424fcbba2', '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhVUxj0fekq6FiJI1n7v7CdZHbidvEJNeAOELlAJM9SG12FKSA8%2BA3maONPri6HcfxZ6B0VLRzuWZOBgC%2BU2WMNA%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb)
    ON CONFLICT (id) DO UPDATE SET 
        configuration = '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhVUxj0fekq6FiJI1n7v7CdZHbidvEJNeAOELlAJM9SG12FKSA8%2BA3maONPri6HcfxZ6B0VLRzuWZOBgC%2BU2WMNA%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb,
        name = 'SAICA ZORITA';
INSERT INTO sensors (id, name, type, strategy_type, location_km, is_active, created_at, twin_id, configuration)
    VALUES ('C307', 'SAICA SANTOS DE LA HUMOSA', 'UNKNOWN', 'REAL_IoT_WEBHOOK', 0.0, true, NOW(), '6bea03cf-3acb-488e-b93f-508424fcbba2', '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhfJlDCs8SDUITGcGFJWdkjcANzXyxc3pnQhGk59%2F1VgpGNPAoHTJjHtkucuhSbZSaiHcMYd0mDnZ38%2FKN1QOvY8%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb)
    ON CONFLICT (id) DO UPDATE SET 
        configuration = '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhfJlDCs8SDUITGcGFJWdkjcANzXyxc3pnQhGk59%2F1VgpGNPAoHTJjHtkucuhSbZSaiHcMYd0mDnZ38%2FKN1QOvY8%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb,
        name = 'SAICA SANTOS DE LA HUMOSA';
INSERT INTO sensors (id, name, type, strategy_type, location_km, is_active, created_at, twin_id, configuration)
    VALUES ('C309', 'SAICA ESPINILLOS', 'UNKNOWN', 'REAL_IoT_WEBHOOK', 0.0, true, NOW(), '6bea03cf-3acb-488e-b93f-508424fcbba2', '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhdVlWFCBwPn05gQkieJkfMoxcycQG%2F3oguvVVGijtZJMNiUlfpyZzV7u32bqlERD7%2FDM1cv%2BWepCJhBhoPnjUyU%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb)
    ON CONFLICT (id) DO UPDATE SET 
        configuration = '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhdVlWFCBwPn05gQkieJkfMoxcycQG%2F3oguvVVGijtZJMNiUlfpyZzV7u32bqlERD7%2FDM1cv%2BWepCJhBhoPnjUyU%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb,
        name = 'SAICA ESPINILLOS';
INSERT INTO sensors (id, name, type, strategy_type, location_km, is_active, created_at, twin_id, configuration)
    VALUES ('C303', 'SAICA LAS NIEVES', 'UNKNOWN', 'REAL_IoT_WEBHOOK', 0.0, true, NOW(), '6bea03cf-3acb-488e-b93f-508424fcbba2', '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhRH1PQL8jF2UQuhaKkgsw%2BeopsXsrvYqvawtl6c216NIavszjzEKoHYeSc3I8PbnDFjRjU%2Fb2rFL48MMVxbP5KE%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb)
    ON CONFLICT (id) DO UPDATE SET 
        configuration = '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhRH1PQL8jF2UQuhaKkgsw%2BeopsXsrvYqvawtl6c216NIavszjzEKoHYeSc3I8PbnDFjRjU%2Fb2rFL48MMVxbP5KE%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb,
        name = 'SAICA LAS NIEVES';
INSERT INTO sensors (id, name, type, strategy_type, location_km, is_active, created_at, twin_id, configuration)
    VALUES ('C304', 'SAICA PUENTE LARGO', 'UNKNOWN', 'REAL_IoT_WEBHOOK', 0.0, true, NOW(), '6bea03cf-3acb-488e-b93f-508424fcbba2', '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2Fyxpbhc4c7bOqPAdeKSYfQgjZwd6qu4SaSzKFd%2BOgjR0bBNHf4NhVIU3QRuS54facPPpymHcLuSGTZIe5Bk%2BCanXzJI0%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb)
    ON CONFLICT (id) DO UPDATE SET 
        configuration = '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2Fyxpbhc4c7bOqPAdeKSYfQgjZwd6qu4SaSzKFd%2BOgjR0bBNHf4NhVIU3QRuS54facPPpymHcLuSGTZIe5Bk%2BCanXzJI0%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb,
        name = 'SAICA PUENTE LARGO';
INSERT INTO sensors (id, name, type, strategy_type, location_km, is_active, created_at, twin_id, configuration)
    VALUES ('C306', 'SAICA RIVAS', 'UNKNOWN', 'REAL_IoT_WEBHOOK', 0.0, true, NOW(), '6bea03cf-3acb-488e-b93f-508424fcbba2', '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhcYWKhFTX4pXb%2BymJc0VQOa2FcyXVuCUXeezA1lQfq68i97SnJwKPy2C3CJajgqBECxfMjMGFL4GaZiuy7Vh5Ug%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb)
    ON CONFLICT (id) DO UPDATE SET 
        configuration = '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhcYWKhFTX4pXb%2BymJc0VQOa2FcyXVuCUXeezA1lQfq68i97SnJwKPy2C3CJajgqBECxfMjMGFL4GaZiuy7Vh5Ug%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb,
        name = 'SAICA RIVAS';
INSERT INTO sensors (id, name, type, strategy_type, location_km, is_active, created_at, twin_id, configuration)
    VALUES ('C308', 'SAICA BATRES', 'UNKNOWN', 'REAL_IoT_WEBHOOK', 0.0, true, NOW(), '6bea03cf-3acb-488e-b93f-508424fcbba2', '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhSWkhYFRyc4n0W6hr3XVyhgxgnuMAshOly7%2FWpIZS0wtQ7DfZJySSIEQbnxdff0eqywuYLUjm832Ivxtjj1t4ms%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb)
    ON CONFLICT (id) DO UPDATE SET 
        configuration = '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhSWkhYFRyc4n0W6hr3XVyhgxgnuMAshOly7%2FWpIZS0wtQ7DfZJySSIEQbnxdff0eqywuYLUjm832Ivxtjj1t4ms%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb,
        name = 'SAICA BATRES';
INSERT INTO sensors (id, name, type, strategy_type, location_km, is_active, created_at, twin_id, configuration)
    VALUES ('C314', 'SAICA BARGAS', 'UNKNOWN', 'REAL_IoT_WEBHOOK', 0.0, true, NOW(), '6bea03cf-3acb-488e-b93f-508424fcbba2', '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhYYwxFS1R77pEZvH8BC7zlbh8nTQKc72RKMqPCu%2FDZqQZipWzOe4FeV40QE6Q%2Fc8weAStB6AAGxK24Pow%2FC6LP0%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb)
    ON CONFLICT (id) DO UPDATE SET 
        configuration = '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhYYwxFS1R77pEZvH8BC7zlbh8nTQKc72RKMqPCu%2FDZqQZipWzOe4FeV40QE6Q%2Fc8weAStB6AAGxK24Pow%2FC6LP0%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb,
        name = 'SAICA BARGAS';
INSERT INTO sensors (id, name, type, strategy_type, location_km, is_active, created_at, twin_id, configuration)
    VALUES ('C327', 'SAICA PRESA DEL REY', 'UNKNOWN', 'REAL_IoT_WEBHOOK', 0.0, true, NOW(), '6bea03cf-3acb-488e-b93f-508424fcbba2', '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2Fyxpbha9hf7myJdOnf%2FBSi%2BHnEMVfXeVVqNbhggNqUHsiFpPXFHJa5%2FhmcQhN56sahRPIJlS2XU2VdiDbMDein397was%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb)
    ON CONFLICT (id) DO UPDATE SET 
        configuration = '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2Fyxpbha9hf7myJdOnf%2FBSi%2BHnEMVfXeVVqNbhggNqUHsiFpPXFHJa5%2FhmcQhN56sahRPIJlS2XU2VdiDbMDein397was%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb,
        name = 'SAICA PRESA DEL REY';
INSERT INTO sensors (id, name, type, strategy_type, location_km, is_active, created_at, twin_id, configuration)
    VALUES ('C328', 'SAICA MEJORADA', 'UNKNOWN', 'REAL_IoT_WEBHOOK', 0.0, true, NOW(), '6bea03cf-3acb-488e-b93f-508424fcbba2', '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhSB9hTjLQaEmG%2B1huOL0iw1pMua1cyIISksDhYEs2TrwxZl4juTF1U1RYeMnQ1iT912wH%2FPfdOpt%2FgyQYq9m6%2Bc%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb)
    ON CONFLICT (id) DO UPDATE SET 
        configuration = '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhSB9hTjLQaEmG%2B1huOL0iw1pMua1cyIISksDhYEs2TrwxZl4juTF1U1RYeMnQ1iT912wH%2FPfdOpt%2FgyQYq9m6%2Bc%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb,
        name = 'SAICA MEJORADA';
INSERT INTO sensors (id, name, type, strategy_type, location_km, is_active, created_at, twin_id, configuration)
    VALUES ('C330', 'SAICA ALGETE', 'UNKNOWN', 'REAL_IoT_WEBHOOK', 0.0, true, NOW(), '6bea03cf-3acb-488e-b93f-508424fcbba2', '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2Fyxpbhe1qZ%2FaOKvBJ%2Fkozd9Ca5Za%2FsuDYbQN1Dg3OcZb5BDZ93Pj0suw7oRVup5kZ53D09AsgSK%2FGwypC%2FUbNYJX6HCA%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb)
    ON CONFLICT (id) DO UPDATE SET 
        configuration = '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2Fyxpbhe1qZ%2FaOKvBJ%2Fkozd9Ca5Za%2FsuDYbQN1Dg3OcZb5BDZ93Pj0suw7oRVup5kZ53D09AsgSK%2FGwypC%2FUbNYJX6HCA%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb,
        name = 'SAICA ALGETE';
INSERT INTO sensors (id, name, type, strategy_type, location_km, is_active, created_at, twin_id, configuration)
    VALUES ('C310', 'SAICA TALAVERA', 'UNKNOWN', 'REAL_IoT_WEBHOOK', 0.0, true, NOW(), '6bea03cf-3acb-488e-b93f-508424fcbba2', '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2Fyxpbhd3cvpVg6PMY64pdOsKx3sihm9LtkoMrzTkwGGoSEEoOPovMtGiL6xyd2xBb61fHw%2FbX0Z8wyt73xFaQtFf4PxM%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb)
    ON CONFLICT (id) DO UPDATE SET 
        configuration = '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2Fyxpbhd3cvpVg6PMY64pdOsKx3sihm9LtkoMrzTkwGGoSEEoOPovMtGiL6xyd2xBb61fHw%2FbX0Z8wyt73xFaQtFf4PxM%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb,
        name = 'SAICA TALAVERA';
INSERT INTO sensors (id, name, type, strategy_type, location_km, is_active, created_at, twin_id, configuration)
    VALUES ('C315', 'SAICA SAFONT', 'UNKNOWN', 'REAL_IoT_WEBHOOK', 0.0, true, NOW(), '6bea03cf-3acb-488e-b93f-508424fcbba2', '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhVy7udRY6miHsZ9hfEme7txKxzz%2FIuhhySrRRSxUDBiweiAGxjm3tQtpgWHKLu0U9O8T3eViwrxOKa5U%2BimeZ7s%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb)
    ON CONFLICT (id) DO UPDATE SET 
        configuration = '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhVy7udRY6miHsZ9hfEme7txKxzz%2FIuhhySrRRSxUDBiweiAGxjm3tQtpgWHKLu0U9O8T3eViwrxOKa5U%2BimeZ7s%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb,
        name = 'SAICA SAFONT';
INSERT INTO sensors (id, name, type, strategy_type, location_km, is_active, created_at, twin_id, configuration)
    VALUES ('C329', 'SAICA TITULCIA', 'UNKNOWN', 'REAL_IoT_WEBHOOK', 0.0, true, NOW(), '6bea03cf-3acb-488e-b93f-508424fcbba2', '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhebKsoNavb3uMYsa9Ok2QfZw6JcmHXHnu2WT627%2B1VgmTNNe5CAHeJ%2FSMYrcQMNc57L8Be2tQ04%2BAXYNEnjvcFs%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb)
    ON CONFLICT (id) DO UPDATE SET 
        configuration = '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhebKsoNavb3uMYsa9Ok2QfZw6JcmHXHnu2WT627%2B1VgmTNNe5CAHeJ%2FSMYrcQMNc57L8Be2tQ04%2BAXYNEnjvcFs%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb,
        name = 'SAICA TITULCIA';
INSERT INTO sensors (id, name, type, strategy_type, location_km, is_active, created_at, twin_id, configuration)
    VALUES ('C333', 'SAICA ESTIVIEL', 'UNKNOWN', 'REAL_IoT_WEBHOOK', 0.0, true, NOW(), '6bea03cf-3acb-488e-b93f-508424fcbba2', '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhRVdeaerVA14scfSRt8BRL8MtJSqqsbofGjrz9WyP6LA9MeYGczBTJjv%2F18OKcTA9Wz5CkwLFCWshmrG27PdofM%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb)
    ON CONFLICT (id) DO UPDATE SET 
        configuration = '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhRVdeaerVA14scfSRt8BRL8MtJSqqsbofGjrz9WyP6LA9MeYGczBTJjv%2F18OKcTA9Wz5CkwLFCWshmrG27PdofM%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb,
        name = 'SAICA ESTIVIEL';
INSERT INTO sensors (id, name, type, strategy_type, location_km, is_active, created_at, twin_id, configuration)
    VALUES ('C344', 'SAICA AZUTAN', 'UNKNOWN', 'REAL_IoT_WEBHOOK', 0.0, true, NOW(), '6bea03cf-3acb-488e-b93f-508424fcbba2', '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhYK%2Br7zXZK0mT28b2LMi%2FuhdDCYYnQ4HvTZA3iAWZAzu%2F0jYTIkZyMRk083B6uTb%2B7%2Fa%2Bdj1aOoxe031Gee5DG0%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb)
    ON CONFLICT (id) DO UPDATE SET 
        configuration = '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhYK%2Br7zXZK0mT28b2LMi%2FuhdDCYYnQ4HvTZA3iAWZAzu%2F0jYTIkZyMRk083B6uTb%2B7%2Fa%2Bdj1aOoxe031Gee5DG0%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb,
        name = 'SAICA AZUTAN';
INSERT INTO sensors (id, name, type, strategy_type, location_km, is_active, created_at, twin_id, configuration)
    VALUES ('C325', 'SAICA MONFRAGÜE', 'UNKNOWN', 'REAL_IoT_WEBHOOK', 0.0, true, NOW(), '6bea03cf-3acb-488e-b93f-508424fcbba2', '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhYbxTxl1y2HsBUTmuWKKssWkaflk0M%2B57WFvLAnMC8oqt%2B0NzmVQqQ52o5AJ50MvMCKW4KIS9kgCdOewPgZihoU%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb)
    ON CONFLICT (id) DO UPDATE SET 
        configuration = '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhYbxTxl1y2HsBUTmuWKKssWkaflk0M%2B57WFvLAnMC8oqt%2B0NzmVQqQ52o5AJ50MvMCKW4KIS9kgCdOewPgZihoU%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb,
        name = 'SAICA MONFRAGÜE';
INSERT INTO sensors (id, name, type, strategy_type, location_km, is_active, created_at, twin_id, configuration)
    VALUES ('C321', 'SAICA RIVERA DE GATA', 'UNKNOWN', 'REAL_IoT_WEBHOOK', 0.0, true, NOW(), '6bea03cf-3acb-488e-b93f-508424fcbba2', '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhSjDiMJQ%2Fvedv1X69%2Fa6MbSiiToGrW8WfM0iiH76%2BmzdiYReXcPcGpDlJcLJj%2Fh09bUAX9eKgNYHnCU43GMiBDg%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb)
    ON CONFLICT (id) DO UPDATE SET 
        configuration = '{"url": "https://saihtajo.chtajo.es/index.php?w=get-estacion&x=%2F4ZGs%2B6M%2BZWfvq6%2FyxpbhSjDiMJQ%2Fvedv1X69%2Fa6MbSiiToGrW8WfM0iiH76%2BmzdiYReXcPcGpDlJcLJj%2Fh09bUAX9eKgNYHnCU43GMiBDg%3D", "token": "", "strategy": "REAL_IoT_WEBHOOK"}'::jsonb,
        name = 'SAICA RIVERA DE GATA';