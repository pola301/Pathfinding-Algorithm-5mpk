import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Load edges_table.csv and routes.csv
edges_table = pd.read_csv('edges_table.csv')
routes = pd.read_csv('updated_routes.csv')

# Create a dictionary mapping route_id to bus
route_bus_map = dict(zip(routes['route_id'], routes['route_short_name']))

def add_bus_column(row):
    # Add 'bus' column based on route_id
    if row['route_id'] in route_bus_map:
        return route_bus_map[row['route_id']]
    else:
        return None

# Add 'bus' column using parallelization and progress bar
with ThreadPoolExecutor() as executor, tqdm(total=len(edges_table)) as pbar:
    futures = []
    for _, row in edges_table.iterrows():
        future = executor.submit(add_bus_column, row)
        future.add_done_callback(lambda p: pbar.update())
        futures.append(future)
    results = [future.result() for future in futures]

edges_table['route_short_name'] = results

# Save the updated edges_table.csv
edges_table.to_csv('edges_table_with_route_short_name.csv', index=False)
