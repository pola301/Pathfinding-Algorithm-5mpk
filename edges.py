import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

nodes_table = pd.read_csv('nodes_table.csv')
stop_times_with_route_id = pd.read_csv('stop_times_with_route_id.csv')

def haversine(lat1, lon1, lat2, lon2):
    import math

    R = 6371  # Radius of the Earth in kilometers
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    a = math.sin(dLat / 2) * math.sin(dLat / 2) + \
        math.sin(dLon / 2) * math.sin(dLon / 2) * math.cos(lat1) * math.cos(lat2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c

    return round(distance*1000)

def create_edges(nodes_table, stop_times_with_route_id):
    edges = []

    for i in range(len(stop_times_with_route_id) - 1):
        start_stop = nodes_table[nodes_table['stop_id'] == stop_times_with_route_id.iloc[i]['stop_id']]
        end_stop = nodes_table[nodes_table['stop_id'] == stop_times_with_route_id.iloc[i + 1]['stop_id']]

        if not start_stop.empty and not end_stop.empty:
            distance = haversine(start_stop['stop_latitude'].values[0], start_stop['stop_longitude'].values[0],
                                 end_stop['stop_latitude'].values[0], end_stop['stop_longitude'].values[0])
            edges.append((start_stop['stop_id'].values[0], end_stop['stop_id'].values[0], distance,
                          stop_times_with_route_id.iloc[i]['route_id']))

    return edges

num_cores = 4  # Number of CPU cores to use for parallelization
edges_list = Parallel(n_jobs=num_cores)(
    delayed(create_edges)(nodes_table, group)
    for _, group in tqdm(stop_times_with_route_id.groupby('trip_id'))
)

# Concatenate the edges from all trips into a single list
edges = [edge for sublist in edges_list for edge in sublist]

edges_df = pd.DataFrame(edges, columns=['start_node', 'end_node', 'distance', 'route_id'])
edges_df.to_csv('edges_table.csv', index=False)
