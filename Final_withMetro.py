import pandas as pd
import networkx as nx
from networkx.exception import NetworkXNoPath
from math import radians, sin, cos, sqrt, atan2
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import pickle
import time

# Load data into pandas DataFrames
nodes_df = pd.read_csv('nodes_table.csv')
edges_df = pd.read_csv('merged_edges_table.csv')
routes_with_stops_df = pd.read_csv('routes_with_stops.csv')

# Preprocess routes_with_stops_df to create a dictionary mapping stop_id to route_ids
stop_routes_dict = defaultdict(list)
for _, row in routes_with_stops_df.iterrows():
    route_id = row['route_id']
    stops = row['stops'].split(', ')
    for stop_id in stops:
        stop_routes_dict[stop_id].append(route_id)

# Check if the graph object is saved
try:
    with open('graph.pkl', 'rb') as file:
        G = pickle.load(file)
except FileNotFoundError:
    # Create a directed graph
    G = nx.Graph()

    # Add nodes to the graph
    for _, row in nodes_df.iterrows():
        G.add_node(row['stop_id'], stop_id=row['stop_id'], stop_name=row['stop_name'], stop_latitude=row['stop_latitude'], stop_longitude=row['stop_longitude'])

    # Add edges to the graph
    for _, row in edges_df.iterrows():
        G.add_edge(row['start_node'], row['end_node'], distance=row['distance'], route_id=row['route_id'])

    # Save the graph object
    with open('graph.pkl', 'wb') as file:
        pickle.dump(G, file)

# Function to calculate the distance between two points using the Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Radius of the Earth in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

# Function to find the closest node to a given coordinate
def find_closest_node(coords):
    min_distance = float('inf')
    closest_node = None
    for node_id, node_data in G.nodes(data=True):
        distance = haversine(coords[0], coords[1], node_data['stop_latitude'], node_data['stop_longitude'])
        if distance < min_distance:
            min_distance = distance
            closest_node = node_id
    return closest_node

# Heuristics function for the shortest path A* algorithm
def heuristic(u, v, dest_node):
    # Define a heuristic function to estimate the remaining distance between two nodes
    # Penalize route changes and prioritize edges that lead towards the destination
    edge_data = G.get_edge_data(u, v)
    if edge_data:
        route_id = edge_data.get('route_id')
        if route_id and G.edges[u, v]['route_id'] != route_id:
            # Calculate the distance from node v to the destination
            dest_lat, dest_lon = G.nodes[dest_node]['stop_latitude'], G.nodes[dest_node]['stop_longitude']
            v_lat, v_lon = G.nodes[v]['stop_latitude'], G.nodes[v]['stop_longitude']
            dist_to_dest = haversine(v_lat, v_lon, dest_lat, dest_lon)
            return 10000 + dist_to_dest  # High cost for route changes, plus distance to destination
    return 0  # Default cost if no route change

# Shortest path algorithm uses the heurisitcs and A* path finding algortihm
def shortest_path(start_node, end_node):
    path_nodes = nx.astar_path(G, start_node, end_node, heuristic=lambda u, v: heuristic(u, v, end_node), weight='distance')
    path = []
    for i in range(len(path_nodes) - 1):
        u = path_nodes[i]
        v = path_nodes[i + 1]
        edge_data = G.get_edge_data(u, v)
        if edge_data:
            path.append((G.nodes[v]['stop_id'], G.nodes[v]['stop_name'], G.nodes[v]['stop_latitude'], G.nodes[v]['stop_longitude']))
    return path

def find_route_for_path(path):
    path_with_routes = []
    for node in path:
        node_id, stop_name, stop_lat, stop_lon = node[:4]
        route_ids = set(stop_routes_dict.get(node_id, []))
        route_names = set(routes_with_stops_df.loc[routes_with_stops_df['route_id'].isin(route_ids), 'route_short_name'])
        path_with_routes.append((node_id, stop_name, stop_lat, stop_lon, *route_ids, *route_names))

    return path_with_routes


def count_nodes_covered(route_id, remaining_path):
    # Get the stops covered by the given route
    covered_stops = set(routes_with_stops_df[routes_with_stops_df['route_id'] == route_id]['stops'].str.split(', ').explode().unique())
    # Count the number of stops in the remaining path covered by the route
    return sum(1 for stop in remaining_path if stop[0] in covered_stops)


def route_exists(start_node, end_node):
    start_node_routes = stop_routes_dict.get(start_node, [])
    end_node_routes = stop_routes_dict.get(end_node, [])
    common_routes = set(start_node_routes) & set(end_node_routes)
    if common_routes:
        route_id = common_routes.pop()
        route_short_name = routes_with_stops_df[routes_with_stops_df['route_id'] == route_id]['route_short_name'].values[0]
        return route_id, route_short_name
    return None, None

# Your existing Python code here
def find_shortest_path(start_coords, end_coords):
    try:
        # Find the closest nodes to the start and end coordinates
        start_node = find_closest_node(start_coords)
        end_node = find_closest_node(end_coords)
        
        # Check if there's a valid path between the nodes
        if not nx.has_path(G, start_node, end_node):
            return f"No path exists between {start_node} and {end_node}."

        # Calculate the shortest path using the A* algorithm
        output_path = shortest_path(start_node, end_node)
        return output_path
    
    except NetworkXNoPath:
        return f"No path found between stop {start_node} and stop {end_node}."

# Example usage
start_coords = (30.066556, 31.21302) # Current location coordinates
end_coords = (30.010576,31.171602)# Destination coordinates

start_time = time.time()

print(find_shortest_path(start_coords,end_coords))

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")