import pandas as pd
from math import radians, sin, cos, sqrt, atan2

# Function to calculate the Haversine distance between two points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Radius of the Earth in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 8* atan2(sqrt(a), sqrt(1 - a))
    distance = R * c * 1000  # Convert distance to meters
    return distance

# Load the stops data (assuming 'stop_id', 'stop_latitude', 'stop_longitude' columns)
nodes_df = pd.read_csv('F:/College/el big one/V11/nodes_table.csv')

# Create a new DataFrame to store the walking connections
walking_edges = []

# Iterate through each pair of stops
for i, stop1 in nodes_df.iterrows():
    for j, stop2 in nodes_df.iterrows():
        if i != j:  # Avoid self-connections
            distance = haversine(stop1['stop_latitude'], stop1['stop_longitude'], stop2['stop_latitude'], stop2['stop_longitude'])
            
            # If the distance is less than 400 meters, create a walking connection
            if distance < 400:
                walking_edges.append({
                    'start_node': stop1['stop_id'],
                    'end_node': stop2['stop_id'],
                    'distance': distance,
                    'route_id': 'walking'
                })

# Convert walking edges to DataFrame
walking_edges_df = pd.DataFrame(walking_edges)

# Save walking edges to a new CSV
walking_edges_df.to_csv('F:/College/el big one/V11/walking_edges.csv', index=False)

print(f"Added {len(walking_edges)} walking connections between nearby stops.")