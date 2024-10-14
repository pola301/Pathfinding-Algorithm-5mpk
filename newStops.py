import pandas as pd
from geopy.distance import geodesic

# Load the CSV file into a DataFrame
df = pd.read_csv('updated_stops.csv')

# Function to calculate distance between two points
def calculate_distance(lat1, lon1, lat2, lon2):
    coords_1 = (lat1, lon1)
    coords_2 = (lat2, lon2)
    return geodesic(coords_1, coords_2).kilometers

# Iterate through each pair of stops and merge if distance is less than 0.019 km
merged_stops = []
for i in range(len(df)):
    for j in range(i+1, len(df)):
        stop1 = df.iloc[i]
        stop2 = df.iloc[j]
        distance = calculate_distance(stop1['stop_latitude'], stop1['stop_longitude'], stop2['stop_latitude'], stop2['stop_longitude'])
        if distance < 0.019:
            merged_stops.append((stop1['stop_id'], stop2['stop_id']))

# Merge stops
for stop1, stop2 in merged_stops:
    df.loc[df['stop_id'] == stop2, 'stop_id'] = stop1

# Drop duplicate stop_ids
df.drop_duplicates(subset=['stop_id'], keep='first', inplace=True)

# Save the merged DataFrame to a new CSV file
df.to_csv('merged_stops.csv', index=False)
