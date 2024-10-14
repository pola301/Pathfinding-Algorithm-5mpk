import pandas as pd

# Load the original edges table
edges_df = pd.read_csv('walking_edges.csv')

# Group edges by start_node and end_node, then aggregate distance and route information
merged_edges_df = edges_df.groupby(['start_node', 'end_node']).agg({
    'distance': 'mean',
    'route_id': lambda x: ', '.join(set(x)),
    'route_short_name': lambda x: ', '.join(set(x))
}).reset_index()

# Save the merged edges table to a new CSV file
merged_edges_df.to_csv('merged_walking_edges_table.csv', index=False)
