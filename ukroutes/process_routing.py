import numpy as np
import pandas as pd
from scipy.spatial import KDTree


def add_to_graph(df, nodes, edges, weights, k=10):
    nodes_tree = KDTree(nodes[["easting", "northing"]].values)
    distances, indices = nodes_tree.query(df[["easting", "northing"]].values, k=k)

    nearest_nodes_df = pd.DataFrame(
        {
            "nearest_node": nodes.iloc[indices.flatten()]["node_id"]
            .reset_index(drop=True)
            .to_numpy(),
            "distance": distances.flatten() + 0.01,
        }
    )

    new_node_ids = np.arange(len(nodes) + 1, len(nodes) + 1 + len(df))
    df["node_id"] = new_node_ids
    new_nodes = df[["node_id", "easting", "northing"]]
    nodes = pd.concat([nodes, new_nodes])
    new_edges = pd.DataFrame(
        {
            "start_node": df.loc[np.repeat(df.index, k)].reset_index(drop=True)[
                "node_id"
            ],
            "end_node": nearest_nodes_df["nearest_node"],
            "length": nearest_nodes_df["distance"],
        }
    )
    if weights == "time_weighted":
        new_edges[weights] = (
            (new_edges["length"].astype(float) / 1000) / 25 * 1.609344 * 60
        )
    elif weights == "pedestrian_time":
        new_edges[weights] = (new_edges["length"].astype(float) / 1000) / 5 * 60
    edges = pd.concat([edges, new_edges])

    return (
        df.reset_index(drop=True),
        nodes.reset_index(drop=True),
        edges.reset_index(drop=True),
    )
