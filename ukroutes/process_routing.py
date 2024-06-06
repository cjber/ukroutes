import cudf
import cupy as cp
import pandas as pd
from scipy.spatial import cKDTree


def add_to_graph(df, nodes, edges, keep_col, topk=1):
    nodes_tree = cKDTree(nodes[["easting", "northing"]].values.get())
    distances, indices = nodes_tree.query(df[["easting", "northing"]].values, k=topk)
    if topk > 1:
        indices_1 = indices[:, 0]
        distances_1 = distances[:, 0]
    else:
        indices_1 = indices
        distances_1 = distances

    nearest_nodes_df = pd.DataFrame(
        {
            keep_col: df[keep_col],
            "nearest_node": nodes.iloc[indices_1]["node_id"]
            .reset_index(drop=True)
            .to_numpy(),
            "distance": distances_1,
        }
    )

    df["top_10_nodes"] = (
        nodes.iloc[indices.flatten()]["node_id"].values.reshape(-1, topk).tolist()
    )
    new_node_ids = cp.arange(len(nodes) + 1, len(nodes) + 1 + len(df))
    df["node_id"] = new_node_ids.get()
    new_nodes = df[["node_id", "easting", "northing"]]
    nodes = cudf.concat([nodes, cudf.from_pandas(new_nodes)])

    new_edges = cudf.DataFrame(
        {
            "start_node": df["node_id"],
            "end_node": nearest_nodes_df["nearest_node"],
            "length": nearest_nodes_df["distance"],
        }
    )
    edges = cudf.concat([edges, new_edges])
    edges["time_weighted"] = (edges["length"].astype(float) / 1000) / 25 * 1.609344 * 60

    return df, nodes, edges
