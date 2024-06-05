import cudf
import cupy as cp
from scipy.spatial import cKDTree
import pandas as pd

from ukroutes.common.utils import Paths


def add_to_graph(df, nodes, edges, keep_col):
    nodes_tree = cKDTree(nodes[["easting", "northing"]].to_pandas().values)
    distances, indices = nodes_tree.query(df[["easting", "northing"]].values)

    # bug with cudf here
    nearest_nodes_df = cudf.from_pandas(
        pd.DataFrame(
            {
                keep_col: df[keep_col],
                "nearest_node": nodes.iloc[indices]["node_id"]
                .reset_index(drop=True)
                .to_numpy(),
                "distance": distances,
            }
        )
    )

    new_node_ids = cp.arange(len(nodes) + 1, len(nodes) + 1 + len(df))
    df["node_id"] = new_node_ids.get().astype(str)
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


if __name__ == "__main__":
    pcs = (
        pd.read_csv(Paths.RAW / "onspd" / "ONSPD_FEB_2024.csv")
        .rename(
            columns={"PCD": "postcode", "OSEAST1M": "easting", "OSNRTH1M": "northing"}
        )[["postcode", "easting", "northing"]]
        .dropna()
    )
    add_to_graph(pcs, "postcodes", "postcode")
