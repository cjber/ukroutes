import cudf
import geopandas as gpd
import pandas as pd
from cuml.neighbors.nearest_neighbors import NearestNeighbors
from tqdm import tqdm

from ukroutes.common.logger import logger
from ukroutes.common.utils import Paths


def nearest_nodes(df: cudf.DataFrame, nodes: cudf.DataFrame) -> cudf.DataFrame:
    """
    Find nearest road node to point of interest

    Uses `cuml` nearest neighbours for GPU accelerated nearest points.
    It is assumed that all points use a planar coordinate system like BNG.

    Parameters
    ----------
    df : cudf.DataFrame
        POI df containing coordinate information
    nodes : cudf.DataFrame
        Road nodes with coordinate information

    Returns
    -------
    cudf.DataFrame:
        Road nodes that are nearest neighbour to some POI
    """
    nbrs = NearestNeighbors(n_neighbors=1, output_type="cudf", algorithm="brute").fit(
        nodes[["easting", "northing"]]
    )
    _, indices = nbrs.kneighbors(df[["easting", "northing"]])
    df = df.assign(
        node_id=nodes.iloc[indices]["node_id"].reset_index(drop=True).values.get()
    )
    return df


def get_buffers(
    df: cudf.DataFrame,
    postcodes: cudf.DataFrame,
    k: int,
) -> cudf.DataFrame:
    """
    Estimate buffer sizes required to capture each necessary road node
    Calculates k nearest neighbours for each POI to each road node. Finds
    each node that is considered a neighbour to a poi `k*len(poi)`. Buffers
    are taken as the distance to the further neighbour and all nodes associated with
    each POI are saved.
    Parameters
    ----------
    poi : cudf.DataFrame
        Dataframe of all POIs
    postcodes : cudf.DataFrame
        Dataframe of postcodes
    k : int
        Number of neigbours to use
    Returns
    -------
    cudf.DataFrame:
        POI dataframe including buffer and column with list of nodes
    """
    nbrs = NearestNeighbors(n_neighbors=k, output_type="cudf", algorithm="brute").fit(
        df[["easting", "northing"]]
    )
    distances, indices = nbrs.kneighbors(postcodes[["easting", "northing"]])

    poi_nn = (
        postcodes.join(indices)[["node_id"] + indices.columns.tolist()]
        .set_index("node_id")
        .to_pandas()
        .stack()
        .rename("poi_idx")
        .reset_index()
        .rename(columns={"node_id": "pc_node"})
        .drop("level_1", axis=1)
        .groupby("poi_idx")
        .agg(list)
        .join(df, how="right")
    )

    # retain only unique postcode ids
    poi_nn["pc_node"] = poi_nn["pc_node"].apply(
        lambda row: list(set(row)) if isinstance(row, list) else row
    )

    distances = distances.stack().rename("dist").reset_index().drop("level_1", axis=1)
    indices = indices.stack().rename("ind").reset_index().drop("level_1", axis=1)

    poi_nodes = (
        poi_nn[["node_id"]]
        .iloc[indices["ind"].values.get()]["node_id"]
        .reset_index(drop=True)
    )
    buffers = cudf.DataFrame({"node_id": poi_nodes, "buffer": distances["dist"].values})
    buffers = buffers.sort_values("buffer", ascending=False).drop_duplicates("node_id")
    buffers["buffer"] = buffers["buffer"].astype("int")

    # this will drop rows that did not appear in the KNN i.e unneeded poi
    return cudf.from_pandas(
        poi_nn.merge(buffers.to_pandas(), on="node_id", how="left")
        .dropna()
        .drop_duplicates("node_id")
    )


def process_routing(df: pd.DataFrame, name: str):
    nodes: cudf.DataFrame = cudf.from_pandas(
        pd.read_parquet(Paths.OS_GRAPH / "nodes.parquet")
    )
    pcs = cudf.from_pandas(pd.read_parquet(Paths.PROCESSED / "postcodes.parquet"))
    df = nearest_nodes(df, nodes=nodes)
    df = get_buffers(df=df, postcodes=pcs, k=10)
    df.to_parquet(Paths.PROCESSED / f"{name}.parquet")
