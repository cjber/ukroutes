from typing import Optional

import geopandas as gpd
import pandas as pd
import polars as pl
from scipy.spatial import KDTree
from shapely.geometry import Point

from ukroutes.common.utils import Paths, filter_deadends


def _process_road_edges() -> pl.DataFrame:
    """
    Process road edges to estimate speed and time weights.

    Returns:
        pl.DataFrame: A DataFrame containing processed road edges with estimated speed and time weights.
    """
    a_roads = ["A Road", "A Road Primary"]
    b_roads = ["B Road", "B Road Primary"]
    dual_carriageway = ["Dual Carriageway", "Collapsed Dual Carriageway"]

    road_edges: pl.DataFrame = pl.from_pandas(
        gpd.read_file(
            Paths.OPROAD, layer="road_link", ignore_geometry=True, engine="pyogrio"
        )
    )

    road_edges = (
        road_edges.with_columns(
            pl.when(pl.col("road_classification") == "Motorway")
            .then(67)
            .when(
                (pl.col("form_of_way").is_in(dual_carriageway))
                & (pl.col("road_classification").is_in(a_roads))
            )
            .then(57)
            .when(
                (pl.col("form_of_way").is_in(dual_carriageway))
                & (pl.col("road_classification").is_in(b_roads))
            )
            .then(45)
            .when(
                (pl.col("form_of_way") == "Single Carriageway")
                & (pl.col("road_classification").is_in(a_roads + b_roads))
            )
            .then(25)
            .when(pl.col("road_classification").is_in(["Unclassified"]))
            .then(24)
            .when(pl.col("form_of_way") == "Roundabout")
            .then(10)
            .when(pl.col("form_of_way").is_in(["Track", "Layby"]))
            .then(5)
            .otherwise(10)
            .alias("speed_estimate")
        )
        .with_columns(pl.col("speed_estimate") * 1.609344)
        .with_columns(
            (((pl.col("length") / 1000) / pl.col("speed_estimate")) * 60).alias(
                "time_weighted"
            ),
            (((pl.col("length") / 1000) / 5) * 60).alias("pedestrian_time"),
            ((pl.col("length") / 1000)).alias("distance"),
        )
    )
    return road_edges.select(
        [
            "start_node",
            "end_node",
            "time_weighted",
            "length",
            "pedestrian_time",
            "distance",
        ]
    )


def _process_road_nodes() -> pl.DataFrame:
    """
    Process road nodes to extract easting and northing coordinates.

    Returns:
        pl.DataFrame: A DataFrame containing processed road nodes with easting and northing coordinates.
    """
    road_nodes = gpd.read_file(Paths.OPROAD, layer="road_node", engine="pyogrio")
    road_nodes["easting"], road_nodes["northing"] = (
        road_nodes.geometry.x,
        road_nodes.geometry.y,
    )
    return pl.from_pandas(road_nodes[["id", "easting", "northing"]]).rename(
        {"id": "node_id"}
    )


def _ferry_routes(nodes: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Process ferry routes to create ferry nodes and edges.

    Args:
        nodes (pl.DataFrame): DataFrame containing road nodes.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: A tuple containing DataFrames for ferry nodes and ferry edges.
    """
    ferries = gpd.read_file(Paths.RAW / "oproad" / "ferries.geojson")[
        ["geometry"]
    ].to_crs("EPSG:27700")
    ferries = ferries[~ferries["geometry"].apply(lambda x: isinstance(x, Point))]

    ferry_edges = ferries[["geometry"]].explode().copy()

    ferry_edges["start_node"] = ferry_edges.geometry.apply(lambda x: x.coords[0])
    ferry_edges["end_node"] = ferry_edges.geometry.apply(lambda x: x.coords[-1])
    ferry_edges["length"] = ferry_edges.geometry.length
    ferry_edges = ferry_edges.assign(
        time_weighted=(
            (ferry_edges["length"].astype(float) / 1000) / 25 * 1.609344 * 60
        ),
        distance=(ferry_edges["length"].astype(float) / 1000),
    )
    ferry_nodes = pd.DataFrame(
        {
            "node_id": ferry_edges["start_node"].to_list()
            + ferry_edges["end_node"].to_list()
        }
    )
    ferry_nodes["easting"] = ferry_nodes["node_id"].apply(lambda x: x[0])
    ferry_nodes["northing"] = ferry_nodes["node_id"].apply(lambda x: x[1])
    ferry_nodes = ferry_nodes.drop_duplicates().reset_index(drop=True)

    ferry_nodes["node_id"] = ferry_nodes["node_id"].astype(str)
    ferry_edges["start_node"] = ferry_edges["start_node"].astype(str)
    ferry_edges["end_node"] = ferry_edges["end_node"].astype(str)

    nodes_tree = KDTree(nodes[["easting", "northing"]])
    _, indices = nodes_tree.query(ferry_nodes[["easting", "northing"]], k=1)
    mapping = dict(
        zip(
            ferry_nodes["node_id"], nodes.to_pandas().iloc[indices.flatten()]["node_id"]
        )
    )
    ferry_nodes["node_id"] = ferry_nodes["node_id"].map(mapping)
    ferry_edges["end_node"] = ferry_edges["end_node"].map(mapping)
    ferry_edges["start_node"] = ferry_edges["start_node"].map(mapping)
    ferry_edges["pedestrian_time"] = ferry_edges["time_weighted"]

    return pl.from_pandas(
        ferry_nodes[["node_id", "easting", "northing"]]
    ), pl.from_pandas(
        ferry_edges[
            [
                "start_node",
                "end_node",
                "time_weighted",
                "length",
                "pedestrian_time",
                "distance",
            ]
        ]
    )


def process_oproad(save: Optional[bool] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process the Ordnance Survey road data and optionally save the results.

    Args:
        save (Optional[bool]): If True, save the processed nodes and edges to parquet files.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing DataFrames for nodes and edges.
    """
    edges_pl = _process_road_edges()
    nodes_pl = _process_road_nodes()

    ferry_nodes, ferry_edges = _ferry_routes(nodes_pl)
    nodes = pl.concat([nodes_pl, ferry_nodes]).to_pandas().drop_duplicates("node_id")
    edges = (
        pl.concat([edges_pl, ferry_edges])
        .to_pandas()
        .drop_duplicates(["start_node", "end_node"])
    )

    nodes, edges = filter_deadends(nodes, edges)

    unique_node_ids = nodes["node_id"].unique()
    node_id_mapping = {
        node_id: new_id for new_id, node_id in enumerate(unique_node_ids)
    }
    nodes["node_id"] = nodes["node_id"].map(node_id_mapping)
    edges["start_node"] = edges["start_node"].map(node_id_mapping)
    edges["end_node"] = edges["end_node"].map(node_id_mapping)

    nodes = nodes.reset_index(drop=True)
    edges = edges.reset_index(drop=True)

    if save:
        nodes.to_parquet(Paths.GRAPH / "nodes.parquet", index=False)
        edges.to_parquet(Paths.GRAPH / "edges.parquet", index=False)
    return nodes, edges
