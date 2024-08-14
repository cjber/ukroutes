from pathlib import Path
from typing import Optional

import cudf
import geopandas as gpd
import pandas as pd
import polars as pl
from scipy.spatial import KDTree
from shapely import Point

from ukroutes.common.utils import Paths, filter_deadends


def process_road_edges() -> pl.DataFrame:
    """
    Create time estimates for road edges based on OS documentation

    Time estimates based on speed estimates and edge length. Speed estimates
    taken from OS documentation. This also filters to remove extra cols.

    Parameters
    ----------
    edges : pd.DataFrame
        OS highways df containing edges, and other metadata

    Returns
    -------
    pd.DataFrame:
        OS highways df with time weighted estimates
    """

    a_roads = ["A Road", "A Road Primary"]
    b_roads = ["B Road", "B Road Primary"]

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
                (
                    pl.col("form_of_way").is_in(
                        ["Dual Carriageway", "Collapsed Dual Carriageway"]
                    )
                )
                & (pl.col("road_classification").is_in(a_roads))
            )
            .then(57)
            .when(
                (
                    pl.col("form_of_way").is_in(
                        ["Dual Carriageway", "Collapsed Dual Carriageway"]
                    )
                )
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
            .when(pl.col("form_of_way").is_in(["Roundabout"]))
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
        )
    )
    return road_edges.select(
        ["start_node", "end_node", "time_weighted", "length", "pedestrian_time"]
    )


def process_road_nodes() -> pl.DataFrame:
    road_nodes = gpd.read_file(Paths.OPROAD, layer="road_node", engine="pyogrio")
    road_nodes["easting"], road_nodes["northing"] = (
        road_nodes.geometry.x,
        road_nodes.geometry.y,
    )
    return pl.from_pandas(road_nodes[["id", "easting", "northing"]]).rename(
        {"id": "node_id"}
    )


def ferry_routes(nodes: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    # http://overpass-turbo.eu/?q=LyoKVGhpcyBoYcSGYmVlbiBnxI1lcmF0ZWQgYnkgdGhlIG92xJJwxIlzLXR1cmJvIHdpemFyZC7EgsSdxJ9yaWdpbmFsIHNlxLBjaMSsxIk6CsOiwoDCnHJvdcSVPWbEknJ5xYjCnQoqLwpbxYx0Ompzb25dW3RpbWXFmzoyNV07Ci8vxI_ElMSdciByZXN1bHRzCigKICDFryBxdcSSxJrEo3J0IGZvcjogxYjFisWbZcWPxZHFk8KAxZXGgG5vZGVbIsWLxY1lIj0ixZByxZIiXSh7e2LEqnh9fSnFrcaAd2F5xp_GocSVxqTGpsaWxqrGrMauxrDGssa0xb_FtWVsxJRpxaDGusaTxr3Gp8apxqvGrcavb8axxrPFrceFxoJwxLduxorFtsW4xbrFvMWbxJjGnHnFrT7Frcejc2vHiMaDdDs&c=BH1aTWQmgG

    ferries = gpd.read_file(Paths.RAW / "oproad" / "ferries.geojson")[
        ["geometry"]
    ].to_crs("EPSG:27700")
    ferries = ferries[~ferries["geometry"].apply(lambda x: isinstance(x, Point))]

    ferry_edges = ferries[["geometry"]].explode().copy()

    ferry_edges["start_node"] = ferry_edges.geometry.apply(lambda x: x.coords[0])
    ferry_edges["end_node"] = ferry_edges.geometry.apply(lambda x: x.coords[-1])
    ferry_edges["length"] = ferry_edges.geometry.length
    ferry_edges = ferry_edges.assign(
        time_weighted=(ferry_edges["length"].astype(float) / 1000) / 25 * 1.609344 * 60
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
    distances, indices = nodes_tree.query(ferry_nodes[["easting", "northing"]], k=1)
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
            ["start_node", "end_node", "time_weighted", "length", "pedestrian_time"]
        ]
    )


def process_oproad(
    outdir: Optional[Path] = None,
) -> tuple[cudf.DataFrame, cudf.DataFrame]:
    edges_pl = process_road_edges()
    nodes_pl = process_road_nodes()

    ferry_nodes, ferry_edges = ferry_routes(nodes_pl)
    nodes_pl = pl.concat([nodes_pl, ferry_nodes]).to_pandas().drop_duplicates("node_id")
    edges_pl = (
        pl.concat([edges_pl, ferry_edges])
        .to_pandas()
        .drop_duplicates(["start_node", "end_node"])
    )

    nodes_cu: cudf.DataFrame = cudf.from_pandas(nodes_pl)
    edges_cu: cudf.DataFrame = cudf.from_pandas(edges_pl)
    nodes_cu, edges_cu = filter_deadends(nodes_cu, edges_cu)
    nodes_pd: pd.DataFrame = nodes_cu.to_pandas()
    edges_pd: pd.DataFrame = edges_cu.to_pandas()

    unique_node_ids = nodes_pd["node_id"].unique()
    node_id_mapping = {
        node_id: new_id for new_id, node_id in enumerate(unique_node_ids)
    }
    nodes_pd["node_id"] = nodes_pd["node_id"].map(node_id_mapping)
    edges_pd["start_node"] = edges_pd["start_node"].map(node_id_mapping)
    edges_pd["end_node"] = edges_pd["end_node"].map(node_id_mapping)

    nodes_cu: cudf.DataFrame = cudf.from_pandas(nodes_pd).reset_index(drop=True)
    edges_cu: cudf.DataFrame = cudf.from_pandas(edges_pd).reset_index(drop=True)

    if outdir:
        nodes_cu.to_pandas().to_parquet(Paths.GRAPH / "nodes.parquet", index=False)
        edges_cu.to_pandas().to_parquet(Paths.GRAPH / "edges.parquet", index=False)
    return nodes_cu, edges_cu
