import cudf
from scipy.spatial import distance_matrix
import pandas as pd
import cugraph
import geopandas as gpd
import polars as pl
from scipy.spatial import KDTree

from ukroutes.common.logger import logger
from ukroutes.common.utils import Paths, filter_deadends
from ukroutes.process_routing import add_to_graph


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
            Paths.OPROAD,
            layer="road_link",
            ignore_geometry=True,
            engine="pyogrio",  # much faster
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
        )
    )
    return road_edges.select(["start_node", "end_node", "time_weighted", "length"])


def process_road_nodes() -> pl.DataFrame:
    road_nodes = gpd.read_file(Paths.OPROAD, layer="road_node", engine="pyogrio")
    road_nodes["easting"], road_nodes["northing"] = (
        road_nodes.geometry.x,
        road_nodes.geometry.y,
    )
    return pl.from_pandas(road_nodes[["id", "easting", "northing"]]).rename(
        {"id": "node_id"}
    )


def ferry_routes(road_nodes: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    # http://overpass-turbo.eu/?q=LyoKVGhpcyBoYcSGYmVlbiBnxI1lcmF0ZWQgYnkgdGhlIG92xJJwxIlzLXR1cmJvIHdpemFyZC7EgsSdxJ9yaWdpbmFsIHNlxLBjaMSsxIk6CsOiwoDCnHJvdcSVPWbEknJ5xYjCnQoqLwpbxYx0Ompzb25dW3RpbWXFmzoyNV07Ci8vxI_ElMSdciByZXN1bHRzCigKICDFryBxdcSSxJrEo3J0IGZvcjogxYjFisWbZcWPxZHFk8KAxZXGgG5vZGVbIsWLxY1lIj0ixZByxZIiXSh7e2LEqnh9fSnFrcaAd2F5xp_GocSVxqTGpsaWxqrGrMauxrDGssa0xb_FtWVsxJRpxaDGusaTxr3Gp8apxqvGrcavb8axxrPFrceFxoJwxLduxorFtsW4xbrFvMWbxJjGnHnFrT7Frcejc2vHiMaDdDs&c=BH1aTWQmgG

    ferries = gpd.read_file(Paths.RAW / "oproad" / "ferries.geojson")[
        ["id", "geometry"]
    ].to_crs("EPSG:27700")
    ferry_nodes = (
        ferries[ferries["id"].str.startswith("node")].copy().reset_index(drop=True)
    )
    ferry_nodes["easting"], ferry_nodes["northing"] = (
        ferry_nodes.geometry.x,
        ferry_nodes.geometry.y,
    )
    ferry_edges = (
        ferries[ferries["id"].str.startswith("relation")]
        .explode(index_parts=False)
        .copy()
        .reset_index(drop=True)
    )
    road_nodes = road_nodes.to_pandas().copy()

    nodes_tree = KDTree(road_nodes[["easting", "northing"]].values)
    distances, indices = nodes_tree.query(ferry_nodes[["easting", "northing"]].values)
    ferry_nodes["node_id"] = road_nodes.iloc[indices]["node_id"].reset_index(drop=True)

    ferry_edges["length"] = ferry_edges["geometry"].apply(lambda x: x.length)
    ferry_edges = ferry_edges.assign(
        time_weighted=(ferry_edges["length"].astype(float) / 1000) / 25 * 1.609344 * 60
    )

    ferry_edges["start_node"] = ferry_edges["geometry"].apply(lambda x: x.coords[0])
    ferry_edges["easting"], ferry_edges["northing"] = (
        ferry_edges["start_node"].apply(lambda x: x[0]),
        ferry_edges["start_node"].apply(lambda x: x[1]),
    )
    distances, indices = nodes_tree.query(ferry_edges[["easting", "northing"]])
    ferry_edges["start_node"] = road_nodes.iloc[indices]["node_id"].reset_index(
        drop=True
    )

    ferry_edges["end_node"] = ferry_edges["geometry"].apply(lambda x: x.coords[-1])
    ferry_edges["easting"], ferry_edges["northing"] = (
        ferry_edges["end_node"].apply(lambda x: x[0]),
        ferry_edges["end_node"].apply(lambda x: x[1]),
    )
    distances, indices = nodes_tree.query(ferry_edges[["easting", "northing"]])
    ferry_edges["end_node"] = road_nodes.iloc[indices]["node_id"].reset_index(drop=True)
    return (
        pl.from_pandas(ferry_nodes[["node_id", "easting", "northing"]]),
        pl.from_pandas(
            ferry_edges[["start_node", "end_node", "time_weighted", "length"]]
        ),
    )


def combine_subgraphs(nodes, edges):
    graph = cugraph.Graph()
    graph.from_cudf_edgelist(
        cudf.from_pandas(edges), source="start_node", destination="end_node"
    )
    components = cugraph.connected_components(graph)
    component_counts = components["labels"].value_counts().reset_index()

    largest_component_label = component_counts[
        component_counts["count"] == component_counts["count"].max()
    ]["labels"][0]
    largest_component = components[components["labels"] == largest_component_label]
    largest_cn = nodes[nodes["node_id"].isin(largest_component["vertex"].to_pandas())]
    largest_ce = edges[
        edges["start_node"].isin(largest_component["vertex"].to_pandas())
        | edges["end_node"].isin(largest_component["vertex"].to_pandas())
    ]

    subgraph_component_labels = component_counts[
        component_counts["labels"] != largest_component_label
    ]["labels"]
    subgraph_component = components[
        components["labels"].isin(subgraph_component_labels)
    ]
    sub_cn = nodes[nodes["node_id"].isin(subgraph_component["vertex"].to_pandas())]

    _, nodes, edges = add_to_graph(
        sub_cn,
        cudf.from_pandas(largest_cn),
        cudf.from_pandas(largest_ce),
    )
    return nodes, edges


def process_os():
    logger.info("Starting OS highways processing...")
    edges = process_road_edges()
    nodes = process_road_nodes()

    ferry_nodes, ferry_edges = ferry_routes(nodes)
    nodes = pl.concat([nodes, ferry_nodes]).to_pandas()
    edges = pl.concat([edges, ferry_edges]).to_pandas()

    unique_node_ids = nodes["node_id"].unique()
    node_id_mapping = {
        node_id: new_id for new_id, node_id in enumerate(unique_node_ids)
    }
    nodes["node_id"] = nodes["node_id"].map(node_id_mapping)
    edges["start_node"] = edges["start_node"].map(node_id_mapping)
    edges["end_node"] = edges["end_node"].map(node_id_mapping)

    # nodes, edges = filter_deadends(cudf.from_pandas(nodes), cudf.from_pandas(edges))
    nodes, edges = combine_subgraphs(nodes, edges)

    nodes.to_pandas().to_parquet(Paths.OS_GRAPH / "nodes.parquet", index=False)
    logger.debug(f"Nodes saved to {Paths.OS_GRAPH / 'nodes.parquet'}")
    edges.to_pandas().to_parquet(Paths.OS_GRAPH / "edges.parquet", index=False)
    logger.debug(f"Edges saved to {Paths.OS_GRAPH / 'edges.parquet'}")


if __name__ == "__main__":
    process_os()
