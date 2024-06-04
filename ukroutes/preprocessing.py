import geopandas as gpd
import polars as pl
from cuml.neighbors.nearest_neighbors import NearestNeighbors

from ukroutes.common.logger import logger
from ukroutes.common.utils import Paths


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
    nbrs = NearestNeighbors(n_neighbors=1).fit(road_nodes[["easting", "northing"]])
    indices = nbrs.kneighbors(
        ferry_nodes[["easting", "northing"]], return_distance=False
    )
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
    indices = nbrs.kneighbors(
        ferry_edges[["easting", "northing"]], return_distance=False
    )
    ferry_edges["start_node"] = road_nodes.iloc[indices]["node_id"].reset_index(
        drop=True
    )

    ferry_edges["end_node"] = ferry_edges["geometry"].apply(lambda x: x.coords[-1])
    ferry_edges["easting"], ferry_edges["northing"] = (
        ferry_edges["end_node"].apply(lambda x: x[0]),
        ferry_edges["end_node"].apply(lambda x: x[1]),
    )
    indices = nbrs.kneighbors(
        ferry_edges[["easting", "northing"]], return_distance=False
    )
    ferry_edges["end_node"] = road_nodes.iloc[indices]["node_id"].reset_index(drop=True)
    return (
        pl.from_pandas(ferry_nodes[["node_id", "easting", "northing"]]),
        pl.from_pandas(
            ferry_edges[["start_node", "end_node", "time_weighted", "length"]]
        ),
    )


def process_os():
    logger.info("Starting OS highways processing...")
    edges = process_road_edges()
    nodes = process_road_nodes()
    ferry_nodes, ferry_edges = ferry_routes(nodes)
    nodes = pl.concat([nodes, ferry_nodes])
    edges = pl.concat([edges, ferry_edges])

    nodes.write_parquet(Paths.OS_GRAPH / "nodes.parquet")
    logger.debug(f"Nodes saved to {Paths.OS_GRAPH / 'nodes.parquet'}")
    edges.write_parquet(Paths.OS_GRAPH / "edges.parquet")
    logger.debug(f"Edges saved to {Paths.OS_GRAPH / 'edges.parquet'}")

if __name__ == "__main__":
    process_os()
