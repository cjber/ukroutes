import cudf
import cuspatial
import geopandas as gpd
import numpy as np
import pandas as pd

from ukroutes import Routing
from ukroutes.common.utils import Paths
from ukroutes.preprocessing import process_os
from ukroutes.process_routing import add_to_graph


def process_ev():
    ev = pd.read_csv(
        Paths.RAW / "process" / "national-charge-point-registry.csv",
        usecols=["name", "latitude", "longitude"],
    )
    ev = gpd.GeoDataFrame(
        ev,
        geometry=gpd.points_from_xy(ev["longitude"], ev["latitude"]),
        crs=4326,
    ).to_crs(27700)
    ev["easting"], ev["northing"] = (ev.geometry.x, ev.geometry.y)
    ev = (
        ev.drop(columns=["geometry", "latitude", "longitude"])
        .replace([np.inf, -np.inf], np.nan)
        .dropna(how="any")
    )
    return ev


ev = process_ev()[:1000]
process_os()

nodes: cudf.DataFrame = cudf.read_parquet(Paths.OS_GRAPH / "nodes.parquet")  # type:ignore
edges: cudf.DataFrame = cudf.read_parquet(Paths.OS_GRAPH / "edges.parquet")  # type:ignore
ev, nodes, edges = add_to_graph(ev, nodes, edges, "name")

postcodes: cudf.DataFrame = pd.read_parquet(  # type:ignore
    Paths.PROCESSED / "postcodes.parquet"
)
postcodes, nodes, edges = add_to_graph(postcodes, nodes, edges, "postcode")

routing = Routing(
    name="ev",
    edges=edges,
    nodes=nodes,
    sources=postcodes,
    targets=ev,
    weights="time_weighted",
    buffer=100_000,
)
routing.fit()

distances = (
    routing.distances.set_index("vertex")
    .join(postcodes.set_index("node_id"), how="right")
    .reset_index()
)
OUT_FILE = Paths.OUT_DATA / "distances_ev.csv"
distances.to_pandas()[["postcode", "distance"]].to_csv(OUT_FILE, index=False)
