import cudf
import cuspatial
import geopandas as gpd
import numpy as np
import pandas as pd

from ukroutes import Routing
from ukroutes.common.utils import Paths
from ukroutes.preprocessing import process_os
from ukroutes.process_routing import process_pcs, process_routing


def process_ev():
    ev_raw = pd.read_csv(
        Paths.RAW / "process" / "national-charge-point-registry.csv",
        usecols=["name", "latitude", "longitude"],
    )
    ev_raw = gpd.GeoDataFrame(
        ev_raw,
        geometry=gpd.points_from_xy(ev_raw["longitude"], ev_raw["latitude"]),
        crs=4326,
    ).to_crs(27700)
    ev_raw["easting"], ev_raw["northing"] = (ev_raw.geometry.x, ev_raw.geometry.y)
    ev_raw = (
        ev_raw.drop(columns=["geometry", "latitude", "longitude"])
        .replace([np.inf, -np.inf], np.nan)
        .dropna(how="any")
    )
    return ev_raw


ev_raw = process_ev()
process_os()
process_pcs()
name = 'ev'
process_routing(ev_raw, name)
ev = pd.read_parquet(Paths.PROCESSED / "ev.parquet")


edges: cudf.DataFrame = cudf.read_parquet(Paths.OS_GRAPH / "edges.parquet")  # type:ignore
nodes: pd.DataFrame = pd.read_parquet(Paths.OS_GRAPH / "nodes.parquet")  # type:ignore
nodes: cuspatial.GeoDataFrame = cuspatial.from_geopandas(  # type:ignore
    gpd.GeoDataFrame(  # type:ignore
        nodes,
        geometry=gpd.points_from_xy(nodes["easting"], nodes["northing"]),
    )
)
postcodes: cudf.DataFrame = cudf.read_parquet(  # type:ignore
    Paths.PROCESSED / "postcodes.parquet"
)

routing = Routing(
    name="ev",
    edges=edges,
    nodes=nodes,
    sources=postcodes,
    targets=ev,
    weights="time_weighted",
)
routing.fit()

distances = (
    routing.distances.set_index("vertex")
    .join(postcodes.set_index("node_id"), how="right")
    .reset_index()
)
OUT_FILE = Paths.OUT_DATA / f"distances_{name}.csv"
distances.to_pandas()[["postcode", "distance"]].to_csv(OUT_FILE, index=False)
