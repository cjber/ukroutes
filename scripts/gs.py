import cudf
import cugraph
import geopandas as gpd
import numpy as np
import pandas as pd

from ukroutes import Routing
from ukroutes.common.utils import Paths
from ukroutes.preprocessing import process_os
from ukroutes.process_routing import add_to_graph, add_topk


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


gs = pd.read_parquet("./data/cillian/osgsl.parquet")
# process_os()

nodes: cudf.DataFrame = cudf.from_pandas(
    pd.read_parquet(Paths.OS_GRAPH / "nodes.parquet")
)
edges: cudf.DataFrame = cudf.from_pandas(
    pd.read_parquet(Paths.OS_GRAPH / "edges.parquet")
)


gs, nodes, edges = add_to_graph(gs, nodes, edges, 1)

uprn = pd.read_parquet("./data/cillian/postcodes.parquet")
uprn, nodes, edges = add_to_graph(uprn, nodes, edges, 1)
gs, uprn = add_topk(gs, uprn, 10)

routing = Routing(
    name="gs",
    edges=edges,
    nodes=nodes,
    outputs=uprn,
    inputs=gs,
    weights="time_weighted",
    min_buffer=5000,
    max_buffer=500_000,
)
routing.fit()

distances = (
    routing.distances.set_index("vertex")
    .join(cudf.from_pandas(uprn).set_index("node_id"), how="right")
    .reset_index()
)

OUT_FILE = Paths.OUT / "distances_greenspace_uprn.csv"
distances.to_pandas()[["postcode", "distance"]].to_csv(OUT_FILE, index=False)

distances = pd.read_csv("./data/out/distances_greenspace_uprn.csv")
distances
distances.reset_index().sort_values("distance").plot(
    x="index", y="distance", kind="scatter"
)

import matplotlib.pyplot as plt

plt.show()
