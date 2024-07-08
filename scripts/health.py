import cudf
import cugraph
import geopandas as gpd
import numpy as np
import pandas as pd

from ukroutes import Routing
from ukroutes.common.utils import Paths
from ukroutes.preprocessing import process_os
from ukroutes.process_routing import add_to_graph, add_topk

health = pd.read_parquet("./data/processed/health.parquet").dropna().sample(1000)

nodes: cudf.DataFrame = cudf.from_pandas(
    pd.read_parquet(Paths.OS_GRAPH / "nodes.parquet")
)
edges: cudf.DataFrame = cudf.from_pandas(
    pd.read_parquet(Paths.OS_GRAPH / "edges.parquet")
)

health, nodes, edges = add_to_graph(health, nodes, edges, 1)

postcodes = pd.read_csv(
    "./data/raw/onspd/ONSPD_FEB_2024.csv",
    usecols=["PCD", "OSEAST1M", "OSNRTH1M", "DOTERM", "CTRY"],
)
postcodes = (
    postcodes[
        (postcodes["DOTERM"].isnull())
        & (~postcodes["CTRY"].isin(["N92000002", "L93000001", "M83000003"]))
    ]
    .drop(columns=["DOTERM", "CTRY"])
    .rename({"PCD": "postcode", "OSEAST1M": "easting", "OSNRTH1M": "northing"}, axis=1)
    .dropna()
)

postcodes, nodes, edges = add_to_graph(postcodes, nodes, edges, 1)
health = add_topk(health, postcodes, 10)

routing = Routing(
    name="health",
    edges=edges,
    nodes=nodes,
    outputs=postcodes,
    inputs=health,
    weights="time_weighted",
    min_buffer=5000,
    max_buffer=500_000,
    # cutoff=300,
)
routing.fit()

distances = (
    routing.distances.set_index("vertex")
    .join(cudf.from_pandas(postcodes).set_index("node_id"), how="right")
    .reset_index()
    .to_pandas()
)

# OUT_FILE = Paths.OUT_DATA / "distances_health.csv"
# distances.to_csv(OUT_FILE, index=False)

distances = gpd.GeoDataFrame(
    distances, geometry=gpd.points_from_xy(distances.easting, distances.northing)
)
distances.reset_index().sort_values("distance").plot(column="distance")
distances.sort_values("distance").tail(25)

import matplotlib.pyplot as plt

plt.show()
