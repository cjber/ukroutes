import cudf
import geopandas as gpd
import pandas as pd

from ukroutes import Routing
from ukroutes.common.utils import Paths
from ukroutes.process_routing import add_to_graph, add_topk
from ukroutes.oproad.utils import process_oproad

# process oproad nodes and edges
nodes, edges = process_oproad(outdir=Paths.OS_GRAPH)  # or outdir=None

# read in health dataa and postcodes
health = pd.read_parquet("./data/processed/health.parquet").dropna().sample(1000)
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

# add health and postcodes to road network
health, nodes, edges = add_to_graph(health, nodes, edges, 10)
postcodes, nodes, edges = add_to_graph(postcodes, nodes, edges, 10)

# find the top 10 closest health facilities to each postcode
health = add_topk(health, postcodes, 25)

# run the routing class
routing = Routing(
    edges=edges,
    nodes=nodes,
    inputs=health,
    outputs=postcodes,
    weights="time_weighted",
    min_buffer=5000,
    max_buffer=500_000,
    cutoff=300,
)
routing.fit()

# join distances to postcodes
distances = (
    routing.distances.set_index("vertex")
    .join(cudf.from_pandas(postcodes).set_index("node_id"), how="right")
    .reset_index()
    .to_pandas()
)

import matplotlib.pyplot as plt

distances = gpd.GeoDataFrame(
    distances, geometry=gpd.points_from_xy(distances.easting, distances.northing)
)
distances.reset_index().sort_values("distance").plot(column="distance")


plt.savefig("./figs/health_example.png")
