import cudf
import geopandas as gpd
import networkx as nx
import pandas as pd

from ukroutes import Routing
from ukroutes.common.utils import Paths
from ukroutes.oproad.utils import process_oproad
from ukroutes.process_routing import add_to_graph, add_topk

# process oproad nodes and edges
nodes, edges = process_oproad(outdir=Paths.OPROAD)


# read in health data and postcodes
health = (
    pd.read_parquet("./data/processed/health.parquet")
    .dropna()
    .sample(10)
    .reset_index(drop=True)
)

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
    .reset_index(drop=True)
)

# add health and postcodes to road network
health, nodes, edges = add_to_graph(health, nodes, edges, "pedestrian_time", 1)
postcodes, nodes, edges = add_to_graph(postcodes, nodes, edges, "pedestrian_time", 1)

# find the top 10 closest health facilities to each postcode
health = add_topk(health, postcodes, 10)


# run the routing class
routing = Routing(
    edges=edges,
    nodes=nodes,
    inputs=health,
    outputs=postcodes,
    weights="pedestrian_time",
    min_buffer=5000,
    max_buffer=500_000,
    # cutoff=300,
)
routing.fit()

# join distances to postcodes
distances = (
    routing.distances.set_index("vertex")
    .join(cudf.from_pandas(postcodes).set_index("node_id"), how="right")
    .reset_index()
    .to_pandas()
)
distances


distances = gpd.GeoDataFrame(
    distances, geometry=gpd.points_from_xy(distances.easting, distances.northing)
)
distances.loc[distances["distance"] > 60, "distance"] = 60
distances

fig, ax = plt.subplots()
distances.reset_index().sort_values("distance").plot(
    column="distance", cmap="viridis_r", legend=True, markersize=0.5, ax=ax
)
health.plot(x="easting", y="northing", color="red", ax=ax, kind="scatter", s=1)
ax.set_axis_off()

plt.savefig("./figs/health_example.png")
