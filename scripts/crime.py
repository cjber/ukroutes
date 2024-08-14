import cudf
import cugraph
import geopandas as gpd
import numpy as np
import pandas as pd

from ukroutes import Routing
from ukroutes.common.utils import Paths
from ukroutes.preprocessing import process_os
from ukroutes.process_routing import add_to_graph, add_topk

# process_os()

postcodes = pd.read_parquet("./data/processed/postcodes.parquet")
postcodes["postcode"] = postcodes["postcode"].str.replace(" ", "")
postcodes = postcodes[
    postcodes["postcode"].str.contains("^CH\d|^L\d|^WA\d|^WN\d|^PR\d")
]


def process_crime(postcodes):
    crime = pd.read_excel(
        "./data/raw/Serious Violence Asset Directory.xlsx",
        skiprows=3,
        usecols=["Asset Name", "Postcode"],
    )
    crime["Postcode"] = crime["Postcode"].str.replace(" ", "")
    crime = crime.merge(
        postcodes, left_on="Postcode", right_on="postcode", how="left"
    ).drop(columns=["Postcode", "postcode", "node_id"])
    return crime.dropna()


nodes: cudf.DataFrame = cudf.from_pandas(
    pd.read_parquet(Paths.OS_GRAPH / "nodes.parquet")
)
edges: cudf.DataFrame = cudf.from_pandas(
    pd.read_parquet(Paths.OS_GRAPH / "edges.parquet")
)

crime = process_crime(postcodes)
crime
crime.to_csv("./data/raw/violence_assets.csv", index=False)
crime, nodes, edges = add_to_graph(crime, nodes, edges, 1)
crime
postcodes, nodes, edges = add_to_graph(postcodes, nodes, edges, 1)
crime, postcodes = add_topk(crime, postcodes, 10)

routing = Routing(
    name="violence_asset",
    edges=edges,
    nodes=nodes,
    outputs=postcodes,
    inputs=crime,
    weights="time_weighted",
    min_buffer=5000,
    max_buffer=500_000,
)
routing.fit()
routing.distances

distances = (
    routing.distances.set_index("vertex")
    .join(cudf.from_pandas(postcodes).set_index("node_id"), how="right")
    .reset_index()
)
distances[distances["distance"].isna()]

OUT_FILE = Paths.OUT / "distances_violence_assets.csv"
distances.to_pandas().to_csv(OUT_FILE, index=False)

import matplotlib.pyplot as plt

distances = pd.read_csv(OUT_FILE)
fig, ax = plt.subplots()
distances.reset_index().sort_values("distance").plot(
    x="easting", y="northing", kind="scatter", c="distance", cmap="viridis", ax=ax
)
distances[distances["distance"].isna()].plot(
    x="easting", y="northing", kind="scatter", c="red", ax=ax
)

plt.show()
