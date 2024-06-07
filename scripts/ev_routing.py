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


ev = process_ev()
# process_os()

nodes: cudf.DataFrame = cudf.from_pandas(
    pd.read_parquet(Paths.OS_GRAPH / "nodes.parquet")
)
edges: cudf.DataFrame = cudf.from_pandas(
    pd.read_parquet(Paths.OS_GRAPH / "edges.parquet")
)


def filter_deadends(nodes, edges):
    G = cugraph.Graph()
    G.from_cudf_edgelist(
        edges, source="start_node", destination="end_node", edge_attr="time_weighted"
    )
    components = cugraph.connected_components(G)
    component_counts = components["labels"].value_counts().reset_index()
    component_counts.columns = ["labels", "count"]

    largest_component_label = component_counts[
        component_counts["count"] == component_counts["count"].max()
    ]["labels"][0]

    largest_component_nodes = components[
        components["labels"] == largest_component_label
    ]["vertex"]
    filtered_edges = edges[
        edges["start_node"].isin(largest_component_nodes)
        & edges["end_node"].isin(largest_component_nodes)
    ]
    filtered_nodes = nodes[nodes["node_id"].isin(largest_component_nodes)]
    return filtered_nodes, filtered_edges


nodes, edges = filter_deadends(nodes, edges)

ev, nodes, edges = add_to_graph(ev, nodes, edges, 1)
postcodes = pd.read_parquet(Paths.PROCESSED / "postcodes.parquet").sample(100)
postcodes, nodes, edges = add_to_graph(postcodes, nodes, edges, 1)
ev = add_topk(ev, postcodes, 1)

routing = Routing(
    name="ev",
    edges=edges,
    nodes=nodes,
    sources=postcodes,
    targets=ev,
    weights="time_weighted",
    min_buffer=5_000,
    max_buffer=1_000_000,
    cutoff=None,
)
routing.fit()
distances = routing.fetch_distances()

istances = (
    distances.set_index("vertex")
    .join(postcodes.set_index("node_id"), how="right")
    .reset_index()
)

OUT_FILE = Paths.OUT_DATA / "distances_ev.csv"
distances.to_pandas()[["postcode", "distance"]].to_csv(OUT_FILE, index=False)
