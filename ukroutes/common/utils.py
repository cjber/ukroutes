from pathlib import Path

import cudf
import cugraph


class Paths:
    DATA = Path("data")
    RAW = DATA / "raw"
    OUT_DATA = DATA / "out"
    OPROAD = RAW / "oproad" / "oproad_gb.gpkg"

    PROCESSED = DATA / "processed"
    OS_GRAPH = PROCESSED / "oproad"


def filter_deadends(
    nodes: cudf.DataFrame, edges: cudf.DataFrame
) -> tuple[cudf.DataFrame, cudf.DataFrame]:
    G = cugraph.Graph()
    G.from_cudf_edgelist(
        edges, source="start_node", destination="end_node", edge_attr="time_weighted"
    )
    components: cudf.DataFrame = cugraph.connected_components(G)
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
