from pathlib import Path

import networkx as nx


class Paths:
    DATA = Path("data")
    RAW = DATA / "raw"
    OUT = DATA / "out"
    OPROAD = RAW / "oproad" / "oproad_gb.gpkg"

    PROCESSED = DATA / "processed"
    GRAPH = PROCESSED / "oproad"


def filter_deadends(nodes, edges):
    G = nx.from_pandas_edgelist(
        edges, source="start_node", target="end_node", edge_attr="time_weighted"
    )
    largest_cc = max(nx.connected_components(G), key=len)
    Gsub = G.subgraph(largest_cc)

    edges = nx.to_pandas_edgelist(
        Gsub,
        source="start_node",
        target="end_node",
        edge_key="time_weighted",
    )
    nodes = nodes[
        nodes["node_id"].isin(edges["start_node"])
        | nodes["node_id"].isin(edges["end_node"])
    ]
    return nodes, edges
