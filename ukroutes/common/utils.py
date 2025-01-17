from pathlib import Path

import networkx as nx
import pandas as pd


class Paths:
    DATA = Path("data")
    RAW = DATA / "raw"
    OUT = DATA / "out"
    OPROAD = RAW / "oproad" / "oproad_gb.gpkg"

    PROCESSED = DATA / "processed"
    GRAPH = PROCESSED / "oproad"


def filter_deadends(
    nodes: pd.DataFrame, edges: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    G = nx.from_pandas_edgelist(edges, source="start_node", target="end_node")
    largest_cc = max(nx.connected_components(G), key=len)
    Gsub = G.subgraph(largest_cc)

    edges = nx.to_pandas_edgelist(Gsub, source="start_node", target="end_node")
    nodes = nodes[
        nodes["node_id"].isin(edges["start_node"])
        | nodes["node_id"].isin(edges["end_node"])
    ]  # type: ignore
    return nodes, edges
