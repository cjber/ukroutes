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

    nodes = nodes[nodes["node_id"].isin(largest_cc)]
    edges = edges[
        edges["start_node"].isin(largest_cc) | edges["end_node"].isin(largest_cc)
    ]
    return nodes, edges
