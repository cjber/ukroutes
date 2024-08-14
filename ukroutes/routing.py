import time

import networkx as nx
import pandas as pd

from ukroutes.process_routing import add_to_graph


class Route:
    def __init__(
        self,
        source: pd.DataFrame,
        target: pd.DataFrame,
        nodes: pd.DataFrame,
        edges: pd.DataFrame,
    ):
        self.source = source
        self.nodes = nodes
        self.edges = edges
        self.target = target

        self.build()

    def build(self):
        self.source, self.nodes, self.edges = add_to_graph(
            self.source, self.nodes, self.edges, "time_weighted", 1
        )
        self.target, self.nodes, self.edges = add_to_graph(
            self.target, self.nodes, self.edges, "time_weighted", 1
        )

    def route(self):
        edge_list = list(
            self.edges[["start_node", "end_node", "time_weighted"]].itertuples(
                index=False, name=None
            )
        )
        G = nx.Graph()
        G.add_weighted_edges_from(edge_list, weight="time_weighted")

        t1 = time.time()
        print("Starting CPU routing...")
        distances = nx.multi_source_dijkstra_path_length(
            G, sources=self.source["node_id"].to_list(), weight="time_weighted"
        )
        print("CPU routing complete!")
        t2 = time.time()
        print(f"CPU routing took {t2 - t1:.2f} seconds")

        distances = pd.DataFrame(
            {"node_id": distances.keys(), "time_weighted": distances.values()}
        )
        distances = distances[distances["node_id"].isin(self.target["node_id"])]
        distances = self.target.merge(distances, on="node_id")
        return distances
