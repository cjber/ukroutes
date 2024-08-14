import logging
import time

import networkx as nx
import pandas as pd

from ukroutes.process_routing import add_to_graph

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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

        logging.info("Initializing Route class")
        self.build()

    def build(self):
        logging.info("Building graph with source and target nodes")
        self.source, self.nodes, self.edges = add_to_graph(
            self.source, self.nodes, self.edges, "time_weighted", 1
        )
        logging.info("Source nodes added to graph")
        self.target, self.nodes, self.edges = add_to_graph(
            self.target, self.nodes, self.edges, "time_weighted", 1
        )
        logging.info("Target nodes added to graph")

    def route(self):
        G = nx.from_pandas_edgelist(
            self.edges,
            source="start_node",
            target="end_node",
            edge_attr="time_weighted",
        )

        t1 = time.time()
        logging.info("Starting routing...")
        distances = nx.multi_source_dijkstra_path_length(
            G, sources=self.source["node_id"].to_list(), weight="time_weighted"
        )
        logging.info("Routing complete!")
        t2 = time.time()
        logging.info(f"Routing took {t2 - t1:.2f} seconds")

        distances = pd.DataFrame(
            {"node_id": distances.keys(), "time_weighted": distances.values()}
        )
        distances = distances[distances["node_id"].isin(self.target["node_id"])]
        distances = self.target.merge(distances, on="node_id")
        logging.info("Distances calculated and merged with target nodes")
        return distances
