import time
from pathlib import Path

import networkx as nx
import pandas as pd
from tqdm import tqdm

from ukroutes.common.utils import Paths
from ukroutes.process_routing import add_to_graph


class Route:
    def __init__(self, infile):
        self.infile = infile
        self.get_data()

    def get_data(self):
        self.nodes = pd.read_parquet(Paths.GRAPH / "nodes.parquet")
        self.edges = pd.read_parquet(Paths.GRAPH / "edges.parquet")
        self.postcodes = pd.read_parquet(
            Paths.PROCESSED / "onspd" / "postcodes.parquet"
        )

        self.poi = pd.read_parquet(self.infile).dropna(subset=["easting", "northing"])
        self.poi, self.nodes, self.edges = add_to_graph(
            self.poi, self.nodes, self.edges, "time_weighted", 1
        )
        self.postcodes, self.nodes, self.edges = add_to_graph(
            self.postcodes, self.nodes, self.edges, "time_weighted", 1
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
            G, sources=self.poi["node_id"].to_list(), weight="time_weighted"
        )
        print("CPU routing complete!")
        t2 = time.time()
        print(f"CPU routing took {t2 - t1:.2f} seconds")

        distances = pd.DataFrame(
            {"node_id": distances.keys(), "time_weighted": distances.values()}
        )
        distances = distances[distances["node_id"].isin(self.postcodes["node_id"])]
        distances = self.postcodes.merge(distances, on="node_id")
        return distances
