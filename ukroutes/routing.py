import time

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from ukroutes.common.logging_config import setup_logging

logger = setup_logging()


class Route:
    """
    A class to represent a route calculation using network graphs.

    Attributes:
        source (pd.DataFrame): DataFrame containing source nodes.
        target (pd.DataFrame): DataFrame containing target nodes.
        nodes (pd.DataFrame): DataFrame containing all nodes.
        edges (pd.DataFrame): DataFrame containing all edges.
    """

    def __init__(
        self,
        source: pd.DataFrame,
        target: pd.DataFrame,
        nodes: pd.DataFrame,
        edges: pd.DataFrame,
        weights: str = "time_weighted",
        k: int = 1,
    ):
        """
        Constructs all the necessary attributes for the Route object.

        Parameters:
        -----------
        source : pd.DataFrame
            DataFrame containing source nodes with easting and northing coordinates.
        target : pd.DataFrame
            DataFrame containing target nodes with easting and northing coordinates.
        nodes : pd.DataFrame
            DataFrame containing all nodes with easting and northing coordinates.
        edges : pd.DataFrame
            DataFrame containing edges between nodes.
        weights : str, optional
            The type of weight to use for edges (default is "time_weighted").
        k : int, optional
            The number of nearest neighbors to consider (default is 1).
        """
        self.source = source
        self.nodes = nodes
        self.edges = edges
        self.target = target
        self.weights = weights
        self.k = k

        logger.info("Initializing Route class")
        self.build()

    def add_to_graph(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds nodes from the given DataFrame to the graph and creates edges to the nearest nodes.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing nodes to be added with easting and northing coordinates.

        Returns:
        --------
        pd.DataFrame
            DataFrame with added nodes including their new node IDs.
        """
        nodes_tree = KDTree(self.nodes[["easting", "northing"]].values)  # type: ignore
        distances, indices = nodes_tree.query(
            df[["easting", "northing"]].values,  # type: ignore
            k=self.k,
        )

        nearest_nodes_df = pd.DataFrame(
            {
                "nearest_node": self.nodes.iloc[indices.flatten()]["node_id"].values,
                "distance": distances.flatten() + 0.01,
            }
        )

        new_node_ids = np.arange(len(self.nodes) + 1, len(self.nodes) + 1 + len(df))
        df["node_id"] = new_node_ids
        new_nodes = df[["node_id", "easting", "northing"]]
        self.nodes = pd.concat([self.nodes, new_nodes], ignore_index=True)

        new_edges = pd.DataFrame(
            {
                "start_node": np.repeat(df["node_id"], self.k),
                "end_node": nearest_nodes_df["nearest_node"],
                "length": nearest_nodes_df["distance"],
            }
        )

        if self.weights == "time_weighted":
            new_edges[self.weights] = (
                (new_edges["length"].astype(float) / 1000) / 25 * 1.609344 * 60
            )
        elif self.weights == "pedestrian_time":
            new_edges[self.weights] = (
                (new_edges["length"].astype(float) / 1000) / 5 * 60
            )
        elif self.weights == "distance":
            new_edges[self.weights] = new_edges["length"].astype(float) / 1000

        self.edges = pd.concat([self.edges, new_edges], ignore_index=True)
        return df

    def build(self) -> None:
        """
        Builds the graph by adding source and target nodes.
        """
        logger.info("Building graph with source and target nodes")
        self.source = self.add_to_graph(self.source)
        logger.info("Source nodes added to graph")
        self.target = self.add_to_graph(self.target)
        logger.info("Target nodes added to graph")

    def route(self) -> pd.DataFrame:
        """
        Calculates the shortest path distances from source to target nodes.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing the shortest path distances from source to target nodes.
        """
        logger.info("Starting routing...")
        G = nx.from_pandas_edgelist(
            self.edges,
            source="start_node",
            target="end_node",
            edge_attr=self.weights,
        )

        t1 = time.time()
        distances = nx.multi_source_dijkstra_path_length(
            G, sources=self.source["node_id"].to_list(), weight=self.weights
        )
        t2 = time.time()
        logger.info(f"Routing took {t2 - t1:.2f} seconds")

        distances = pd.DataFrame(
            {"node_id": distances.keys(), self.weights: distances.values()}
        )
        distances = distances[distances["node_id"].isin(self.target["node_id"])]
        distances = self.target.merge(distances, on="node_id")
        logger.info("Distances calculated and merged with target nodes")
        return distances
