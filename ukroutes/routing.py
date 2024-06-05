from __future__ import annotations
import cupy as cp

import time
import warnings
from typing import NamedTuple

import cudf
import cugraph
import cuspatial
import pandas as pd
from rich.progress import track

from ukroutes.common.logger import logger


class Routing:
    """
    Main class for calculating routing from POI to postcodes within a road network.

    Primarily uses `cugraph` to GPU accelerate routing. While the interest is distance
    from postcodes to POI, this class does routing from POI to postcodes, appending to
    a large intermediate file. When complete the routing takes the minimum distance
    for each postcode.

    Parameters
    ----------
    name : str
        Name of POI
    edges : cudf.DataFrame
        Dataframe containing road edges
    nodes : cudf.DataFrame
        Dataframe containing road nodes
    postcodes : cudf.DataFrame
        Dataframe containing all postcodes
    pois : pd.DataFrame
        Dataframe containing all POIs
    weights : str
        Graph weights to use, e.g. `time_weighted` or `distance`
    """

    def __init__(
        self,
        name: str,
        edges: cudf.DataFrame,
        nodes: cuspatial.GeoDataFrame,
        sources: cudf.DataFrame,
        targets: pd.DataFrame,
        weights: str = "time_weighted",
        buffer: int = 100_000,
    ):
        self.name: str = name
        self.sources: cudf.DataFrame = sources
        self.targets: pd.DataFrame = targets

        self.road_edges: cudf.DataFrame = edges
        self.road_nodes: cuspatial.GeoDataFrame = nodes
        self.weights: str = weights
        self.buffer: int = buffer

        self.graph = cugraph.Graph()
        self.graph.from_cudf_edgelist(
            self.road_edges,
            source="start_node",
            destination="end_node",
            edge_attr=self.weights,
            renumber=True,
        )

        self.distances: cudf.DataFrame = cudf.DataFrame()

    def fit(self) -> None:
        """
        Iterate and apply routing to each POI

        This function primarily allows for the intermediate steps in routing to be
        logged. This means that if the routing is stopped midway it can be restarted.
        """
        t1 = time.time()
        for target in track(
            self.targets.itertuples(),
            description=f"Processing {self.name}...",
            total=len(self.targets),
        ):
            self.get_shortest_dists(target)
        t2 = time.time()
        tdiff = t2 - t1
        logger.debug(
            f"Routing complete for {self.name} in {tdiff / 60:.2f} minutes,"
            " finding minimum distances."
        )

    def create_sub_graph(self, target) -> cugraph.Graph:
        buffer = self.buffer
        nodes_subset = self.road_nodes
        nodes_subset["distance"] = cp.sqrt(
            (nodes_subset["easting"] - target.easting) ** 2
            + (nodes_subset["northing"] - target.northing) ** 2
        )
        nodes_subset = nodes_subset[nodes_subset["distance"] <= self.buffer]
        return cugraph.subgraph(self.graph, nodes_subset["node_id"])

    def get_shortest_dists(self, target: NamedTuple) -> None:
        if self.buffer:
            sub_graph = self.create_sub_graph(target=target)
        else:
            sub_graph = self.graph

        shortest_paths: cudf.DataFrame = cugraph.filter_unreachable(  # type: ignore
            cugraph.sssp(sub_graph, source=target.node_id)  # type:ignore
        )
        pc_dist = shortest_paths[shortest_paths.vertex.isin(self.sources["node_id"])]

        self.distances = cudf.concat([self.distances, pc_dist])

        self.distances = (
            self.distances.sort_values("distance")
            .drop_duplicates("vertex")
            .reset_index()[["vertex", "distance"]]
        )
