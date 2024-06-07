from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import NamedTuple

import cudf
import cugraph
import cupy as cp
import pandas as pd
from rich.progress import track
from sqlalchemy import create_engine

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
        nodes: cudf.DataFrame,
        sources: cudf.DataFrame,
        targets: pd.DataFrame,
        weights: str = "time_weighted",
        min_buffer: int = 5_000,
        max_buffer: int = 1_000_000,
        cutoff: int | None = None,
    ):
        self.name: str = name
        self.sources: cudf.DataFrame = sources
        self.targets: pd.DataFrame = targets

        self.road_edges: cudf.DataFrame = edges
        self.road_nodes: cudf.GeoDataFrame = nodes
        self.weights: str = weights
        self.min_buffer: int = min_buffer
        self.max_buffer: int = max_buffer
        self.cutoff: int = cutoff

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.graph = cugraph.Graph()
            self.graph.from_cudf_edgelist(
                self.road_edges,
                source="start_node",
                destination="end_node",
                edge_attr=self.weights,
                renumber=False,
            )

        self.distances: cudf.DataFrame = cudf.DataFrame()

        db_path = Path("distances.db")
        if db_path.exists():
            db_path.unlink()
        self.engine = create_engine(f"sqlite:///{db_path}")

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
        logger.debug(f"Routing complete for {self.name} in {tdiff / 60:.2f} minutes.")

    def create_sub_graph(self, target) -> cugraph.Graph:
        buffer = max(self.min_buffer, target.buffer)
        while True:
            nodes_subset = self.road_nodes.copy()
            nodes_subset["distance"] = cp.sqrt(
                (nodes_subset["easting"] - target.easting) ** 2
                + (nodes_subset["northing"] - target.northing) ** 2
            )
            nodes_subset = nodes_subset[nodes_subset["distance"] <= buffer]

            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=FutureWarning)
                sub_graph = cugraph.subgraph(self.graph, nodes_subset["node_id"])
                sub_graph = self._remove_partial_graphs(sub_graph)

                if sub_graph is None:
                    if buffer >= self.max_buffer:
                        sub_graph = self.graph
                        return None
                    buffer = buffer * 2
                    continue

            ntarget_nds = cudf.Series(target.top_nodes).isin(sub_graph.nodes()).sum()
            df_node = target.node_id in sub_graph.nodes().to_arrow().to_pylist()

            if df_node & (ntarget_nds == len(target.top_nodes)) or buffer >= 1_000_000:
                return sub_graph
            buffer = buffer * 2

    def _remove_partial_graphs(self, sub_graph):
        components = cugraph.connected_components(sub_graph)
        component_counts = components["labels"].value_counts().reset_index()
        component_counts.columns = ["labels", "count"]

        largest_component_label = component_counts[
            component_counts["count"] == component_counts["count"].max()
        ]["labels"][0]

        largest_component_nodes = components[
            components["labels"] == largest_component_label
        ]["vertex"]
        nodes_subset = self.road_nodes[
            self.road_nodes["node_id"].isin(largest_component_nodes)
        ]
        return cugraph.subgraph(self.graph, nodes_subset["node_id"])

    def get_shortest_dists(self, target: NamedTuple) -> None:
        sub_graph = self.create_sub_graph(target=target)
        if sub_graph is None:
            return
        shortest_paths: cudf.DataFrame = cugraph.filter_unreachable(
            cugraph.sssp(sub_graph, source=target.node_id, cutoff=self.cutoff)
        )
        pc_dist = shortest_paths[shortest_paths.vertex.isin(self.sources["node_id"])]
        pc_dist.to_pandas().to_sql("distances", self.engine, if_exists="append")

    def fetch_distances(self):
        return (
            pd.read_sql("distances", self.engine)
            .sort_values("distance")
            .drop_duplicates("vertex")
            .reset_index()[["vertex", "distance"]]
        )
