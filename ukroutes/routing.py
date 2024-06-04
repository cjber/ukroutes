from __future__ import annotations

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
        buffer: int = 5_000,
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
        """
        Create a subgraph of road nodes based on buffer distance

        The subgraph is created using euclidean distance and
        `cuspatial.points_in_spatial_window`. If buffers are not large enough to
        include all nodes identified as important to that particular POI, it is
        increased in size.

        Parameters
        ----------
        poi : namedtuple
            Single POI created by `df.itertuples()`

        Returns
        -------
        cugraph.Graph:
            Graph object that is a subset of all road nodes
        """
        buffer = max(target.buffer, self.buffer)
        while True:
            node_subset = cuspatial.points_in_spatial_window(
                points=self.road_nodes["geometry"],  # type: ignore
                min_x=target.easting - buffer,
                max_x=target.easting + buffer,
                min_y=target.northing - buffer,
                max_y=target.northing + buffer,
            )
            node_subset = cudf.DataFrame(
                {"easting": node_subset.points.x, "northing": node_subset.points.y}
            ).merge(
                cudf.DataFrame(self.road_nodes.drop("geometry", axis=1)),
                on=["easting", "northing"],
            )

            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=FutureWarning)
                sub_graph = cugraph.subgraph(self.graph, node_subset["node_id"])  # type: ignore

            if sub_graph is None:
                continue

            pc_nodes = cudf.Series(target.pc_node).isin(sub_graph.nodes()).sum()
            poi_node = sub_graph.nodes().isin([target.node_id]).sum()

            # ensure all postcode nodes in + poi node
            # don't incrase buffer for large pois lists
            if poi_node & (pc_nodes == len(target.pc_node)):
                return sub_graph
            buffer = buffer * 2

    def get_shortest_dists(self, target: NamedTuple) -> None:
        """
        Use `cugraph.sssp` to calculate shortest paths from POI to postcodes

        First subsets road graph, then finds shortest paths, ensuring all paths are
        routed that are known to be important to each POI. Saves to `hdf` to allow
        restarts.

        Parameters
        ----------
        poi : namedtuple
            Single POI created from `df.itertuples()`
        """
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
