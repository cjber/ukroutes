from __future__ import annotations

import time
import warnings
from typing import NamedTuple

import cudf
import cugraph
import cupy as cp
from rich.progress import track

from ukroutes.common.logger import logger


class Routing:
    def __init__(
        self,
        name: str,
        edges: cudf.DataFrame,
        nodes: cudf.DataFrame,
        outputs: cudf.DataFrame,
        inputs: cudf.DataFrame,
        weights: str = "time_weighted",
        min_buffer: int = 5_000,
        max_buffer: int = 1_000_000,
        cutoff: int | None = None,
    ):
        self.name: str = name
        self.outputs: cudf.DataFrame = outputs
        self.inputs: cudf.DataFrame = inputs

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

    def fit(self) -> None:
        """
        Iterate and apply routing to each POI

        This function primarily allows for the intermediate steps in routing to be
        logged. This means that if the routing is stopped midway it can be restarted.
        """
        t1 = time.time()
        process_df = (
            self.inputs  # if len(self.inputs) < len(self.outputs) else self.outputs
        )
        for item in track(
            process_df.itertuples(),
            description=f"Processing {self.name}...",
            total=len(process_df),
        ):
            self.get_shortest_dists(item)
        t2 = time.time()
        tdiff = t2 - t1
        logger.debug(f"Routing complete for {self.name} in {tdiff / 60:.2f} minutes.")

    def create_sub_graph(self, item) -> cugraph.Graph:
        buffer = max(self.min_buffer, item.buffer)
        while True:
            nodes_subset = self.road_nodes.copy()
            nodes_subset["distance"] = cp.sqrt(
                (nodes_subset["easting"] - item.easting) ** 2
                + (nodes_subset["northing"] - item.northing) ** 2
            )
            nodes_subset = nodes_subset[nodes_subset["distance"] <= buffer]

            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=FutureWarning)
                sub_graph = cugraph.subgraph(self.graph, nodes_subset["node_id"])
                main_sub_graph = self._remove_partial_graphs(sub_graph)

                if main_sub_graph is None:
                    if buffer >= self.max_buffer:
                        main_sub_graph = self.graph
                        return None
                    buffer = buffer * 2
                    continue

            ntarget_nds = cudf.Series(item.top_nodes).isin(main_sub_graph.nodes()).sum()
            df_node = item.node_id in main_sub_graph.nodes().to_arrow().to_pylist()

            if (
                df_node & (ntarget_nds == len(item.top_nodes))
                or buffer >= self.max_buffer
            ):
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

    def get_shortest_dists(self, item: NamedTuple) -> None:
        sub_graph = self.create_sub_graph(item=item)
        if sub_graph is None:
            return
        spaths: cudf.DataFrame = cugraph.filter_unreachable(
            cugraph.sssp(sub_graph, source=item.node_id, cutoff=self.cutoff)
        )
        # if len(self.inputs) > len(self.outputs):
        min_dist = (
            spaths[spaths.vertex.isin(self.inputs["node_id"])]
            .sort_values("distance")
            .iloc[0]
        )
        dist = cudf.DataFrame(
            {"vertex": [item.node_id], "distance": [min_dist["distance"]]}
        )
        # else:
        #     dist = spaths[spaths.vertex.isin(self.outputs["node_id"])]
        self.distances = cudf.concat([self.distances, dist])
        self.distances = (
            self.distances.sort_values("distance")
            .drop_duplicates("vertex")
            .reset_index()[["vertex", "distance"]]
        )
