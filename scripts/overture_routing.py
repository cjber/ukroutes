from pathlib import Path

import cudf
import cuspatial
import geopandas as gpd
import pandas as pd

from ukroutes import Routing
from ukroutes.common.utils import Paths

edges: cudf.DataFrame = cudf.read_parquet(Paths.OS_GRAPH / "edges.parquet")  # type:ignore
nodes: pd.DataFrame = pd.read_parquet(Paths.OS_GRAPH / "nodes.parquet")  # type:ignore
nodes: cuspatial.GeoDataFrame = cuspatial.from_geopandas(  # type:ignore
    gpd.GeoDataFrame(  # type:ignore
        nodes, geometry=gpd.points_from_xy(nodes["easting"], nodes["northing"])
    )
)
postcodes: cudf.DataFrame = cudf.read_parquet(  # type:ignore
    Paths.PROCESSED / "postcodes.parquet"
)

for file in Path("./data/processed/overture").glob("*.parquet"):
    name = file.stem
    overture: pd.DataFrame = pd.read_parquet(file)
    routing = Routing(
        name=name,
        edges=edges,
        nodes=nodes,
        outputs=postcodes,
        inputs=overture,
        weights="time_weighted",
    )
    routing.fit()

    distances = (
        routing.distances.set_index("vertex")
        .join(postcodes.set_index("node_id"), how="right")
        .reset_index()
    )
    OUT_FILE = Paths.OUT_DATA / f"distances_{name}.csv"
    distances.to_pandas()[["postcode", "distance"]].to_csv(OUT_FILE, index=False)
