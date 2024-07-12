# UKRoutes

This project uses `cugraph`, a RAPIDS library, to estimate drive times from source nodes to destination nodes on the UK road network.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Sources](#data-sources)
- [Contributing](#contributing)
- [Licence](#licence)

## Introduction

This project aims to provide an efficient way to estimate drive times across the UK road network. By leveraging the GPU-accelerated graph analytics capabilities of `cugraph`, we can handle large-scale graph data and perform drive time estimations quickly.

## Features

- **GPU-accelerated computation**: Utilises `cugraph` for high-performance graph analytics.
- **Drive time estimation**: Calculates estimated drive times between given source and destination nodes.
- **Scalability**: Capable of handling large-scale road networks.

## Installation

### Requirements

This project requires a CUDA-enabled GPU, and Python >=3.10 and <=3.11.

You may install this project directly with pip using:

```bash
pip install git+https://github.com/cjber/ukroutes
```

Alternatively you can install the project locally, using the following steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/cjber/ukroutes.git
    cd ukroutes
    ```

2. Set up a Python virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.lock
    ```

## Usage

This project requires several data files that cannot be redistributed.

* OS Open Roads - [download](https://www.ordnancesurvey.co.uk/products/os-open-roads)
* ONS Postcodes - [download](https://geoportal.statistics.gov.uk/search?q=PRD_ONSPD&sort=Date%20Created%7Ccreated%7Cdesc)

The following gives an example script that would be used to find the nearest 'health' assets to all Postcodes within the UK.

```python
import cudf
import geopandas as gpd
import pandas as pd

from ukroutes import Routing
from ukroutes.common.utils import Paths
from ukroutes.process_routing import add_to_graph, add_topk
from ukroutes.oproad.utils import process_oproad

# process oproad nodes and edges
nodes, edges = process_oproad(outdir=Paths.OS_GRAPH)  # or outdir=None

# read in health dataa and postcodes
health = pd.read_parquet("./data/processed/health.parquet").dropna().sample(1000)
postcodes = pd.read_csv(
    "./data/raw/onspd/ONSPD_FEB_2024.csv",
    usecols=["PCD", "OSEAST1M", "OSNRTH1M", "DOTERM", "CTRY"],
)
postcodes = (
    postcodes[
        (postcodes["DOTERM"].isnull())
        & (~postcodes["CTRY"].isin(["N92000002", "L93000001", "M83000003"]))
    ]
    .drop(columns=["DOTERM", "CTRY"])
    .rename({"PCD": "postcode", "OSEAST1M": "easting", "OSNRTH1M": "northing"}, axis=1)
    .dropna()
)

# add health and postcodes to road network
health, nodes, edges = add_to_graph(health, nodes, edges, 1)
postcodes, nodes, edges = add_to_graph(postcodes, nodes, edges, 1)

# find the top 10 closest health facilities to each postcode
health = add_topk(health, postcodes, 10)

# run the routing class
routing = Routing(
    edges=edges,
    nodes=nodes,
    outputs=postcodes,
    inputs=health,
    weights="time_weighted",
    min_buffer=5000,
    max_buffer=500_000,
    cutoff=300,
)
routing.fit()

# join distances to postcodes
distances = (
    routing.distances.set_index("vertex")
    .join(cudf.from_pandas(postcodes).set_index("node_id"), how="right")
    .reset_index()
    .to_pandas()
)
```

This example code produces the following result:

```python
import matplotlib.pyplot as plt

distances = gpd.GeoDataFrame(
    distances, geometry=gpd.points_from_xy(distances.easting, distances.northing)
)
distances.reset_index().sort_values("distance").plot(column="distance")

plt.show()
```

![](./figs/health_example.png)
