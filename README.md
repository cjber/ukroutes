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

To get started with this project, follow these steps:

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

    Ensure you have CUDA and the RAPIDS libraries installed. For detailed instructions, refer to the [RAPIDS installation guide](https://rapids.ai/start.html).

## Usage

TODO: Implement something similar

To estimate drive times, you need a road network graph dataset in a compatible format. Here's a basic example to get you started:

1. Prepare your road network data. Ensure you have the node and edge data ready.

2. Run the drive time estimation script:
    ```bash
    python estimate_drive_time.py --source_nodes sources.csv --destination_nodes destinations.csv --edges edges.csv --nodes nodes.csv
    ```

    Replace `sources.csv`, `destinations.csv`, `edges.csv`, and `nodes.csv` with your actual data files.

### Example

Here is a simple example script to illustrate usage:

```python
import cudf
import cugraph

# Load the data
nodes = cudf.read_csv('nodes.csv')
edges = cudf.read_csv('edges.csv')

# Create the graph
G = cugraph.Graph()
G.from_cudf_edgelist(edges, source='source', destination='destination', edge_attr='weight')

# Perform shortest path calculation
source_nodes = cudf.Series([0, 1, 2])
dest_nodes = cudf.Series([3, 4, 5])
drive_times = cugraph.shortest_path(G, source_nodes, dest_nodes)

print(drive_times)
```
# Data Sources

The project relies on road network data which can be sourced from:

    Ordnance Survey
    OpenStreetMap (Ferry routes)
    ONS Postcodes

Ensure your data is preprocessed and formatted correctly before use.

# Contributing

Contributions are welcome. Please follow these steps to contribute:

    Fork the repository.
    Create a new branch: git checkout -b feature-name.
    Commit your changes: git commit -m 'Add new feature'.
    Push to the branch: git push origin feature-name.
    Open a pull request.

# Licence

This project is licensed under the MIT Licence. See the LICENCE file for details.
