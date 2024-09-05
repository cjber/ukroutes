![](./figs/logo/ukroutes-high-resolution-logo-transparent.png)

> This project uses `networkx` to estimate drive times from source nodes to destination nodes on the UK road network.

## Installation

### Requirements

You may install this project directly with pip (or similar) using:

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
* ONS Postcodes - [download](https://geoportal.statistics.gov.uk/search?q=PRD_ONSPD&sort=Date%20Created%7Ccreated%7Cdesc) (Or similar target data)
* Ferry Routes - [download](http://overpass-turbo.eu/?q=LyoKVGhpcyBoYcSGYmVlbiBnxI1lcmF0ZWQgYnkgdGhlIG92xJJwxIlzLXR1cmJvIHdpemFyZC7EgsSdxJ9yaWdpbmFsIHNlxLBjaMSsxIk6CsOiwoDCnHJvdcSVPWbEknJ5xYjCnQoqLwpbxYx0Ompzb25dW3RpbWXFmzoyNV07Ci8vxI_ElMSdciByZXN1bHRzCigKICDFryBxdcSSxJrEo3J0IGZvcjogxYjFisWbZcWPxZHFk8KAxZXGgG5vZGVbIsWLxY1lIj0ixZByxZIiXSh7e2LEqnh9fSnFrcaAd2F5xp_GocSVxqTGpsaWxqrGrMauxrDGssa0xb_FtWVsxJRpxaDGusaTxr3Gp8apxqvGrcavb8axxrPFrceFxoJwxLduxorFtsW4xbrFvMWbxJjGnHnFrT7Frcejc2vHiMaDdDs&c=BH1aTWQmgG)

The Python script `scripts/demo.py` gives a simple overview of this library;

```python
import pandas as pd
from tqdm import tqdm

from ukroutes.common.utils import Paths
from ukroutes.routing import Route


def process_dentists():
    dentists_eng = pd.read_csv(Paths.RAW / "dentists_england.csv")
    dentists_scot = pd.read_csv(Paths.RAW / "dentists_scotland.csv")
    postcodes = pd.read_parquet(Paths.PROCESSED / "onspd" / "postcodes.parquet")

    dentists_eng["postcode"] = dentists_eng["postcode"].str.replace(" ", "")
    dentists_scot["postcode"] = dentists_scot["postcode"].str.replace(" ", "")

    dentists = pd.concat([dentists_eng, dentists_scot])
    dentists = dentists.merge(postcodes, on="postcode")
    dentists.drop(columns="postcode").to_parquet(Paths.PROCESSED / "dentists.parquet")


process_dentists()

postcodes = pd.read_parquet(Paths.PROCESSED / "postcodes.parquet")
nodes = pd.read_parquet(Paths.PROCESSED / "oproad" / "nodes.parquet")
edges = pd.read_parquet(Paths.PROCESSED / "oproad" / "edges.parquet")

pq_files = list(Paths.PROCESSED.glob("*.parquet"))
for file in tqdm(pq_files):
    source = pd.read_parquet(file).dropna(subset=["easting", "northing"])
    route = Route(source=source, target=postcodes, nodes=nodes, edges=edges)
    distances = route.route()
    distances.to_parquet(Paths.OUT / f"{file.stem}_distances.parquet")
```

# Routing Methodology

The primary goal of this project is to determine the distance of points of interest to each postcode within Great Britain. Given there are over 1.7 million postcodes, instead of routing from each postcode to each point of interest, the processing is inverted, routing from points of interest to all nodes in a graph, these nodes are then filtered to find postcodes. The following gives an overview of the sequential processing involved to achieve this.

1. **Process the OS Open Road Network**

Ordnance Survey publish road speed estimates alongside their road network documentation. These estimates are used to provide average speed estimates and subsequent drive-time estimates using the length of `linestring` geometries. For example the road speed estimate for all motorways is 67mph, while for single carriageway A and B roads the estimate is 25mph. These speeds are converted to drive-time in minutes using the road length.

OS Open Roads does not include ferry routes. These were therefore taken from OpenStreetMap (OSM), using the Overpass API (http://overpass-turbo.eu) with the query found [here](http://overpass-turbo.eu/?q=LyoKVGhpcyBoYcSGYmVlbiBnxI1lcmF0ZWQgYnkgdGhlIG92xJJwxIlzLXR1cmJvIHdpemFyZC7EgsSdxJ9yaWdpbmFsIHNlxLBjaMSsxIk6CsOiwoDCnHJvdcSVPWbEknJ5xYjCnQoqLwpbxYx0Ompzb25dW3RpbWXFmzoyNV07Ci8vxI_ElMSdciByZXN1bHRzCigKICDFryBxdcSSxJrEo3J0IGZvcjogxYjFisWbZcWPxZHFk8KAxZXGgG5vZGVbIsWLxY1lIj0ixZByxZIiXSh7e2LEqnh9fSnFrcaAd2F5xp_GocSVxqTGpsaWxqrGrMauxrDGssa0xb_FtWVsxJRpxaDGusaTxr3Gp8apxqvGrcavb8axxrPFrceFxoJwxLduxorFtsW4xbrFvMWbxJjGnHnFrT7Frcejc2vHiMaDdDs&c=BH1aTWQmgG). `KDTree` from `scipy.spatial` is then used to determine the nearest road node point to the start and end location of these routes, allowing for them to be added directly to the road network. The speed estimate for these routes is 25mph, around the speed of an average ferry.

Despite the addition of ferry routes connecting isolated road networks on islands to the mainland, there were still road nodes that did not connect directly to the road network. These did not appear to follow any pattern; distributed evenly across GB. These were therefore removed after being identified using the `nx.connected_components()` function.

2. **Add Postcodes and POIs to the road network**

The `add_to_graph` method creates new nodes at the location of a collection of easting and northing coordinates. These nodes are then added to the road network by generating a new edge between this point and the nearest `k` road nodes using a `KDTree`, with a speed estimate of 25mph.

3. **Routing from POIs to postcodes**

While the interest is in determining the distance from postcodes to POIs, the previous processing allows for a large speed-up by considering the reverse of this task. The `Route` class in `routing.py` primarily routes using the Multi Source Shortest Path `nx.multi_source_dijkstra` algorithm, which allows for weighted routing from points of interest to all other nodes in a graph. This approach means that for each node associated with a postcode, the minimum returned distance indicates the nearest POI by drive-time.
