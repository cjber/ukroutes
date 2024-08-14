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

pq_files = list(Paths.PROCESSED.glob("*.parquet"))
for file in tqdm(pq_files):
    route = Route(file)
    distances = route.route()
    distances.to_parquet(Paths.OUT / f"{file.stem}_distances.parquet")
