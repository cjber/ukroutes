from pathlib import Path

# import polars as pl
#
# pl.Config.set_tbl_formatting("NOTHING")
# pl.Config.with_columns_kwargs = True
# pl.Config.set_tbl_dataframe_shape_below(True)
# pl.Config.set_tbl_rows(6)


class Paths:
    DATA = Path("data")
    RAW = DATA / "raw"
    OUT_DATA = DATA / "out"
    OPROAD = RAW / "oproad" / "oproad_gb.gpkg"
    FERRY = RAW / "oproad" / "strtgi_essh_gb" / "ferry_line.shp"

    PROCESSED = DATA / "processed"
    OS_GRAPH = PROCESSED / "osm"
