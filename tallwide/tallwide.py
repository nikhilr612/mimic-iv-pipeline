"""
Script to convert MIMIC-IV data from "tall" format to "wide" format.
Requires preprocessed cohort chart events in the database.
"""

import duckdb
import pandas as pd
import logging
from jinja2 import Template

# Global configs.
TABLE_NAME = "preproc_chart_icu"
OUTPUT_PATH = "./data/cohort/widechart.parquet"
TEMPLATE_FILE = __file__.replace("tallwide.py", "sqltemplate.jinja")
CHART_ITEMIDS_FILE = "./utils/chart_itemids.txt"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Loading DuckDB database.")
    conn = duckdb.connect(database="../mimiciv.duckdb")
    logger.info("Reading SQL template.")
    sql_template: Template
    with open(TEMPLATE_FILE, "r") as file:
        sql_template = Template(file.read())
        itemids = []

    with open(CHART_ITEMIDS_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):  # Skip empty lines and comments
                try:
                    itemids.append(int(line))
                except ValueError:
                    logger.warning(
                        f"Invalid itemid '{line}' in chart event whitelist - skipping"
                    )

    itemid_set = set(itemids) if itemids else None

    # Warn if filter is empty
    if not itemids:
        logger.warning("Empty whitelist loaded.")

    query = sql_template.render(
        table_name=TABLE_NAME,
        itemid_whitelist=itemids,
    )

    logger.debug(f"Executing query:\n{query}")
    df: pd.DataFrame = conn.execute(query).df()
    logger.info(f"Query returned {len(df)} rows and {len(df.columns)} columns.")
    logger.info(f"Writing output to {OUTPUT_PATH}.")
    df.to_parquet(OUTPUT_PATH, index=False)
    logger.info("Done.")
