import numpy as np
import pandas as pd
import sys, os
import re
import ast
import datetime as dt
import logging
from typing import Dict, Optional, Set
from tqdm import tqdm

from sklearn.preprocessing import MultiLabelBinarizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


########################## ITEMID FILTERING ##########################
# Global mutable container to store whitelisted itemids for different data types
_ITEMID_WHITELISTS: Dict[str, Optional[Set[int]]] = {
    "chart": None,
    "med": None,
    "proc": None,
    "output": None,
}


def load_itemid_whitelist(data_type="chart", filepath=None):
    """Load itemid whitelist from file for specific data type.

    Args:
        data_type: "chart", "med", "proc", or "output"
        filepath: Custom file path, or None to use defaults
    """
    global _ITEMID_WHITELISTS

    logger.info(f"[ITEMID] Loading {data_type} itemid whitelist...")

    if filepath is None:
        # Try default locations for each data type
        default_files = {
            "chart": ["./utils/chart_itemids.txt", "./chart_itemids.txt"],
            "med": ["./utils/med_itemids.txt", "./med_itemids.txt"],
            "proc": ["./utils/proc_itemids.txt", "./proc_itemids.txt"],
            "output": ["./utils/output_itemids.txt", "./output_itemids.txt"],
        }

        # Also try generic itemid_whitelist.txt for backward compatibility
        generic_paths = [
            "./utils/itemid_whitelist.txt",
            "./itemid_whitelist.txt",
            os.path.join(os.path.dirname(__file__), "itemid_whitelist.txt"),
        ]

        filepath = None
        # First try specific file for data type
        for path in default_files.get(data_type, []):
            if os.path.exists(path):
                filepath = path
                break

        # If no specific file found, try generic
        if filepath is None:
            for path in generic_paths:
                if os.path.exists(path):
                    filepath = path
                    logger.info(
                        f"Using generic itemid whitelist for {data_type}: {filepath}"
                    )
                    break

    if filepath is None or not os.path.exists(filepath):
        logger.warning(
            f"No itemid whitelist file found for {data_type}. Including all itemids - no filtering will be applied."
        )
        return None

    try:
        itemids = []
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):  # Skip empty lines and comments
                    try:
                        itemids.append(int(line))
                    except ValueError:
                        logger.warning(
                            f"Invalid itemid '{line}' in {data_type} whitelist - skipping"
                        )

        itemid_set = set(itemids) if itemids else None

        # Warn if filter is empty
        if not itemids:
            logger.warning(
                f"Empty itemid whitelist loaded for {data_type} - no filtering will be applied"
            )

        # Store in global container by mutating in-place
        _ITEMID_WHITELISTS[data_type] = itemid_set

        logger.info(
            f"[ITEMID] Loaded {len(itemids)} {data_type} itemids from: {filepath}"
        )
        return itemid_set
    except Exception as e:
        logger.error(f"Error loading {data_type} itemid whitelist from {filepath}: {e}")
        return None


def get_itemid_whitelist(data_type="chart"):
    """Get the current itemid whitelist for specified data type."""
    global _ITEMID_WHITELISTS

    current_whitelist = _ITEMID_WHITELISTS.get(data_type)
    if current_whitelist is None:
        # Try to load it
        load_itemid_whitelist(data_type)
        # Get updated value after loading
        current_whitelist = _ITEMID_WHITELISTS.get(data_type)

    return current_whitelist


def get_itemid_filter_clause(data_type="chart", itemid_col="itemid"):
    """Get SQL WHERE clause for itemid filtering. Returns empty string if no filtering."""
    whitelist = get_itemid_whitelist(data_type)
    if whitelist is None or len(whitelist) == 0:
        logger.warning(
            f"No {data_type} itemid filter available - query will include all itemids"
        )
        return ""

    itemid_list = ",".join(str(id) for id in sorted(whitelist))
    return f"AND CAST({itemid_col} AS INTEGER) IN ({itemid_list})"


def filter_by_itemid(df, data_type="chart", itemid_col="itemid"):
    """Filter dataframe by whitelisted itemids if whitelist is available."""
    whitelist = get_itemid_whitelist(data_type)
    if whitelist is None:
        logger.warning(
            f"No {data_type} itemid filter available - returning unfiltered dataframe"
        )
        return df

    if itemid_col not in df.columns:
        logger.warning(
            f"Column '{itemid_col}' not found. Skipping {data_type} itemid filtering."
        )
        return df

    original_count = len(df)
    df_filtered = df[df[itemid_col].isin(whitelist)]
    filtered_count = len(df_filtered)

    logger.info(
        f"[FILTER] {data_type.title()} itemid filtering: {original_count} -> {filtered_count} rows ({original_count - filtered_count} removed)"
    )
    return df_filtered


def load_all_itemid_whitelists():
    """Load all itemid whitelists for all data types."""
    data_types = ["chart", "med", "proc", "output"]
    loaded = {}

    for data_type in data_types:
        whitelist = load_itemid_whitelist(data_type)
        loaded[data_type] = len(whitelist) if whitelist else 0

    logger.info(f"[ITEMID] Itemid whitelists loaded: {loaded}")
    return loaded


########################## GENERAL ##########################
def dataframe_from_query(conn, query):
    """Execute a DuckDB query and return a pandas DataFrame"""
    logger.info(f"[QUERY] Executing query...")
    df = conn.execute(query).df()
    logger.info(f"[QUERY] Query returned DataFrame with shape: {df.shape}")
    return df


def load_cohort_from_csv_gz(conn, csv_gz_path, table_name="cohort_temp"):
    """
    Load cohort data from a local CSV.gz file into a temporary DuckDB table.

    Args:
        conn: DuckDB connection object
        csv_gz_path: Path to the CSV.gz file containing cohort data
        table_name: Name for the temporary table (default: "cohort_temp")

    Returns:
        str: The table name that was created

    Raises:
        FileNotFoundError: If the CSV.gz file doesn't exist
        Exception: If there's an error loading the data
    """
    if not os.path.exists(csv_gz_path):
        raise FileNotFoundError(f"CSV.gz file not found: {csv_gz_path}")

    try:
        logger.info(
            f"[COHORT] Loading cohort data from {csv_gz_path} into temporary table '{table_name}'..."
        )

        # Drop the table if it already exists
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")

        # Load CSV.gz using pandas first for better compatibility
        logger.info("[COHORT] Reading CSV.gz file with pandas...")
        df = pd.read_csv(csv_gz_path, compression="gzip")
        logger.info(f"[COHORT] CSV DataFrame loaded with shape: {df.shape}")

        if df.empty:
            logger.warning("[COHORT] CSV file is empty")
            # Create empty table with basic structure
            conn.execute(f"""
                CREATE TABLE {table_name} (
                    subject_id INTEGER,
                    hadm_id INTEGER,
                    stay_id INTEGER,
                    intime TIMESTAMP,
                    outtime TIMESTAMP,
                    label INTEGER
                )
            """)
        else:
            # Register DataFrame as table in DuckDB
            conn.register(table_name, df)

        # Get row count for confirmation
        count_result = conn.execute(
            f"SELECT COUNT(*) as count FROM {table_name}"
        ).fetchone()
        row_count = count_result[0] if count_result else 0

        if row_count == 0:
            logger.warning(f"[COHORT] No data loaded from {csv_gz_path}")
        else:
            logger.info(
                f"[COHORT] Successfully loaded {row_count} rows into table '{table_name}'"
            )

        # Show table schema for debugging
        schema_result = conn.execute(f"DESCRIBE {table_name}").df()
        logger.info(f"[COHORT] Table schema:")
        logger.info(schema_result.to_string(index=False))

        # Validate required columns for cohort data
        required_columns = ["stay_id", "intime"]
        columns = schema_result["column_name"].tolist()
        missing_columns = [col for col in required_columns if col not in columns]

        if missing_columns:
            logger.warning(
                f"[COHORT] Missing recommended cohort columns: {missing_columns}"
            )
            logger.warning(
                "[COHORT] Expected columns for ICU cohort: stay_id, subject_id, hadm_id, intime, outtime"
            )

        return table_name

    except Exception as e:
        logger.error(f"[COHORT] Error loading cohort data from {csv_gz_path}: {str(e)}")
        # Try to clean up if table creation failed
        try:
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        except:
            pass
        raise


def load_cohort_from_dataframe(conn, df, table_name="cohort_temp"):
    """
    Load cohort data from a pandas DataFrame into a temporary DuckDB table.

    Args:
        conn: DuckDB connection object
        df: pandas DataFrame containing cohort data
        table_name: Name for the temporary table (default: "cohort_temp")

    Returns:
        str: The table name that was created
    """
    try:
        logger.info(
            f"[COHORT] Loading cohort data from DataFrame (shape: {df.shape}) into temporary table '{table_name}'..."
        )

        # Drop the table if it already exists
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")

        # Check if DataFrame is empty
        if df.empty:
            logger.warning("[COHORT] DataFrame is empty, creating table with no rows")
            # Create an empty table with basic columns
            conn.execute(f"""
                CREATE TABLE {table_name} (
                    subject_id INTEGER,
                    hadm_id INTEGER,
                    stay_id INTEGER,
                    intime TIMESTAMP,
                    outtime TIMESTAMP,
                    label INTEGER
                )
            """)
        else:
            # Register the DataFrame as a temporary table
            conn.register(table_name, df)

        # Get row count for confirmation
        if df.empty:
            row_count = 0
        else:
            row_count = len(df)
        logger.info(
            f"[COHORT] Successfully loaded {row_count} rows into table '{table_name}'"
        )

        # Show table schema for debugging
        schema_result = conn.execute(f"DESCRIBE {table_name}").df()
        logger.info(f"[COHORT] Table schema:")
        logger.info(schema_result.to_string(index=False))

        return table_name

    except Exception as e:
        logger.error(f"[COHORT] Error loading cohort data from DataFrame: {str(e)}")
        raise


def preproc_with_csv_cohort(
    conn,
    csv_gz_path,
    data_type="chart",
    chart_table_name="chartevents",
    procedure_table_name="procedureevents_mv",
    output_table_name="outputevents",
    time_col="charttime",
    chunksize=10000000,
    cohort_table_name="cohort_temp",
    create_table=True,
):
    """
    Convenience function to load cohort from CSV.gz and preprocess data in one step.

    Args:
        conn: DuckDB connection object
        csv_gz_path: Path to the CSV.gz file containing cohort data
        data_type: Type of data to preprocess ("chart", "med", "proc", "output")
        chart_table_name: Name of chart events table (default: "chartevents")
        procedure_table_name: Name of procedure events table (default: "procedureevents_mv")
        output_table_name: Name of output events table (default: "outputevents")
        time_col: Time column name (default: "charttime")
        chunksize: Chunk size for processing large datasets (default: 10000000)
        cohort_table_name: Name for temporary cohort table (default: "cohort_temp")

    Returns:
        pd.DataFrame: Preprocessed data
    """
    logger.info(
        f"[PREPROC] Starting preprocessing with CSV cohort for data type: {data_type}"
    )

    # Load cohort data from CSV.gz into temporary table
    cohort_table = load_cohort_from_csv_gz(conn, csv_gz_path, cohort_table_name)

    # Preprocess data based on type
    if data_type == "chart":
        result = preproc_chart(
            conn, chart_table_name, cohort_table, time_col, chunksize
        )
    elif data_type == "med":
        result = preproc_meds(conn, cohort_table)
    elif data_type == "proc":
        result = preproc_proc(conn, procedure_table_name, cohort_table, time_col)
    elif data_type == "output":
        result = preproc_out(conn, output_table_name, cohort_table, time_col)
    else:
        raise ValueError(
            f"Unsupported data_type: {data_type}. Supported types: 'chart', 'med', 'proc', 'output'"
        )

    logger.info(
        f"[PREPROC] Completed preprocessing, final result shape: {result.shape}"
    )
    return result


def cleanup_cohort_table(conn, table_name="cohort_temp"):
    """
    Clean up temporary cohort table from DuckDB.

    Args:
        conn: DuckDB connection object
        table_name: Name of the temporary table to drop (default: "cohort_temp")
    """
    try:
        # Try to drop both table and view (DuckDB might register DataFrames as views)
        try:
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        except:
            conn.execute(f"DROP VIEW IF EXISTS {table_name}")
        logger.info(f"[CLEANUP] Successfully cleaned up temporary table '{table_name}'")
    except Exception as e:
        logger.warning(f"[CLEANUP] Could not clean up table '{table_name}': {str(e)}")


def read_admissions_table(conn):
    """Read admissions table from DuckDB with appropriate type casting"""
    logger.info("[ADMISSIONS] Reading admissions table...")
    query = """
    SELECT
        CAST(subject_id AS INTEGER) as subject_id,
        CAST(hadm_id AS INTEGER) as hadm_id,
        CAST(admittime AS TIMESTAMP) as admittime,
        CAST(dischtime AS TIMESTAMP) as dischtime,
        CAST(deathtime AS TIMESTAMP) as deathtime,
        ethnicity
    FROM admissions
    """
    admits = dataframe_from_query(conn, query)
    admits = admits[
        ["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "ethnicity"]
    ]
    admits.admittime = pd.to_datetime(admits.admittime)
    admits.dischtime = pd.to_datetime(admits.dischtime)
    admits.deathtime = pd.to_datetime(admits.deathtime)
    logger.info(f"[ADMISSIONS] Loaded admissions data with shape: {admits.shape}")
    return admits


def read_patients_table(conn):
    """Read patients table from DuckDB with appropriate type casting"""
    logger.info("[PATIENTS] Reading patients table...")
    query = """
    SELECT
        CAST(subject_id AS INTEGER) as subject_id,
        gender,
        CAST(dod AS TIMESTAMP) as dod,
        CAST(anchor_age AS INTEGER) as anchor_age,
        CAST(anchor_year AS INTEGER) as anchor_year,
        anchor_year_group
    FROM patients
    """
    pats = dataframe_from_query(conn, query)
    pats = pats[
        [
            "subject_id",
            "gender",
            "dod",
            "anchor_age",
            "anchor_year",
            "anchor_year_group",
        ]
    ]
    pats["yob"] = pats["anchor_year"] - pats["anchor_age"]
    pats.dod = pd.to_datetime(pats.dod)
    logger.info(f"[PATIENTS] Loaded patients data with shape: {pats.shape}")
    return pats


########################## DIAGNOSES ##########################
def read_diagnoses_icd_table(conn):
    """Read diagnoses_icd table from DuckDB with appropriate type casting"""
    logger.info("[DIAGNOSES] Reading diagnoses_icd table...")
    query = """
    SELECT
        CAST(subject_id AS INTEGER) as subject_id,
        CAST(hadm_id AS INTEGER) as hadm_id,
        CAST(seq_num AS INTEGER) as seq_num,
        icd_code,
        CAST(icd_version AS INTEGER) as icd_version
    FROM diagnoses_icd
    """
    diag = dataframe_from_query(conn, query)
    logger.info(f"[DIAGNOSES] Loaded diagnoses_icd data with shape: {diag.shape}")
    return diag


def read_d_icd_diagnoses_table(conn):
    """Read d_icd_diagnoses table from DuckDB"""
    logger.info("[DIAGNOSES] Reading d_icd_diagnoses table...")
    query = """
    SELECT
        icd_code,
        long_title
    FROM d_icd_diagnoses
    """
    d_icd = dataframe_from_query(conn, query)
    result = d_icd[["icd_code", "long_title"]]
    logger.info(f"[DIAGNOSES] Loaded d_icd_diagnoses data with shape: {result.shape}")
    return result


def read_diagnoses(conn):
    """Read and merge diagnoses tables"""
    logger.info("[DIAGNOSES] Reading and merging diagnoses tables...")
    diag_icd = read_diagnoses_icd_table(conn)
    d_icd = read_d_icd_diagnoses_table(conn)

    result = diag_icd.merge(
        d_icd,
        how="inner",
        left_on=["icd_code"],
        right_on=["icd_code"],
    )
    logger.info(f"[DIAGNOSES] Merged diagnoses data with shape: {result.shape}")
    return result


def standardize_icd(mapping, df, root=False):
    """Takes an ICD9 -> ICD10 mapping table and a diagnosis dataframe; adds column with converted ICD10 column"""
    logger.info(
        f"[ICD] Standardizing ICD codes (root={root}), input DataFrame shape: {df.shape}"
    )

    def icd_9to10(icd):
        # If root is true, only map an ICD 9 -> 10 according to the ICD9's root (first 3 digits)
        if root:
            icd = icd[:3]
        try:
            # Many ICD-9's do not have a 1-to-1 mapping; get first index of mapped codes
            return mapping.loc[mapping.diagnosis_code == icd].icd10cm.iloc[0]
        except:
            logger.error(f"Error on code {icd}")
            return np.nan

    # Create new column with original codes as default
    col_name = "icd10_convert"
    if root:
        col_name = "root_" + col_name
    df[col_name] = df["icd_code"].values

    # Group identical ICD9 codes, then convert all ICD9 codes within a group to ICD10
    icd9_codes = df.loc[df.icd_version == 9]["icd_code"].nunique()
    logger.info(f"[ICD] Converting {icd9_codes} unique ICD-9 codes to ICD-10...")

    for code, group in df.loc[df.icd_version == 9].groupby(by="icd_code"):
        new_code = icd_9to10(code)
        for idx in group.index.values:
            # Modify values of original df at the indexes in the groups
            df.at[idx, col_name] = new_code

    logger.info(f"[ICD] ICD standardization complete, DataFrame shape: {df.shape}")


########################## PROCEDURES ##########################
def read_procedures_icd_table(conn):
    """Read procedures_icd table from DuckDB with appropriate type casting"""
    logger.info("[PROCEDURES] Reading procedures_icd table...")
    query = """
    SELECT
        CAST(subject_id AS INTEGER) as subject_id,
        CAST(hadm_id AS INTEGER) as hadm_id,
        CAST(seq_num AS INTEGER) as seq_num,
        icd_code,
        CAST(icd_version AS INTEGER) as icd_version
    FROM procedures_icd
    """
    proc = dataframe_from_query(conn, query)
    logger.info(f"[PROCEDURES] Loaded procedures_icd data with shape: {proc.shape}")
    return proc


def read_d_icd_procedures_table(conn):
    """Read d_icd_procedures table from DuckDB"""
    logger.info("[PROCEDURES] Reading d_icd_procedures table...")
    query = """
    SELECT
        icd_code,
        long_title
    FROM d_icd_procedures
    """
    p_icd = dataframe_from_query(conn, query)
    result = p_icd[["icd_code", "long_title"]]
    logger.info(f"[PROCEDURES] Loaded d_icd_procedures data with shape: {result.shape}")
    return result


def read_procedures(conn):
    """Read and merge procedures tables"""
    logger.info("[PROCEDURES] Reading and merging procedures tables...")
    proc_icd = read_procedures_icd_table(conn)
    d_icd = read_d_icd_procedures_table(conn)

    result = proc_icd.merge(
        d_icd,
        how="inner",
        left_on=["icd_code"],
        right_on=["icd_code"],
    )
    logger.info(f"[PROCEDURES] Merged procedures data with shape: {result.shape}")
    return result


########################## MAPPING ##########################
def read_icd_mapping(map_path):
    """Read ICD mapping from CSV file (unchanged from original)"""
    logger.info(f"[MAPPING] Reading ICD mapping from: {map_path}")
    mapping = pd.read_csv(map_path, header=0, delimiter="\t")
    mapping.diagnosis_description = mapping.diagnosis_description.apply(str.lower)
    logger.info(f"[MAPPING] Loaded ICD mapping with shape: {mapping.shape}")
    return mapping


########################## PREPROCESSING ##########################


def preproc_meds(conn, adm_cohort_table_name: str) -> pd.DataFrame:
    """Preprocess medications data using DuckDB tables with medication-specific itemid filtering"""
    logger.info(
        f"[MEDS] Starting medication preprocessing with cohort table: {adm_cohort_table_name}"
    )

    # Get admissions cohort data
    adm_query = f"""
    SELECT
        CAST(hadm_id AS INTEGER) as hadm_id,
        CAST(stay_id AS INTEGER) as stay_id,
        CAST(intime AS TIMESTAMP) as intime
    FROM {adm_cohort_table_name}
    """
    adm = dataframe_from_query(conn, adm_query)
    adm["intime"] = pd.to_datetime(adm["intime"])
    logger.info(f"[MEDS] Admissions cohort loaded with shape: {adm.shape}")

    # Get medications data with medication-specific itemid filtering
    med_itemid_filter = get_itemid_filter_clause("med", "itemid")
    if med_itemid_filter:
        logger.info(
            f"[MEDS] Applying medication itemid filter: {med_itemid_filter[:100]}..."
        )
    else:
        logger.warning(
            "[MEDS] No medication itemid filter applied - including all medication itemids"
        )

    med_query = f"""
    SELECT
        CAST(subject_id AS INTEGER) as subject_id,
        CAST(stay_id AS INTEGER) as stay_id,
        CAST(itemid AS INTEGER) as itemid,
        CAST(starttime AS TIMESTAMP) as starttime,
        CAST(endtime AS TIMESTAMP) as endtime,
        CAST(rate AS DOUBLE) as rate,
        CAST(amount AS DOUBLE) as amount,
        CAST(orderid AS INTEGER) as orderid
    FROM inputevents
    WHERE 1=1 {med_itemid_filter}
    """
    med = dataframe_from_query(conn, med_query)
    med["starttime"] = pd.to_datetime(med["starttime"])
    med["endtime"] = pd.to_datetime(med["endtime"])
    logger.info(f"[MEDS] Raw medications data loaded with shape: {med.shape}")

    med = med.merge(adm, left_on="stay_id", right_on="stay_id", how="inner")
    logger.info(f"[MEDS] After merging with cohort, shape: {med.shape}")

    med["start_hours_from_admit"] = med["starttime"] - med["intime"]
    med["stop_hours_from_admit"] = med["endtime"] - med["intime"]

    med = med.dropna()
    logger.info(f"[MEDS] After dropping nulls, final shape: {med.shape}")
    logger.info(f"[MEDS] # of unique type of drug: {med.itemid.nunique()}")
    logger.info(f"[MEDS] # Admissions: {med.stay_id.nunique()}")
    logger.info(f"[MEDS] # Total rows: {med.shape[0]}")

    return med


def preproc_proc(
    conn, procedure_table_name: str, cohort_table_name: str, time_col: str
) -> pd.DataFrame:
    """Function for getting procedure observations pertaining to a cohort using DuckDB with procedure-specific itemid filtering"""
    logger.info(
        f"[PROC] Starting procedure preprocessing with table: {procedure_table_name}, cohort: {cohort_table_name}"
    )

    # Build the query to join procedures with cohort and procedure-specific itemid filtering
    proc_itemid_filter = get_itemid_filter_clause("proc", "p.itemid")
    if proc_itemid_filter:
        logger.info(
            f"[PROC] Applying procedure itemid filter: {proc_itemid_filter[:100]}..."
        )
    else:
        logger.warning(
            "[PROC] No procedure itemid filter applied - including all procedure itemids"
        )

    query = f"""
    SELECT
        p.subject_id,
        CAST(p.stay_id AS INTEGER) as stay_id,
        CAST(p.hadm_id AS INTEGER) as hadm_id,
        CAST(p.itemid AS INTEGER) as itemid,
        CAST(p.{time_col} AS TIMESTAMP) as {time_col},
        CAST(c.intime AS TIMESTAMP) as intime,
        CAST(c.outtime AS TIMESTAMP) as outtime
    FROM {procedure_table_name} p
    INNER JOIN {cohort_table_name} c ON p.stay_id = c.stay_id
    WHERE 1=1 {proc_itemid_filter}
    """

    df_cohort = dataframe_from_query(conn, query)
    df_cohort[time_col] = pd.to_datetime(df_cohort[time_col])
    df_cohort["intime"] = pd.to_datetime(df_cohort["intime"])
    df_cohort["outtime"] = pd.to_datetime(df_cohort["outtime"])
    df_cohort["event_time_from_admit"] = df_cohort[time_col] - df_cohort["intime"]
    logger.info(f"[PROC] After merging and time calculation, shape: {df_cohort.shape}")

    df_cohort = df_cohort.dropna()
    logger.info(f"[PROC] After dropping nulls, final shape: {df_cohort.shape}")

    logger.info(f"[PROC] # Unique Events: {df_cohort.itemid.dropna().nunique()}")
    logger.info(f"[PROC] # Admissions: {df_cohort.stay_id.nunique()}")
    logger.info(f"[PROC] Total rows: {df_cohort.shape[0]}")

    return df_cohort


def preproc_out(
    conn, output_table_name: str, cohort_table_name: str, time_col: str
) -> pd.DataFrame:
    """Function for getting output observations pertaining to a cohort using DuckDB with output-specific itemid filtering"""
    logger.info(
        f"[OUTPUT] Starting output preprocessing with table: {output_table_name}, cohort: {cohort_table_name}"
    )

    # Build the query to join outputs with cohort and output-specific itemid filtering
    output_itemid_filter = get_itemid_filter_clause("output", "o.itemid")
    if output_itemid_filter:
        logger.info(
            f"[OUTPUT] Applying output itemid filter: {output_itemid_filter[:100]}..."
        )
    else:
        logger.warning(
            "[OUTPUT] No output itemid filter applied - including all output itemids"
        )

    query = f"""
    SELECT
        CAST(o.stay_id AS INTEGER) as stay_id,
        CAST(o.itemid AS INTEGER) as itemid,
        CAST(o.{time_col} AS TIMESTAMP) as {time_col},
        CAST(o.value AS DOUBLE) as value,
        CAST(c.intime AS TIMESTAMP) as intime,
        CAST(c.outtime AS TIMESTAMP) as outtime
    FROM {output_table_name} o
    INNER JOIN {cohort_table_name} c ON o.stay_id = c.stay_id
    WHERE 1=1 {output_itemid_filter}
    """

    df_cohort = dataframe_from_query(conn, query)
    df_cohort[time_col] = pd.to_datetime(df_cohort[time_col])
    df_cohort["intime"] = pd.to_datetime(df_cohort["intime"])
    df_cohort["outtime"] = pd.to_datetime(df_cohort["outtime"])
    df_cohort["event_time_from_admit"] = df_cohort[time_col] - df_cohort["intime"]
    logger.info(
        f"[OUTPUT] After merging and time calculation, shape: {df_cohort.shape}"
    )

    df_cohort = df_cohort.dropna()
    logger.info(f"[OUTPUT] After dropping nulls, final shape: {df_cohort.shape}")

    logger.info(f"[OUTPUT] # Unique Events: {df_cohort.itemid.nunique()}")
    logger.info(f"[OUTPUT] # Admissions: {df_cohort.stay_id.nunique()}")
    logger.info(f"[OUTPUT] Total rows: {df_cohort.shape[0]}")

    return df_cohort


EVENT_TIME_ALLOWED_HOURS = 24  # Keep only events within 24 hours of admission


def preproc_chart(
    conn,
    chart_table_name: str,
    cohort_table_name: str,
    time_col: str,
    chunksize: int = 10000000,
) -> pd.DataFrame:
    """Function for getting chart observations pertaining to a cohort using DuckDB with chunked processing and chart-specific itemid filtering"""
    logger.info(
        f"[CHART] Starting chart preprocessing with table: {chart_table_name}, cohort: {cohort_table_name}"
    )

    # Get cohort data first
    cohort_query = f"""
    SELECT
        CAST(stay_id AS INTEGER) as stay_id,
        CAST(intime AS TIMESTAMP) as intime
    FROM {cohort_table_name}
    """
    cohort = dataframe_from_query(conn, cohort_query)
    cohort["intime"] = pd.to_datetime(cohort["intime"])
    logger.info(f"[CHART] Cohort data loaded with shape: {cohort.shape}")

    # Process chart events in chunks
    df_cohort = pd.DataFrame()

    # Get total count for progress tracking with chart-specific itemid filtering
    chart_itemid_filter = get_itemid_filter_clause("chart", "c.itemid")
    if chart_itemid_filter:
        logger.info(
            f"[CHART] Applying chart itemid filter: {chart_itemid_filter[:100]}..."
        )
    else:
        logger.warning(
            "[CHART] No chart itemid filter applied - including all chart itemids"
        )

    count_query = f"""
    SELECT COUNT(*) as total_rows
    FROM {chart_table_name} c
    INNER JOIN {cohort_table_name} coh ON c.stay_id = coh.stay_id
    WHERE c.valuenum IS NOT NULL AND c.valuenum != '' {chart_itemid_filter}
    """
    total_rows = dataframe_from_query(conn, count_query)["total_rows"].iloc[0]
    logger.info(f"[CHART] Total rows to process: {total_rows}")

    # Execute query directly without chunking since DuckDB's df() handles streaming efficiently
    query = f"""
    WITH chart_data AS (
        SELECT
            CAST(c.stay_id AS INTEGER) as stay_id,
            CAST(c.itemid AS INTEGER) as itemid,
            CAST(c.{time_col} AS TIMESTAMP) as {time_col},
            CAST(c.valuenum AS DOUBLE) as valuenum,
            c.valueuom,
            CAST(coh.intime AS TIMESTAMP) as intime
        FROM {chart_table_name} c
        INNER JOIN {cohort_table_name} coh ON c.stay_id = coh.stay_id
        WHERE c.valuenum IS NOT NULL AND c.valuenum != '' {chart_itemid_filter}
    ),
    processed_data AS (
        SELECT DISTINCT
            stay_id,
            itemid,
            valuenum,
            valueuom,
            ({time_col} - intime) as event_time_from_admit
        FROM chart_data
        WHERE valuenum IS NOT NULL
    )
    SELECT * FROM processed_data
    WHERE event_time_from_admit BETWEEN INTERVAL '0 hours' AND INTERVAL '{EVENT_TIME_ALLOWED_HOURS} hours'
    ORDER BY stay_id, itemid
    """

    logger.debug(f"[DEBUG] query=\n {query}")
    df_cohort = dataframe_from_query(conn, query)

    logger.info(f"[CHART] # Unique Events: {df_cohort.itemid.nunique()}")
    logger.info(f"[CHART] # Admissions: {df_cohort.stay_id.nunique()}")
    logger.info(f"[CHART] Total rows: {df_cohort.shape[0]}")

    return df_cohort


def preproc_icd_module(
    conn,
    module_table_name: str,
    adm_cohort_table_name: str,
    icd_map_path=None,
    map_code_colname=None,
    only_icd10=True,
) -> pd.DataFrame:
    """Takes a module dataset with ICD codes and puts it in long_format, optionally mapping ICD-codes by a mapping table path"""

    def get_module_cohort(conn, module_table_name: str, cohort_table_name: str):
        # Note: ICD modules don't have itemid, so no filtering applied here
        query = f"""
        SELECT
            m.subject_id,
            CAST(m.hadm_id AS INTEGER) as hadm_id,
            CAST(m.seq_num AS INTEGER) as seq_num,
            m.icd_code,
            CAST(m.icd_version AS INTEGER) as icd_version,
            CAST(c.stay_id AS INTEGER) as stay_id,
            c.label
        FROM {module_table_name} m
        INNER JOIN {cohort_table_name} c ON m.hadm_id = c.hadm_id
        """
        return dataframe_from_query(conn, query)

    def standardize_icd(mapping, df, root=False):
        """Takes an ICD9 -> ICD10 mapping table and a diagnosis dataframe; adds column with converted ICD10 column"""

        def icd_9to10(icd):
            # If root is true, only map an ICD 9 -> 10 according to the ICD9's root (first 3 digits)
            if root:
                icd = icd[:3]
            try:
                # Many ICD-9's do not have a 1-to-1 mapping; get first index of mapped codes
                return mapping.loc[mapping[map_code_colname] == icd].icd10cm.iloc[0]
            except:
                return np.nan

        # Create new column with original codes as default
        col_name = "icd10_convert"
        if root:
            col_name = "root_" + col_name
        df[col_name] = df["icd_code"].values

        # Group identical ICD9 codes, then convert all ICD9 codes within a group to ICD10
        for code, group in df.loc[df.icd_version == 9].groupby(by="icd_code"):
            new_code = icd_9to10(code)
            for idx in group.index.values:
                # Modify values of original df at the indexes in the groups
                df.at[idx, col_name] = new_code

        if only_icd10:
            # Column for just the roots of the converted ICD10 column
            df["root"] = df[col_name].apply(
                lambda x: x[:3] if type(x) is str else np.nan
            )

    module = get_module_cohort(conn, module_table_name, adm_cohort_table_name)

    # Optional ICD mapping if argument passed
    if icd_map_path:
        icd_map = read_icd_mapping(icd_map_path)
        standardize_icd(icd_map, module, root=True)
        logger.info(
            f"[ICD] # unique ICD-9 codes: {module[module['icd_version'] == 9]['icd_code'].nunique()}"
        )
        logger.info(
            f"[ICD] # unique ICD-10 codes: {module[module['icd_version'] == 10]['icd_code'].nunique()}"
        )
        logger.info(
            f"[ICD] # unique converted ICD-10 codes: {module['root_icd10_convert'].nunique()}"
        )
        logger.info(
            f"[ICD] # unique ICD-10 codes (After clinical grouping ICD-10 codes): {module['root'].nunique()}"
        )
        logger.info(f"[ICD] # Admissions: {module.stay_id.nunique()}")
        logger.info(f"[ICD] Total rows: {module.shape[0]}")
    return module


def pivot_cohort(
    df: pd.DataFrame,
    prefix: str,
    target_col: str,
    values="values",
    use_mlb=False,
    ohe=True,
    max_features=None,
):
    """Pivots long_format data into a multiindex array:
                                        || feature 1 || ... || feature n ||
    || subject_id || label || timedelta ||
    """
    aggfunc = np.mean
    pivot_df = df.dropna(subset=[target_col])

    if use_mlb:
        mlb = MultiLabelBinarizer()
        output = mlb.fit_transform(pivot_df[target_col].apply(ast.literal_eval))
        output = pd.DataFrame(output, columns=mlb.classes_)
        if max_features:
            top_features = (
                output.sum().sort_values(ascending=False).index[:max_features]
            )
            output = output[top_features]
        pivot_df = pd.concat(
            [
                pivot_df[["subject_id", "label", "timedelta"]].reset_index(drop=True),
                output,
            ],
            axis=1,
        )
        pivot_df = pd.pivot_table(
            pivot_df,
            index=["subject_id", "label", "timedelta"],
            values=pivot_df.columns[3:],
            aggfunc=np.max,
        )
    else:
        if max_features:
            top_features = pd.Series(
                pivot_df[["subject_id", target_col]]
                .drop_duplicates()[target_col]
                .value_counts()
                .index[:max_features],
                name=target_col,
            )
            pivot_df = pivot_df.merge(
                top_features, how="inner", left_on=target_col, right_on=target_col
            )
        if ohe:
            pivot_df = pd.concat(
                [
                    pivot_df.reset_index(drop=True),
                    pd.Series(np.ones(pivot_df.shape[0], dtype=int), name="values"),
                ],
                axis=1,
            )
            aggfunc = np.max
        pivot_df = pivot_df.pivot_table(
            index=["subject_id", "label", "timedelta"],
            columns=target_col,
            values=values,
            aggfunc=aggfunc,
        )

    pivot_df.columns = [prefix + str(i) for i in pivot_df.columns]
    return pivot_df
