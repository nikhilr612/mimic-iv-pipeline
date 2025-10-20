import numpy as np
import pandas as pd
import sys, os
import re
import ast
import datetime as dt
from tqdm import tqdm

from sklearn.preprocessing import MultiLabelBinarizer


########################## ITEMID FILTERING ##########################
# Global variables to store whitelisted itemids for different data types
_CHART_ITEMIDS = None
_MED_ITEMIDS = None
_PROC_ITEMIDS = None
_OUTPUT_ITEMIDS = None


def load_itemid_whitelist(data_type="chart", filepath=None):
    """Load itemid whitelist from file for specific data type.

    Args:
        data_type: "chart", "med", "proc", or "output"
        filepath: Custom file path, or None to use defaults
    """
    global _CHART_ITEMIDS, _MED_ITEMIDS, _PROC_ITEMIDS, _OUTPUT_ITEMIDS

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
                    print(f"Using generic itemid whitelist for {data_type}: {filepath}")
                    break

    if filepath is None or not os.path.exists(filepath):
        print(
            f"Warning: No itemid whitelist file found for {data_type}. Including all itemids."
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
                        print(
                            f"Warning: Invalid itemid '{line}' in {data_type} whitelist"
                        )

        itemid_set = set(itemids) if itemids else None

        # Store in appropriate global variable
        if data_type == "chart":
            _CHART_ITEMIDS = itemid_set
        elif data_type == "med":
            _MED_ITEMIDS = itemid_set
        elif data_type == "proc":
            _PROC_ITEMIDS = itemid_set
        elif data_type == "output":
            _OUTPUT_ITEMIDS = itemid_set

        print(f"Loaded {len(itemids)} {data_type} itemids from: {filepath}")
        return itemid_set
    except Exception as e:
        print(f"Error loading {data_type} itemid whitelist from {filepath}: {e}")
        return None


def get_itemid_whitelist(data_type="chart"):
    """Get the current itemid whitelist for specified data type."""
    global _CHART_ITEMIDS, _MED_ITEMIDS, _PROC_ITEMIDS, _OUTPUT_ITEMIDS

    whitelist_map = {
        "chart": _CHART_ITEMIDS,
        "med": _MED_ITEMIDS,
        "proc": _PROC_ITEMIDS,
        "output": _OUTPUT_ITEMIDS,
    }

    current_whitelist = whitelist_map.get(data_type)
    if current_whitelist is None:
        # Try to load it
        load_itemid_whitelist(data_type)
        current_whitelist = whitelist_map.get(data_type)

    return current_whitelist


def get_itemid_filter_clause(data_type="chart", itemid_col="itemid"):
    """Get SQL WHERE clause for itemid filtering. Returns empty string if no filtering."""
    whitelist = get_itemid_whitelist(data_type)
    if whitelist is None or len(whitelist) == 0:
        return ""

    itemid_list = ",".join(str(id) for id in sorted(whitelist))
    return f"AND CAST({itemid_col} AS INTEGER) IN ({itemid_list})"


def filter_by_itemid(df, data_type="chart", itemid_col="itemid"):
    """Filter dataframe by whitelisted itemids if whitelist is available."""
    whitelist = get_itemid_whitelist(data_type)
    if whitelist is None:
        return df

    if itemid_col not in df.columns:
        print(
            f"Warning: Column '{itemid_col}' not found. Skipping {data_type} itemid filtering."
        )
        return df

    original_count = len(df)
    df_filtered = df[df[itemid_col].isin(whitelist)]
    filtered_count = len(df_filtered)

    print(
        f"{data_type.title()} itemid filtering: {original_count} -> {filtered_count} rows ({original_count - filtered_count} removed)"
    )
    return df_filtered


def load_all_itemid_whitelists():
    """Load all itemid whitelists for all data types."""
    data_types = ["chart", "med", "proc", "output"]
    loaded = {}

    for data_type in data_types:
        whitelist = load_itemid_whitelist(data_type)
        loaded[data_type] = len(whitelist) if whitelist else 0

    print(f"Itemid whitelists loaded: {loaded}")
    return loaded


########################## GENERAL ##########################
def dataframe_from_query(conn, query):
    """Execute a DuckDB query and return a pandas DataFrame"""
    return conn.execute(query).df()


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
        print(
            f"Loading cohort data from {csv_gz_path} into temporary table '{table_name}'..."
        )

        # Drop the table if it already exists
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")

        # Load CSV.gz using pandas first for better compatibility
        print("Reading CSV.gz file with pandas...")
        df = pd.read_csv(csv_gz_path, compression="gzip")

        if df.empty:
            print("Warning: CSV file is empty")
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
            print(f"Warning: No data loaded from {csv_gz_path}")
        else:
            print(f"Successfully loaded {row_count} rows into table '{table_name}'")

        # Show table schema for debugging
        schema_result = conn.execute(f"DESCRIBE {table_name}").df()
        print(f"Table schema:")
        print(schema_result.to_string(index=False))

        # Validate required columns for cohort data
        required_columns = ["stay_id", "intime"]
        columns = schema_result["column_name"].tolist()
        missing_columns = [col for col in required_columns if col not in columns]

        if missing_columns:
            print(f"Warning: Missing recommended cohort columns: {missing_columns}")
            print(
                "Expected columns for ICU cohort: stay_id, subject_id, hadm_id, intime, outtime"
            )

        return table_name

    except Exception as e:
        print(f"Error loading cohort data from {csv_gz_path}: {str(e)}")
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
        print(
            f"Loading cohort data from DataFrame into temporary table '{table_name}'..."
        )

        # Drop the table if it already exists
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")

        # Check if DataFrame is empty
        if df.empty:
            print("Warning: DataFrame is empty, creating table with no rows")
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
        print(f"Successfully loaded {row_count} rows into table '{table_name}'")

        # Show table schema for debugging
        schema_result = conn.execute(f"DESCRIBE {table_name}").df()
        print(f"Table schema:")
        print(schema_result.to_string(index=False))

        return table_name

    except Exception as e:
        print(f"Error loading cohort data from DataFrame: {str(e)}")
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
    # Load cohort data from CSV.gz into temporary table
    cohort_table = load_cohort_from_csv_gz(conn, csv_gz_path, cohort_table_name)

    # Preprocess data based on type
    if data_type == "chart":
        return preproc_chart(conn, chart_table_name, cohort_table, time_col, chunksize)
    elif data_type == "med":
        return preproc_meds(conn, cohort_table)
    elif data_type == "proc":
        return preproc_proc(conn, procedure_table_name, cohort_table, time_col)
    elif data_type == "output":
        return preproc_out(conn, output_table_name, cohort_table, time_col)
    else:
        raise ValueError(
            f"Unsupported data_type: {data_type}. Supported types: 'chart', 'med', 'proc', 'output'"
        )


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
        print(f"Successfully cleaned up temporary table '{table_name}'")
    except Exception as e:
        print(f"Warning: Could not clean up table '{table_name}': {str(e)}")


def read_admissions_table(conn):
    """Read admissions table from DuckDB with appropriate type casting"""
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
    return admits


def read_patients_table(conn):
    """Read patients table from DuckDB with appropriate type casting"""
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
    return pats


########################## DIAGNOSES ##########################
def read_diagnoses_icd_table(conn):
    """Read diagnoses_icd table from DuckDB with appropriate type casting"""
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
    return diag


def read_d_icd_diagnoses_table(conn):
    """Read d_icd_diagnoses table from DuckDB"""
    query = """
    SELECT
        icd_code,
        long_title
    FROM d_icd_diagnoses
    """
    d_icd = dataframe_from_query(conn, query)
    return d_icd[["icd_code", "long_title"]]


def read_diagnoses(conn):
    """Read and merge diagnoses tables"""
    return read_diagnoses_icd_table(conn).merge(
        read_d_icd_diagnoses_table(conn),
        how="inner",
        left_on=["icd_code"],
        right_on=["icd_code"],
    )


def standardize_icd(mapping, df, root=False):
    """Takes an ICD9 -> ICD10 mapping table and a diagnosis dataframe; adds column with converted ICD10 column"""

    def icd_9to10(icd):
        # If root is true, only map an ICD 9 -> 10 according to the ICD9's root (first 3 digits)
        if root:
            icd = icd[:3]
        try:
            # Many ICD-9's do not have a 1-to-1 mapping; get first index of mapped codes
            return mapping.loc[mapping.diagnosis_code == icd].icd10cm.iloc[0]
        except:
            print("Error on code", icd)
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


########################## PROCEDURES ##########################
def read_procedures_icd_table(conn):
    """Read procedures_icd table from DuckDB with appropriate type casting"""
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
    return proc


def read_d_icd_procedures_table(conn):
    """Read d_icd_procedures table from DuckDB"""
    query = """
    SELECT
        icd_code,
        long_title
    FROM d_icd_procedures
    """
    p_icd = dataframe_from_query(conn, query)
    return p_icd[["icd_code", "long_title"]]


def read_procedures(conn):
    """Read and merge procedures tables"""
    return read_procedures_icd_table(conn).merge(
        read_d_icd_procedures_table(conn),
        how="inner",
        left_on=["icd_code"],
        right_on=["icd_code"],
    )


########################## MAPPING ##########################
def read_icd_mapping(map_path):
    """Read ICD mapping from CSV file (unchanged from original)"""
    mapping = pd.read_csv(map_path, header=0, delimiter="\t")
    mapping.diagnosis_description = mapping.diagnosis_description.apply(str.lower)
    return mapping


########################## PREPROCESSING ##########################


def preproc_meds(conn, adm_cohort_table_name: str) -> pd.DataFrame:
    """Preprocess medications data using DuckDB tables with medication-specific itemid filtering"""

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

    # Get medications data with medication-specific itemid filtering
    med_itemid_filter = get_itemid_filter_clause("med", "itemid")
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

    med = med.merge(adm, left_on="stay_id", right_on="stay_id", how="inner")
    med["start_hours_from_admit"] = med["starttime"] - med["intime"]
    med["stop_hours_from_admit"] = med["endtime"] - med["intime"]

    med = med.dropna()
    print("# of unique type of drug: ", med.itemid.nunique())
    print("# Admissions:  ", med.stay_id.nunique())
    print("# Total rows", med.shape[0])

    return med


def preproc_proc(
    conn, procedure_table_name: str, cohort_table_name: str, time_col: str
) -> pd.DataFrame:
    """Function for getting procedure observations pertaining to a cohort using DuckDB with procedure-specific itemid filtering"""

    # Build the query to join procedures with cohort and procedure-specific itemid filtering
    proc_itemid_filter = get_itemid_filter_clause("proc", "p.itemid")
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

    df_cohort = df_cohort.dropna()

    print("# Unique Events:  ", df_cohort.itemid.dropna().nunique())
    print("# Admissions:  ", df_cohort.stay_id.nunique())
    print("Total rows", df_cohort.shape[0])

    return df_cohort


def preproc_out(
    conn, output_table_name: str, cohort_table_name: str, time_col: str
) -> pd.DataFrame:
    """Function for getting output observations pertaining to a cohort using DuckDB with output-specific itemid filtering"""

    # Build the query to join outputs with cohort and output-specific itemid filtering
    output_itemid_filter = get_itemid_filter_clause("output", "o.itemid")
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

    df_cohort = df_cohort.dropna()

    print("# Unique Events:  ", df_cohort.itemid.nunique())
    print("# Admissions:  ", df_cohort.stay_id.nunique())
    print("Total rows", df_cohort.shape[0])

    return df_cohort


def preproc_chart(
    conn,
    chart_table_name: str,
    cohort_table_name: str,
    time_col: str,
    chunksize: int = 10000000,
) -> pd.DataFrame:
    """Function for getting chart observations pertaining to a cohort using DuckDB with chunked processing and chart-specific itemid filtering"""

    # Get cohort data first
    cohort_query = f"""
    SELECT
        CAST(stay_id AS INTEGER) as stay_id,
        CAST(intime AS TIMESTAMP) as intime
    FROM {cohort_table_name}
    """
    cohort = dataframe_from_query(conn, cohort_query)
    cohort["intime"] = pd.to_datetime(cohort["intime"])

    # Process chart events in chunks
    df_cohort = pd.DataFrame()

    # Get total count for progress tracking with chart-specific itemid filtering
    chart_itemid_filter = get_itemid_filter_clause("chart", "c.itemid")
    count_query = f"""
    SELECT COUNT(*) as total_rows
    FROM {chart_table_name} c
    INNER JOIN {cohort_table_name} coh ON c.stay_id = coh.stay_id
    WHERE c.valuenum IS NOT NULL AND c.valuenum != '' {chart_itemid_filter}
    """
    total_rows = dataframe_from_query(conn, count_query)["total_rows"].iloc[0]

    # Execute query directly without chunking since DuckDB's df() handles streaming efficiently
    query = f"""
    WITH chart_data AS (
        SELECT
            CAST(c.stay_id AS INTEGER) as stay_id,
            CAST(c.itemid AS INTEGER) as itemid,
            CAST(c.{time_col} AS TIMESTAMP) as {time_col},
            CAST(c.valuenum AS DOUBLE) as valuenum,
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
            ({time_col} - intime) as event_time_from_admit
        FROM chart_data
        WHERE valuenum IS NOT NULL
    )
    SELECT * FROM processed_data
    ORDER BY stay_id, itemid
    """

    df_cohort = dataframe_from_query(conn, query)

    print("# Unique Events:  ", df_cohort.itemid.nunique())
    print("# Admissions:  ", df_cohort.stay_id.nunique())
    print("Total rows", df_cohort.shape[0])

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
        print(
            "# unique ICD-9 codes",
            module[module["icd_version"] == 9]["icd_code"].nunique(),
        )
        print(
            "# unique ICD-10 codes",
            module[module["icd_version"] == 10]["icd_code"].nunique(),
        )
        print(
            "# unique ICD-10 codes (After converting ICD-9 to ICD-10)",
            module["root_icd10_convert"].nunique(),
        )
        print(
            "# unique ICD-10 codes (After clinical grouping ICD-10 codes)",
            module["root"].nunique(),
        )
        print("# Admissions:  ", module.stay_id.nunique())
        print("Total rows", module.shape[0])
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
