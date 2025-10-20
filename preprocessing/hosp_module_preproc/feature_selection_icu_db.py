import os
import pickle
import glob
import importlib
import duckdb
import pandas as pd
import logging
import logging

# Import database-based preprocessing utilities
import utils.icu_preprocess_util_db
from utils.icu_preprocess_util_db import *

importlib.reload(utils.icu_preprocess_util_db)
import utils.icu_preprocess_util_db
from utils.icu_preprocess_util_db import *  # module of preprocessing functions

import utils.outlier_removal
from utils.outlier_removal import *

importlib.reload(utils.outlier_removal)
import utils.outlier_removal
from utils.outlier_removal import *

import utils.uom_conversion
from utils.uom_conversion import *

logger = logging.getLogger(__name__)

importlib.reload(utils.uom_conversion)
import utils.uom_conversion
from utils.uom_conversion import *


logger = logging.getLogger(__name__)

if not os.path.exists("./data/features"):
    os.makedirs("./data/features")
if not os.path.exists("./data/features/chartevents"):
    os.makedirs("./data/features/chartevents")


def feature_icu_db(
    conn,
    cohort_output,
    diag_flag=True,
    out_flag=True,
    chart_flag=True,
    proc_flag=True,
    med_flag=True,
):
    """
    Database-enabled ICU feature extraction using DuckDB and cohort CSV.gz files.

    Args:
        conn: DuckDB connection object (assumed to have MIMIC-IV tables loaded)
        cohort_output: Name of the cohort output file (without .csv.gz extension)
        diag_flag: Extract diagnosis data
        out_flag: Extract output events data
        chart_flag: Extract chart events data
        proc_flag: Extract procedures data
        med_flag: Extract medications data
    """

    try:
        # Path to cohort CSV.gz file
        cohort_csv_path = f"./data/cohort/{cohort_output}.csv.gz"

        if not os.path.exists(cohort_csv_path):
            raise FileNotFoundError(f"Cohort file not found: {cohort_csv_path}")

        logger.info(f"[LOADING COHORT FROM {cohort_csv_path}]")

        # Load cohort into temporary table
        cohort_table = load_cohort_from_csv_gz(conn, cohort_csv_path, "cohort_icu")

        if diag_flag:
            logger.info("[EXTRACTING DIAGNOSIS DATA]")
            diag = preproc_icd_module(
                conn,
                module_table_name="diagnoses_icd",
                adm_cohort_table_name=cohort_table,
                icd_map_path="./utils/mappings/ICD9_to_ICD10_mapping.txt",
                map_code_colname="diagnosis_code",
            )

            # Save diagnosis data
            diag_output = diag[
                [
                    "subject_id",
                    "hadm_id",
                    "stay_id",
                    "icd_code",
                    "root_icd10_convert",
                    "root",
                ]
            ]
            diag_output.to_csv(
                "./data/features/preproc_diag_icu.csv.gz",
                compression="gzip",
                index=False,
            )
            logger.info(f"[SUCCESSFULLY SAVED {len(diag_output):,} DIAGNOSIS RECORDS]")

        if out_flag:
            logger.info("[EXTRACTING OUTPUT EVENTS DATA]")
            out = preproc_out(
                conn,
                output_table_name="outputevents",
                cohort_table_name=cohort_table,
                time_col="charttime",
            )

            # Save output events data
            out_output = out[
                [
                    "stay_id",
                    "itemid",
                    "charttime",
                    "intime",
                    "event_time_from_admit",
                ]
            ]
            out_output.to_csv(
                "./data/features/preproc_out_icu.csv.gz",
                compression="gzip",
                index=False,
            )
            logger.info(
                f"[SUCCESSFULLY SAVED {len(out_output):,} OUTPUT EVENT RECORDS]"
            )

        if chart_flag:
            logger.info("[EXTRACTING CHART EVENTS DATA]")
            chart = preproc_chart(
                conn,
                chart_table_name="chartevents",
                cohort_table_name=cohort_table,
                time_col="charttime",
                chunksize=5000000,  # Process in chunks for large datasets
            )

            # Apply UOM cleaning
            logger.info("[APPLYING UOM CLEANING TO CHART EVENTS]")
            chart = drop_wrong_uom(chart, 0.95)

            # Save chart events data
            chart_output = chart[
                ["stay_id", "itemid", "event_time_from_admit", "valuenum"]
            ]
            chart_output.to_csv(
                "./data/features/preproc_chart_icu.csv.gz",
                compression="gzip",
                index=False,
            )
            logger.info(
                f"[SUCCESSFULLY SAVED {len(chart_output):,} CHART EVENT RECORDS]"
            )

        if proc_flag:
            logger.info("[EXTRACTING PROCEDURES DATA]")
            proc = preproc_proc(
                conn,
                procedure_table_name="procedureevents",
                cohort_table_name=cohort_table,
                time_col="starttime",
            )

            # Save procedures data
            proc_output = proc[
                [
                    "subject_id",
                    "hadm_id",
                    "stay_id",
                    "itemid",
                    "starttime",
                    "intime",
                    "event_time_from_admit",
                ]
            ]
            proc_output.to_csv(
                "./data/features/preproc_proc_icu.csv.gz",
                compression="gzip",
                index=False,
            )
            logger.info(f"[SUCCESSFULLY SAVED {len(proc_output):,} PROCEDURE RECORDS]")

        if med_flag:
            logger.info("[EXTRACTING MEDICATIONS DATA]")
            med = preproc_meds(conn, cohort_table)

            # Save medications data
            med_output = med[
                [
                    "subject_id",
                    "hadm_id",
                    "stay_id",
                    "itemid",
                    "starttime",
                    "endtime",
                    "start_hours_from_admit",
                    "stop_hours_from_admit",
                    "rate",
                    "amount",
                    "orderid",
                ]
            ]
            med_output.to_csv(
                "./data/features/preproc_med_icu.csv.gz",
                compression="gzip",
                index=False,
            )
            logger.info(f"[SUCCESSFULLY SAVED {len(med_output):,} MEDICATION RECORDS]")

    except Exception as e:
        logger.error(f"[ERROR DURING FEATURE EXTRACTION]: {e}")
        raise
    finally:
        # Clean up
        cleanup_cohort_table(conn, "cohort_icu")
        logger.info("[COHORT TABLE CLEANED UP]")


def preprocess_features_icu(
    cohort_output,
    diag_flag,
    group_diag,
    chart_flag,
    clean_chart,
    impute_outlier_chart,
    thresh,
    left_thresh,
):
    """
    Post-process extracted features (same as original but with enhanced logging).
    """
    if diag_flag:
        logger.info("[PROCESSING DIAGNOSIS DATA]")
        diag = pd.read_csv(
            "./data/features/preproc_diag_icu.csv.gz", compression="gzip", header=0
        )
        original_count = len(diag)

        if group_diag == "Keep both ICD-9 and ICD-10 codes":
            diag["new_icd_code"] = diag["icd_code"]
        elif group_diag == "Convert ICD-9 to ICD-10 codes":
            diag["new_icd_code"] = diag["root_icd10_convert"]
        elif group_diag == "Convert ICD-9 to ICD-10 and group ICD-10 codes":
            diag["new_icd_code"] = diag["root"]

        diag = diag[["subject_id", "hadm_id", "stay_id", "new_icd_code"]].dropna()

        dropped_count = original_count - len(diag)
        if dropped_count > 0:
            logger.warning(
                f"Dropped {dropped_count:,} diagnosis records due to missing new_icd_code"
            )

        logger.info(f"Total diagnosis records after processing: {diag.shape[0]:,}")

        diag.to_csv(
            "./data/features/preproc_diag_icu.csv.gz", compression="gzip", index=False
        )
        logger.info("[SUCCESSFULLY SAVED PROCESSED DIAGNOSIS DATA]")

    if chart_flag and clean_chart:
        logger.info("[PROCESSING CHART EVENTS DATA]")
        chart = pd.read_csv(
            "./data/features/preproc_chart_icu.csv.gz", compression="gzip", header=0
        )
        original_count = len(chart)
        logger.info(f"Chart events before outlier processing: {original_count:,}")

        chart = outlier_imputation(
            chart, "itemid", "valuenum", thresh, left_thresh, impute_outlier_chart
        )

        final_count = len(chart)
        dropped_count = original_count - final_count
        if dropped_count > 0:
            logger.info(
                f"Outlier processing removed {dropped_count:,} chart event records"
            )

        logger.info(f"Chart events after outlier processing: {final_count:,}")

        chart.to_csv(
            "./data/features/preproc_chart_icu.csv.gz",
            compression="gzip",
            index=False,
        )
        logger.info("[SUCCESSFULLY SAVED PROCESSED CHART EVENTS DATA]")


def generate_summary_icu(diag_flag, proc_flag, med_flag, out_flag, chart_flag):
    """
    Generate feature summaries (same as original but with enhanced logging).
    """
    logger.info("[GENERATING FEATURE SUMMARY]")

    if diag_flag:
        logger.info("  - Processing diagnosis summary...")
        diag = pd.read_csv(
            "./data/features/preproc_diag_icu.csv.gz", compression="gzip", header=0
        )
        freq = (
            diag.groupby(["stay_id", "new_icd_code"])
            .size()
            .reset_index(name="mean_frequency")
        )
        freq = freq.groupby(["new_icd_code"])["mean_frequency"].mean().reset_index()
        total = diag.groupby("new_icd_code").size().reset_index(name="total_count")
        summary = pd.merge(freq, total, on="new_icd_code", how="right")
        summary = summary.fillna(0)
        summary.to_csv("./data/summary/diag_summary.csv", index=False)
        summary["new_icd_code"].to_csv("./data/summary/diag_features.csv", index=False)
        logger.info(f"    {len(summary)} unique diagnosis codes")

    if med_flag:
        logger.info("  - Processing medication summary...")
        med = pd.read_csv(
            "./data/features/preproc_med_icu.csv.gz", compression="gzip", header=0
        )
        freq = (
            med.groupby(["stay_id", "itemid"]).size().reset_index(name="mean_frequency")
        )
        freq = freq.groupby(["itemid"])["mean_frequency"].mean().reset_index()

        missing = (
            med[med["amount"] == 0]
            .groupby("itemid")
            .size()
            .reset_index(name="missing_count")
        )
        total = med.groupby("itemid").size().reset_index(name="total_count")
        summary = pd.merge(missing, total, on="itemid", how="right")
        summary = pd.merge(freq, summary, on="itemid", how="right")
        summary = summary.fillna(0)
        summary.to_csv("./data/summary/med_summary.csv", index=False)
        summary["itemid"].to_csv("./data/summary/med_features.csv", index=False)
        logger.info(f"    {len(summary)} unique medication items")

    if proc_flag:
        logger.info("  - Processing procedure summary...")
        proc = pd.read_csv(
            "./data/features/preproc_proc_icu.csv.gz", compression="gzip", header=0
        )
        freq = (
            proc.groupby(["stay_id", "itemid"])
            .size()
            .reset_index(name="mean_frequency")
        )
        freq = freq.groupby(["itemid"])["mean_frequency"].mean().reset_index()
        total = proc.groupby("itemid").size().reset_index(name="total_count")
        summary = pd.merge(freq, total, on="itemid", how="right")
        summary = summary.fillna(0)
        summary.to_csv("./data/summary/proc_summary.csv", index=False)
        summary["itemid"].to_csv("./data/summary/proc_features.csv", index=False)
        logger.info(f"    {len(summary)} unique procedure items")

    if out_flag:
        logger.info("  - Processing output events summary...")
        out = pd.read_csv(
            "./data/features/preproc_out_icu.csv.gz", compression="gzip", header=0
        )
        freq = (
            out.groupby(["stay_id", "itemid"]).size().reset_index(name="mean_frequency")
        )
        freq = freq.groupby(["itemid"])["mean_frequency"].mean().reset_index()
        total = out.groupby("itemid").size().reset_index(name="total_count")
        summary = pd.merge(freq, total, on="itemid", how="right")
        summary = summary.fillna(0)
        summary.to_csv("./data/summary/out_summary.csv", index=False)
        summary["itemid"].to_csv("./data/summary/out_features.csv", index=False)
        logger.info(f"    {len(summary)} unique output items")

    if chart_flag:
        logger.info("  - Processing chart events summary...")
        chart = pd.read_csv(
            "./data/features/preproc_chart_icu.csv.gz", compression="gzip", header=0
        )
        freq = (
            chart.groupby(["stay_id", "itemid"])
            .size()
            .reset_index(name="mean_frequency")
        )
        freq = freq.groupby(["itemid"])["mean_frequency"].mean().reset_index()

        missing = (
            chart[chart["valuenum"] == 0]
            .groupby("itemid")
            .size()
            .reset_index(name="missing_count")
        )
        total = chart.groupby("itemid").size().reset_index(name="total_count")
        summary = pd.merge(missing, total, on="itemid", how="right")
        summary = pd.merge(freq, summary, on="itemid", how="right")
        summary = summary.fillna(0)
        summary.to_csv("./data/summary/chart_summary.csv", index=False)
        summary["itemid"].to_csv("./data/summary/chart_features.csv", index=False)
        logger.info(f"    {len(summary)} unique chart items")

    logger.info("[SUCCESSFULLY SAVED FEATURE SUMMARIES]")


def features_selection_icu(
    cohort_output,
    diag_flag,
    proc_flag,
    med_flag,
    out_flag,
    chart_flag,
    group_diag,
    group_med,
    group_proc,
    group_out,
    group_chart,
):
    """
    Feature selection based on generated summaries (same as original but with enhanced logging).
    """
    logger.info("[PERFORMING FEATURE SELECTION]")

    if diag_flag and group_diag:
        logger.info("  - Filtering diagnosis features...")
        diag = pd.read_csv(
            "./data/features/preproc_diag_icu.csv.gz", compression="gzip", header=0
        )
        original_count = len(diag)
        features = pd.read_csv("./data/summary/diag_features.csv", header=0)
        diag = diag[diag["new_icd_code"].isin(features["new_icd_code"].unique())]

        filtered_count = len(diag)
        dropped_count = original_count - filtered_count
        if dropped_count > 0:
            logger.info(
                f"Feature selection dropped {dropped_count:,} diagnosis records"
            )

        logger.info(f"Filtered from {original_count:,} to {filtered_count:,} rows")

        diag.to_csv(
            "./data/features/preproc_diag_icu.csv.gz",
            compression="gzip",
            index=False,
        )
        logger.info("[SUCCESSFULLY SAVED FILTERED DIAGNOSIS DATA]")

    if med_flag and group_med:
        logger.info("  - Filtering medication features...")
        med = pd.read_csv(
            "./data/features/preproc_med_icu.csv.gz", compression="gzip", header=0
        )
        features = pd.read_csv("./data/summary/med_features.csv", header=0)
        original_count = len(med)
        med = med[med["itemid"].isin(features["itemid"].unique())]
        filtered_count = len(diag)
        dropped_count = original_count - filtered_count
        if dropped_count > 0:
            logger.info(
                f"Feature selection dropped {dropped_count:,} diagnosis records"
            )
        logger.info(f"    Filtered from {original_count:,} to {filtered_count:,} rows")

        med.to_csv(
            "./data/features/preproc_med_icu.csv.gz",
            compression="gzip",
            index=False,
        )

    if proc_flag and group_proc:
        logger.info("  - Filtering procedure features...")
        proc = pd.read_csv(
            "./data/features/preproc_proc_icu.csv.gz", compression="gzip", header=0
        )
        features = pd.read_csv("./data/summary/proc_features.csv", header=0)
        original_count = len(proc)
        proc = proc[proc["itemid"].isin(features["itemid"].unique())]
        filtered_count = len(med)
        dropped_count = original_count - filtered_count
        if dropped_count > 0:
            logger.info(
                f"Feature selection dropped {dropped_count:,} medication records"
            )
        logger.info(f"    Filtered from {original_count:,} to {filtered_count:,} rows")

        proc.to_csv(
            "./data/features/preproc_proc_icu.csv.gz",
            compression="gzip",
            index=False,
        )

    if out_flag and group_out:
        logger.info("  - Filtering output events features...")
        out = pd.read_csv(
            "./data/features/preproc_out_icu.csv.gz", compression="gzip", header=0
        )
        features = pd.read_csv("./data/summary/out_features.csv", header=0)
        original_count = len(out)
        out = out[out["itemid"].isin(features["itemid"].unique())]
        filtered_count = len(proc)
        dropped_count = original_count - filtered_count
        if dropped_count > 0:
            logger.info(
                f"Feature selection dropped {dropped_count:,} procedure records"
            )
        logger.info(f"    Filtered from {original_count:,} to {filtered_count:,} rows")

        out.to_csv(
            "./data/features/preproc_out_icu.csv.gz",
            compression="gzip",
            index=False,
        )

    if chart_flag and group_chart:
        logger.info("  - Filtering chart events features...")
        chart = pd.read_csv(
            "./data/features/preproc_chart_icu.csv.gz",
            compression="gzip",
            header=0,
            index_col=None,
        )
        features = pd.read_csv("./data/summary/chart_features.csv", header=0)
        original_count = len(chart)
        chart = chart[chart["itemid"].isin(features["itemid"].unique())]
        filtered_count = len(out)
        dropped_count = original_count - filtered_count
        if dropped_count > 0:
            logger.info(
                f"Feature selection dropped {dropped_count:,} output event records"
            )
        logger.info(f"    Filtered from {original_count:,} to {filtered_count:,} rows")

        chart.to_csv(
            "./data/features/preproc_chart_icu.csv.gz",
            compression="gzip",
            index=False,
        )

    logger.info("[FEATURE SELECTION COMPLETE]")


# Main function that mirrors the original feature_icu interface
def feature_icu(
    conn,
    cohort_output,
    diag_flag=True,
    out_flag=True,
    chart_flag=True,
    proc_flag=True,
    med_flag=True,
):
    """
    Main interface function that maintains compatibility with original feature_selection_icu.
    This is a wrapper around feature_icu_db for backward compatibility.

    Args:
        conn: DuckDB connection object (assumed to have MIMIC-IV tables loaded)
        cohort_output: Name of the cohort output file (without .csv.gz extension)
        diag_flag: Extract diagnosis data
        out_flag: Extract output events data
        chart_flag: Extract chart events data
        proc_flag: Extract procedures data
        med_flag: Extract medications data
    """
    return feature_icu_db(
        conn=conn,
        cohort_output=cohort_output,
        diag_flag=diag_flag,
        out_flag=out_flag,
        chart_flag=chart_flag,
        proc_flag=proc_flag,
        med_flag=med_flag,
    )
