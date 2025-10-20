import os
import pickle
import glob
import importlib
import logging

# print(os.getcwd())
# os.chdir('../../')
# print(os.getcwd())
import utils.icu_preprocess_util
from utils.icu_preprocess_util import *

importlib.reload(utils.icu_preprocess_util)
import utils.icu_preprocess_util
from utils.icu_preprocess_util import *  # module of preprocessing functions

import utils.outlier_removal
from utils.outlier_removal import *

importlib.reload(utils.outlier_removal)
import utils.outlier_removal
from utils.outlier_removal import *

import utils.uom_conversion
from utils.uom_conversion import *

importlib.reload(utils.uom_conversion)
import utils.uom_conversion
from utils.uom_conversion import *

logger = logging.getLogger(__name__)


if not os.path.exists("./data/features"):
    os.makedirs("./data/features")
if not os.path.exists("./data/features/chartevents"):
    os.makedirs("./data/features/chartevents")


def feature_icu(
    cohort_output,
    version_path,
    diag_flag=True,
    out_flag=True,
    chart_flag=True,
    proc_flag=True,
    med_flag=True,
):
    if diag_flag:
        logger.info("[EXTRACTING DIAGNOSIS DATA]")
        diag = preproc_icd_module(
            "./" + version_path + "/hosp/diagnoses_icd.csv.gz",
            "./data/cohort/" + cohort_output + ".csv.gz",
            "./utils/mappings/ICD9_to_ICD10_mapping.txt",
            map_code_colname="diagnosis_code",
        )
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
            "./data/features/preproc_diag_icu.csv.gz", compression="gzip", index=False
        )
        logger.info(f"[SUCCESSFULLY SAVED {len(diag_output):,} DIAGNOSIS RECORDS]")

    if out_flag:
        logger.info("[EXTRACTING OUTPUT EVENTS DATA]")
        out = preproc_out(
            "./" + version_path + "/icu/outputevents.csv.gz",
            "./data/cohort/" + cohort_output + ".csv.gz",
            "charttime",
            dtypes=None,
            usecols=None,
        )
        out_output = out[
            [
                "subject_id",
                "hadm_id",
                "stay_id",
                "itemid",
                "charttime",
                "intime",
                "event_time_from_admit",
            ]
        ]
        out_output.to_csv(
            "./data/features/preproc_out_icu.csv.gz", compression="gzip", index=False
        )
        logger.info(f"[SUCCESSFULLY SAVED {len(out_output):,} OUTPUT EVENT RECORDS]")

    if chart_flag:
        logger.info("[EXTRACTING CHART EVENTS DATA]")
        chart = preproc_chart(
            "./" + version_path + "/icu/chartevents.csv.gz",
            "./data/cohort/" + cohort_output + ".csv.gz",
            "charttime",
            dtypes=None,
            usecols=["stay_id", "charttime", "itemid", "valuenum", "valueuom"],
        )
        logger.info("[APPLYING UOM CLEANING TO CHART EVENTS]")
        chart = drop_wrong_uom(chart, 0.95)
        chart_output = chart[["stay_id", "itemid", "event_time_from_admit", "valuenum"]]
        chart_output.to_csv(
            "./data/features/preproc_chart_icu.csv.gz", compression="gzip", index=False
        )
        logger.info(f"[SUCCESSFULLY SAVED {len(chart_output):,} CHART EVENT RECORDS]")

    if proc_flag:
        logger.info("[EXTRACTING PROCEDURES DATA]")
        proc = preproc_proc(
            "./" + version_path + "/icu/procedureevents.csv.gz",
            "./data/cohort/" + cohort_output + ".csv.gz",
            "starttime",
            dtypes=None,
            usecols=["stay_id", "starttime", "itemid"],
        )
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
            "./data/features/preproc_proc_icu.csv.gz", compression="gzip", index=False
        )
        logger.info(f"[SUCCESSFULLY SAVED {len(proc_output):,} PROCEDURE RECORDS]")

    if med_flag:
        logger.info("[EXTRACTING MEDICATIONS DATA]")
        med = preproc_meds(
            "./" + version_path + "/icu/inputevents.csv.gz",
            "./data/cohort/" + cohort_output + ".csv.gz",
        )
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
            "./data/features/preproc_med_icu.csv.gz", compression="gzip", index=False
        )
        logger.info(f"[SUCCESSFULLY SAVED {len(med_output):,} MEDICATION RECORDS]")


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
    if diag_flag:
        logger.info("[PROCESSING DIAGNOSIS DATA]")
        diag = pd.read_csv(
            "./data/features/preproc_diag_icu.csv.gz", compression="gzip", header=0
        )
        original_count = len(diag)

        if group_diag == "Keep both ICD-9 and ICD-10 codes":
            diag["new_icd_code"] = diag["icd_code"]
        if group_diag == "Convert ICD-9 to ICD-10 codes":
            diag["new_icd_code"] = diag["root_icd10_convert"]
        if group_diag == "Convert ICD-9 to ICD-10 and group ICD-10 codes":
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

    if chart_flag:
        if clean_chart:
            logger.info("[PROCESSING CHART EVENTS DATA]")
            chart = pd.read_csv(
                "./data/features/preproc_chart_icu.csv.gz", compression="gzip", header=0
            )
            original_count = len(chart)
            logger.info(f"Chart events before outlier processing: {original_count:,}")

            chart = outlier_imputation(
                chart, "itemid", "valuenum", thresh, left_thresh, impute_outlier_chart
            )

            #             for i in [227441, 229357, 229358, 229360]:
            #                 try:
            #                     maj = chart.loc[chart.itemid == i].valueuom.value_counts().index[0]
            #                     chart = chart.loc[~((chart.itemid == i) & (chart.valueuom == maj))]
            #                 except IndexError:
            #                     logger.warning(f"{idx} not found")

            final_count = len(chart)
            dropped_count = original_count - final_count
            if dropped_count > 0:
                logger.info(
                    f"Outlier processing removed {dropped_count:,} chart event records"
                )

            logger.info(
                f"Total chart event records after processing: {chart.shape[0]:,}"
            )
            chart.to_csv(
                "./data/features/preproc_chart_icu.csv.gz",
                compression="gzip",
                index=False,
            )
            logger.info("[SUCCESSFULLY SAVED PROCESSED CHART EVENTS DATA]")


def generate_summary_icu(diag_flag, proc_flag, med_flag, out_flag, chart_flag):
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
        # summary['missing%']=100*(summary['missing_count']/summary['total_count'])
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
        # summary['missing_perc']=100*(summary['missing_count']/summary['total_count'])
        # summary=summary.fillna(0)

        #         final.groupby('itemid')['missing_count'].sum().reset_index()
        #         final.groupby('itemid')['total_count'].sum().reset_index()
        #         final.groupby('itemid')['missing%'].mean().reset_index()
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
    if diag_flag:
        if group_diag:
            logger.info("[FEATURE SELECTION DIAGNOSIS DATA]")
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

            logger.info(
                f"Total diagnosis records after feature selection: {diag.shape[0]:,}"
            )
            diag.to_csv(
                "./data/features/preproc_diag_icu.csv.gz",
                compression="gzip",
                index=False,
            )
            logger.info("[SUCCESSFULLY SAVED FILTERED DIAGNOSIS DATA]")

    if med_flag:
        if group_med:
            logger.info("[FEATURE SELECTION MEDICATIONS DATA]")
            med = pd.read_csv(
                "./data/features/preproc_med_icu.csv.gz", compression="gzip", header=0
            )
            original_count = len(med)
            features = pd.read_csv("./data/summary/med_features.csv", header=0)
            med = med[med["itemid"].isin(features["itemid"].unique())]

            filtered_count = len(med)
            dropped_count = original_count - filtered_count
            if dropped_count > 0:
                logger.info(
                    f"Feature selection dropped {dropped_count:,} medication records"
                )

            logger.info(
                f"Total medication records after feature selection: {med.shape[0]:,}"
            )
            med.to_csv(
                "./data/features/preproc_med_icu.csv.gz",
                compression="gzip",
                index=False,
            )
            logger.info("[SUCCESSFULLY SAVED FILTERED MEDICATIONS DATA]")

    if proc_flag:
        if group_proc:
            logger.info("[FEATURE SELECTION PROCEDURES DATA]")
            proc = pd.read_csv(
                "./data/features/preproc_proc_icu.csv.gz", compression="gzip", header=0
            )
            original_count = len(proc)
            features = pd.read_csv("./data/summary/proc_features.csv", header=0)
            proc = proc[proc["itemid"].isin(features["itemid"].unique())]

            filtered_count = len(proc)
            dropped_count = original_count - filtered_count
            if dropped_count > 0:
                logger.info(
                    f"Feature selection dropped {dropped_count:,} procedure records"
                )

            logger.info(
                f"Total procedure records after feature selection: {proc.shape[0]:,}"
            )
            proc.to_csv(
                "./data/features/preproc_proc_icu.csv.gz",
                compression="gzip",
                index=False,
            )
            logger.info("[SUCCESSFULLY SAVED FILTERED PROCEDURES DATA]")

    if out_flag:
        if group_out:
            logger.info("[FEATURE SELECTION OUTPUT EVENTS DATA]")
            out = pd.read_csv(
                "./data/features/preproc_out_icu.csv.gz", compression="gzip", header=0
            )
            original_count = len(out)
            features = pd.read_csv("./data/summary/out_features.csv", header=0)
            out = out[out["itemid"].isin(features["itemid"].unique())]

            filtered_count = len(out)
            dropped_count = original_count - filtered_count
            if dropped_count > 0:
                logger.info(
                    f"Feature selection dropped {dropped_count:,} output event records"
                )

            logger.info(
                f"Total output event records after feature selection: {out.shape[0]:,}"
            )
            out.to_csv(
                "./data/features/preproc_out_icu.csv.gz",
                compression="gzip",
                index=False,
            )
            logger.info("[SUCCESSFULLY SAVED FILTERED OUTPUT EVENTS DATA]")

    if chart_flag:
        if group_chart:
            logger.info("[FEATURE SELECTION CHART EVENTS DATA]")

            chart = pd.read_csv(
                "./data/features/preproc_chart_icu.csv.gz",
                compression="gzip",
                header=0,
                index_col=None,
            )
            original_count = len(chart)

            features = pd.read_csv("./data/summary/chart_features.csv", header=0)
            chart = chart[chart["itemid"].isin(features["itemid"].unique())]

            filtered_count = len(chart)
            dropped_count = original_count - filtered_count
            if dropped_count > 0:
                logger.info(
                    f"Feature selection dropped {dropped_count:,} chart event records"
                )

            logger.info(
                f"Total chart event records after feature selection: {chart.shape[0]:,}"
            )
            chart.to_csv(
                "./data/features/preproc_chart_icu.csv.gz",
                compression="gzip",
                index=False,
            )
            logger.info("[SUCCESSFULLY SAVED FILTERED CHART EVENTS DATA]")
