import datetime
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import importlib
from dataclasses import dataclass
import duckdb
import disease_cohort
import logging

importlib.reload(disease_cohort)
import disease_cohort

logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "./../..")
if not os.path.exists("./data/cohort"):
    os.makedirs("./data/cohort")


@dataclass
class ExtractionParams:
    """Dataclass to package extraction parameters"""

    use_ICU: str
    label: str
    time: int
    icd_code: str
    root_dir: str
    disease_label: str
    cohort_output: str = None
    summary_output: str = None

    # Derived boolean flags
    use_ICU_bool: bool = None
    use_mort: bool = None
    use_admn: bool = None
    use_los: bool = None
    use_disease: bool = None
    los: int = 0

    # Column mappings
    group_col: str = ""
    visit_col: str = ""
    admit_col: str = ""
    disch_col: str = ""
    death_col: str = ""
    adm_visit_col: str = ""

    def __post_init__(self):
        """Initialize derived parameters after creation"""
        # Set default output filenames if not provided
        if not self.cohort_output:
            self.cohort_output = (
                "cohort_"
                + self.use_ICU.lower()
                + "_"
                + self.label.lower().replace(" ", "_")
                + "_"
                + str(self.time)
                + "_"
                + self.disease_label
            )
        if not self.summary_output:
            self.summary_output = (
                "summary_"
                + self.use_ICU.lower()
                + "_"
                + self.label.lower().replace(" ", "_")
                + "_"
                + str(self.time)
                + "_"
                + self.disease_label
            )

        # Set boolean flags
        self.use_mort = self.label == "Mortality"
        self.use_admn = self.label == "Readmission"
        self.use_los = self.label == "Length of Stay"
        self.use_ICU_bool = self.use_ICU == "ICU"
        self.use_disease = self.icd_code != "No Disease Filter"

        if self.use_los:
            self.los = self.time

        # Set column mappings based on ICU usage
        if self.use_ICU_bool:
            self.group_col = "subject_id"
            self.visit_col = "stay_id"
            self.admit_col = "intime"
            self.disch_col = "outtime"
            self.death_col = "dod"
            self.adm_visit_col = "hadm_id"
        else:
            self.group_col = "subject_id"
            self.visit_col = "hadm_id"
            self.admit_col = "admittime"
            self.disch_col = "dischtime"
            self.death_col = "dod"


def get_visit_pts(
    mimic4_path: str,
    group_col: str,
    visit_col: str,
    admit_col: str,
    disch_col: str,
    adm_visit_col: str,
    use_mort: bool,
    use_los: bool,
    los: int,
    use_admn: bool,
    disease_label: str,
    use_ICU: bool,
):
    """Combines the MIMIC-IV core/patients table information with either the icu/icustays or core/admissions data.

    Parameters:
    mimic4_path: path to mimic-iv folder containing MIMIC-IV data
    group_col: patient identifier to group patients (normally subject_id)
    visit_col: visit identifier for individual patient visits (normally hadm_id or stay_id)
    admit_col: column for visit start date information (normally admittime or intime)
    disch_col: column for visit end date information (normally dischtime or outtime)
    use_ICU: describes whether to speficially look at ICU visits in icu/icustays OR look at general admissions from core/admissions
    """

    visit = None  # df containing visit information depending on using ICU or not
    if use_ICU:
        visit = pd.read_csv(
            mimic4_path + "icu/icustays.csv.gz",
            compression="gzip",
            header=0,
            index_col=None,
            parse_dates=[admit_col, disch_col],
        )
        if use_admn:
            # icustays doesn't have a way to identify if patient died during visit; must
            # use core/patients to remove such stay_ids for readmission labels
            pts = pd.read_csv(
                mimic4_path + "hosp/patients.csv.gz",
                compression="gzip",
                header=0,
                index_col=None,
                usecols=["subject_id", "dod"],
                parse_dates=["dod"],
            )
            visit = visit.merge(
                pts, how="inner", left_on="subject_id", right_on="subject_id"
            )
            original_count = len(visit)
            visit = visit.loc[(visit.dod.isna()) | (visit.dod >= visit[disch_col])]
            dropped_count = original_count - len(visit)
            if dropped_count > 0:
                logger.info(f"Dropped {dropped_count:,} ICU stays where patient died during visit")
            if len(disease_label):
                hids = disease_cohort.extract_diag_cohort(
                    visit["hadm_id"], disease_label, mimic4_path
                )
                visit = visit[visit["hadm_id"].isin(hids["hadm_id"])]
                print("[ READMISSION DUE TO " + disease_label + " ]")

    else:
        visit = pd.read_csv(
            mimic4_path + "hosp/admissions.csv.gz",
            compression="gzip",
            header=0,
            index_col=None,
            parse_dates=[admit_col, disch_col],
        )
        visit["los"] = visit[disch_col] - visit[admit_col]

        visit[admit_col] = pd.to_datetime(visit[admit_col])
        visit[disch_col] = pd.to_datetime(visit[disch_col])
        visit["los"] = pd.to_timedelta(visit[disch_col] - visit[admit_col], unit="h")
        visit["los"] = visit["los"].astype(str)
        visit[["days", "dummy", "hours"]] = visit["los"].str.split(" ", -1, expand=True)
        visit["los"] = pd.to_numeric(visit["days"])
        visit = visit.drop(columns=["days", "dummy", "hours"])

        if use_admn:
            # remove hospitalizations with a death; impossible for readmission for such visits
            original_count = len(visit)
            visit = visit.loc[visit.hospital_expire_flag == 0]
            dropped_count = original_count - len(visit)
            if dropped_count > 0:
                logger.info(f"Dropped {dropped_count:,} hospitalizations with death (readmission analysis)")
        if len(disease_label):
            hids = disease_cohort.extract_diag_cohort(
                visit["hadm_id"], disease_label, mimic4_path
            )
            visit = visit[visit["hadm_id"].isin(hids["hadm_id"])]
            print("[ READMISSION DUE TO " + disease_label + " ]")

    pts = pd.read_csv(
        mimic4_path + "hosp/patients.csv.gz",
        compression="gzip",
        header=0,
        index_col=None,
        usecols=[
            group_col,
            "anchor_year",
            "anchor_age",
            "anchor_year_group",
            "dod",
            "gender",
        ],
    )
    pts["yob"] = (
        pts["anchor_year"] - pts["anchor_age"]
    )  # get yob to ensure a given visit is from an adult
    pts["min_valid_year"] = pts["anchor_year"] + (
        2019 - pts["anchor_year_group"].str.slice(start=-4).astype(int)
    )

    # Define anchor_year corresponding to the anchor_year_group 2017-2019. This is later used to prevent consideration
    # of visits with prediction windows outside the dataset's time range (2008-2019)
    # [[group_col, visit_col, admit_col, disch_col]]
    if use_ICU:
        visit_pts = visit[
            [group_col, visit_col, adm_visit_col, admit_col, disch_col, "los"]
        ].merge(
            pts[
                [
                    group_col,
                    "anchor_year",
                    "anchor_age",
                    "yob",
                    "min_valid_year",
                    "dod",
                    "gender",
                ]
            ],
            how="inner",
            left_on=group_col,
            right_on=group_col,
        )
    else:
        visit_pts = visit[[group_col, visit_col, admit_col, disch_col, "los"]].merge(
            pts[
                [
                    group_col,
                    "anchor_year",
                    "anchor_age",
                    "yob",
                    "min_valid_year",
                    "dod",
                    "gender",
                ]
            ],
            how="inner",
            left_on=group_col,
            right_on=group_col,
        )

    # only take adult patients
    #     visit_pts['Age']=visit_pts[admit_col].dt.year - visit_pts['yob']
    #     visit_pts = visit_pts.loc[visit_pts['Age'] >= 18]
    visit_pts["Age"] = visit_pts["anchor_age"]
    original_count = len(visit_pts)
    visit_pts = visit_pts.loc[visit_pts["Age"] >= 18]
    dropped_count = original_count - len(visit_pts)
    if dropped_count > 0:
        logger.info(f"Dropped {dropped_count:,} visits from patients under 18 years old")

    ##Add Demo data
    eth = pd.read_csv(
        mimic4_path + "hosp/admissions.csv.gz",
        compression="gzip",
        header=0,
        usecols=["hadm_id", "insurance", "race"],
        index_col=None,
    )
    visit_pts = visit_pts.merge(eth, how="inner", left_on="hadm_id", right_on="hadm_id")

    if use_ICU:
        return visit_pts[
            [
                group_col,
                visit_col,
                adm_visit_col,
                admit_col,
                disch_col,
                "los",
                "min_valid_year",
                "dod",
                "Age",
                "gender",
                "race",
                "insurance",
            ]
        ]
    else:
        return visit_pts.dropna(subset=["min_valid_year"])[
            [
                group_col,
                visit_col,
                admit_col,
                disch_col,
                "los",
                "min_valid_year",
                "dod",
                "Age",
                "gender",
                "race",
                "insurance",
            ]
        ]


def get_visit_pts_db(conn, params: ExtractionParams):
    """DuckDB version of get_visit_pts using existing connection with imported tables.

    Parameters:
    conn: DuckDB connection object with MIMIC-IV tables already imported
    params: ExtractionParams object containing all configuration
    """

    visit = None  # df containing visit information depending on using ICU or not
    if params.use_ICU_bool:
        # Query ICU stays data from existing table
        visit_query = f"""
        SELECT *,
               EXTRACT(DAYS FROM (CAST({params.disch_col} AS TIMESTAMP) - CAST({params.admit_col} AS TIMESTAMP))) as los
        FROM icustays
        """
        visit = conn.execute(visit_query).df()

        if params.use_admn:
            # icustays doesn't have a way to identify if patient died during visit; must
            # use patients table to remove such stay_ids for readmission labels
            pts_query = """
            SELECT subject_id, dod
            FROM patients
            """
            pts = conn.execute(pts_query).df()

            visit = visit.merge(
                pts, how="inner", left_on="subject_id", right_on="subject_id"
            )
            visit = visit.loc[
                (visit.dod.isna()) | (visit.dod >= visit[params.disch_col])
            ]
            if len(params.disease_label):
                hids = disease_cohort.extract_diag_cohort(
                    visit["hadm_id"],
                    params.disease_label,
                    params.root_dir + "/mimiciv/3.0/",
                )
                visit = visit[visit["hadm_id"].isin(hids["hadm_id"])]
                print("[ READMISSION DUE TO " + params.disease_label + " ]")

    else:
        # Query admissions data from existing table and calculate LOS
        visit_query = f"""
        SELECT *,
               EXTRACT(DAYS FROM (CAST({params.disch_col} AS TIMESTAMP) - CAST({params.admit_col} AS TIMESTAMP))) as los,
               CAST(hospital_expire_flag AS INTEGER) as hospital_expire_flag
        FROM admissions
        """
        visit = conn.execute(visit_query).df()

        if params.use_admn:
            # remove hospitalizations with a death; impossible for readmission for such visits
            visit = visit.loc[visit.hospital_expire_flag == 0]
        if len(params.disease_label):
            hids = disease_cohort.extract_diag_cohort(
                visit["hadm_id"],
                params.disease_label,
                params.root_dir + "/mimiciv/3.0/",
            )
            visit = visit[visit["hadm_id"].isin(hids["hadm_id"])]
            print("[ READMISSION DUE TO " + params.disease_label + " ]")

    # Query patients data from existing table
    pts_cols = [
        params.group_col,
        "anchor_year",
        "anchor_age",
        "anchor_year_group",
        "dod",
        "gender",
    ]
    pts_query = f"""
    SELECT {', '.join(pts_cols)},
           (CAST(anchor_year AS INTEGER) - CAST(anchor_age AS INTEGER)) as yob,
           CAST(anchor_year AS INTEGER) + (2019 - CAST(RIGHT(anchor_year_group, 4) AS INTEGER)) as min_valid_year
    FROM patients
    """
    pts = conn.execute(pts_query).df()

    # Merge visit and patient data
    if params.use_ICU_bool:
        visit_pts_query = f"""
        SELECT v.{params.group_col}, v.{params.visit_col}, v.{params.adm_visit_col},
               v.{params.admit_col}, v.{params.disch_col}, v.los,
               p.anchor_year, p.anchor_age, p.yob, p.min_valid_year, p.dod, p.gender
        FROM ({visit_query}) v
        INNER JOIN ({pts_query}) p ON v.{params.group_col} = p.{params.group_col}
        WHERE CAST(p.anchor_age AS INTEGER) >= 18
        """
    else:
        visit_pts_query = f"""
        SELECT v.{params.group_col}, v.{params.visit_col},
               v.{params.admit_col}, v.{params.disch_col}, v.los,
               p.anchor_year, p.anchor_age, p.yob, p.min_valid_year, p.dod, p.gender
        FROM ({visit_query}) v
        INNER JOIN ({pts_query}) p ON v.{params.group_col} = p.{params.group_col}
        WHERE CAST(p.anchor_age AS INTEGER) >= 18 AND p.min_valid_year IS NOT NULL
        """

    visit_pts = conn.execute(visit_pts_query).df()
    visit_pts["Age"] = visit_pts["anchor_age"]

    # Add demographic data from admissions table
    eth_query = """
    SELECT hadm_id, insurance, race
    FROM admissions
    """
    eth = conn.execute(eth_query).df()
    visit_pts = visit_pts.merge(eth, how="inner", left_on="hadm_id", right_on="hadm_id")

    if params.use_ICU_bool:
        return visit_pts[
            [
                params.group_col,
                params.visit_col,
                params.adm_visit_col,
                params.admit_col,
                params.disch_col,
                "los",
                "min_valid_year",
                "dod",
                "Age",
                "gender",
                "race",
                "insurance",
            ]
        ]
    else:
        return visit_pts[
            [
                params.group_col,
                params.visit_col,
                params.admit_col,
                params.disch_col,
                "los",
                "min_valid_year",
                "dod",
                "Age",
                "gender",
                "race",
                "insurance",
            ]
        ]


def validate_row(row, ctrl, invalid, max_year, disch_col, valid_col, gap):
    """Checks if visit's prediction window potentially extends beyond the dataset range (2008-2019).
    An 'invalid row' is NOT guaranteed to be outside the range, only potentially outside due to
    de-identification of MIMIC-IV being done through 3-year time ranges.

    To be invalid, the end of the prediction window's year must both extend beyond the maximum seen year
    for a patient AND beyond the year that corresponds to the 2017-2019 anchor year range for a patient"""
    print("disch_col", row[disch_col])
    print(gap)
    pred_year = (row[disch_col] + gap).year
    if max_year < pred_year and pred_year > row[valid_col]:
        invalid = invalid.append(row)
    else:
        ctrl = ctrl.append(row)
    return ctrl, invalid


def partition_by_los(
    df: pd.DataFrame,
    los: int,
    group_col: str,
    visit_col: str,
    admit_col: str,
    disch_col: str,
    valid_col: str,
):
    # Type check and cast 'los' column to numeric if necessary
    if not pd.api.types.is_numeric_dtype(df["los"]):
        df["los"] = pd.to_numeric(df["los"], errors="coerce")

    invalid = df.loc[
        (df[admit_col].isna()) | (df[disch_col].isna()) | (df["los"].isna())
    ]
    cohort = df.loc[
        (~df[admit_col].isna()) & (~df[disch_col].isna()) & (~df["los"].isna())
    ]

    # cohort=cohort.fillna(0)
    pos_cohort = cohort[cohort["los"] > los]
    neg_cohort = cohort[cohort["los"] <= los]
    neg_cohort = neg_cohort.fillna(0)
    pos_cohort = pos_cohort.fillna(0)

    pos_cohort["label"] = 1
    neg_cohort["label"] = 0

    cohort = pd.concat([pos_cohort, neg_cohort], axis=0)
    cohort = cohort.sort_values(by=[group_col, admit_col])
    # print("cohort",cohort.shape)
    print("[ LOS LABELS FINISHED ]")
    return cohort, invalid


def partition_by_readmit(
    df: pd.DataFrame,
    gap: datetime.timedelta,
    group_col: str,
    visit_col: str,
    admit_col: str,
    disch_col: str,
    valid_col: str,
):
    """Applies labels to individual visits according to whether or not a readmission has occurred within the specified `gap` days.
    For a given visit, another visit must occur within the gap window for a positive readmission label.
    The gap window starts from the disch_col time and the admit_col of subsequent visits are considered."""

    case = pd.DataFrame()  # hadm_ids with readmission within the gap period
    ctrl = pd.DataFrame()  # hadm_ids without readmission within the gap period
    invalid = pd.DataFrame()  # hadm_ids that are not considered in the cohort

    # Iterate through groupbys based on group_col (subject_id). Data is sorted by subject_id and admit_col (admittime)
    # to ensure that the most current hadm_id is last in a group.
    # grouped= df[[group_col, visit_col, admit_col, disch_col, valid_col]].sort_values(by=[group_col, admit_col]).groupby(group_col)
    grouped = df.sort_values(by=[group_col, admit_col]).groupby(group_col)
    for subject, group in tqdm(grouped):
        max_year = group.max()[disch_col].year

        if group.shape[0] <= 1:
            # ctrl, invalid = validate_row(group.iloc[0], ctrl, invalid, max_year, disch_col, valid_col, gap)   # A group with 1 row has no readmission; goes to ctrl
            ctrl = ctrl.append(group.iloc[0])
        else:
            for idx in range(group.shape[0] - 1):
                visit_time = group.iloc[idx][
                    disch_col
                ]  # For each index (a unique hadm_id), get its timestamp
                if (
                    group.loc[
                        (
                            group[admit_col] > visit_time
                        )  # Readmissions must come AFTER the current timestamp
                        & (
                            group[admit_col] - visit_time <= gap
                        )  # Distance between a timestamp and readmission must be within gap
                    ].shape[0]
                    >= 1
                ):  # If ANY rows meet above requirements, a readmission has occurred after that visit
                    case = case.append(group.iloc[idx])
                else:
                    # If no readmission is found, only add to ctrl if prediction window is guaranteed to be within the
                    # time range of the dataset (2008-2019). Visits with prediction windows existing in potentially out-of-range
                    # dates (like 2018-2020) are excluded UNLESS the prediction window takes place the same year as the visit,
                    # in which case it is guaranteed to be within 2008-2019

                    ctrl = ctrl.append(group.iloc[idx])

            # ctrl, invalid = validate_row(group.iloc[-1], ctrl, invalid, max_year, disch_col, valid_col, gap)  # The last hadm_id datewise is guaranteed to have no readmission logically
            ctrl = ctrl.append(group.iloc[-1])
            # print(f"[ {gap.days} DAYS ] {case.shape[0] + ctrl.shape[0]}/{df.shape[0]} {visit_col}s processed")

    print("[ READMISSION LABELS FINISHED ]")
    return case, ctrl, invalid


def partition_by_mort(
    df: pd.DataFrame,
    group_col: str,
    visit_col: str,
    admit_col: str,
    disch_col: str,
    death_col: str,
):
    """Applies labels to individual visits according to whether or not a death has occurred within
    the times of the specified admit_col and disch_col"""

    invalid = df.loc[(df[admit_col].isna()) | (df[disch_col].isna())]

    cohort = df.loc[(~df[admit_col].isna()) & (~df[disch_col].isna())]

    #     cohort["label"] = (
    #         (~cohort[death_col].isna())
    #         & (cohort[death_col] >= cohort[admit_col])
    #         & (cohort[death_col] <= cohort[disch_col])
    #     )
    #     cohort["label"] = cohort["label"].astype("Int32")
    # print("cohort",cohort.shape)
    # print(np.where(~cohort[death_col].isna(),1,0))
    # print(np.where(cohort.loc[death_col] >= cohort.loc[admit_col],1,0))
    # print(np.where(cohort.loc[death_col] <= cohort.loc[disch_col],1,0))
    cohort["label"] = 0
    # cohort=cohort.fillna(0)
    pos_cohort = cohort[~cohort[death_col].isna()]
    neg_cohort = cohort[cohort[death_col].isna()]
    neg_cohort = neg_cohort.fillna(0)
    pos_cohort = pos_cohort.fillna(0)
    pos_cohort[death_col] = pd.to_datetime(pos_cohort[death_col])

    pos_cohort["label"] = np.where(
        (pos_cohort[death_col] >= pos_cohort[admit_col])
        & (pos_cohort[death_col] <= pos_cohort[disch_col]),
        1,
        0,
    )

    pos_cohort["label"] = pos_cohort["label"].astype("Int32")
    cohort = pd.concat([pos_cohort, neg_cohort], axis=0)
    cohort = cohort.sort_values(by=[group_col, admit_col])
    # print("cohort",cohort.shape)
    print("[ MORTALITY LABELS FINISHED ]")
    return cohort, invalid


def get_case_ctrls(
    df: pd.DataFrame,
    gap: int,
    group_col: str,
    visit_col: str,
    admit_col: str,
    disch_col: str,
    valid_col: str,
    death_col: str,
    use_mort=False,
    use_admn=False,
    use_los=False,
) -> pd.DataFrame:
    """Handles logic for creating the labelled cohort based on arguments passed to extract().

    Parameters:
    df: dataframe with patient data
    gap: specified time interval gap for readmissions
    group_col: patient identifier to group patients (normally subject_id)
    visit_col: visit identifier for individual patient visits (normally hadm_id or stay_id)
    admit_col: column for visit start date information (normally admittime or intime)
    disch_col: column for visit end date information (normally dischtime or outtime)
    valid_col: generated column containing a patient's year that corresponds to the 2017-2019 anchor time range
    dod_col: Date of death column
    """

    case = None  # hadm_ids with readmission within the gap period
    ctrl = None  # hadm_ids without readmission within the gap period
    invalid = None  # hadm_ids that are not considered in the cohort

    if use_mort:
        return partition_by_mort(
            df, group_col, visit_col, admit_col, disch_col, death_col
        )
    elif use_admn:
        gap = datetime.timedelta(days=gap)
        # transform gap into a timedelta to compare with datetime columns
        case, ctrl, invalid = partition_by_readmit(
            df, gap, group_col, visit_col, admit_col, disch_col, valid_col
        )

        # case hadm_ids are labelled 1 for readmission, ctrls have a 0 label
        case["label"] = np.ones(case.shape[0]).astype(int)
        ctrl["label"] = np.zeros(ctrl.shape[0]).astype(int)

        return pd.concat([case, ctrl], axis=0), invalid
    elif use_los:
        return partition_by_los(
            df, gap, group_col, visit_col, admit_col, disch_col, death_col
        )

    # print(f"[ {gap.days} DAYS ] {invalid.shape[0]} hadm_ids are invalid")


def extract_data_before_visit_pts(
    use_ICU: str,
    label: str,
    time: int,
    icd_code: str,
    root_dir,
    disease_label,
    cohort_output=None,
    summary_output=None,
):
    """First part: Prepares extraction configuration and prints extraction information.

    Returns:
        ExtractionParams: Configured parameters for extraction
    """
    # Create parameter object
    params = ExtractionParams(
        use_ICU=use_ICU,
        label=label,
        time=time,
        icd_code=icd_code,
        root_dir=root_dir,
        disease_label=disease_label,
        cohort_output=cohort_output,
        summary_output=summary_output,
    )

    print("===========MIMIC-IV v3.0============")

    # Print extraction information
    if params.icd_code == "No Disease Filter":
        if len(params.disease_label):
            print(
                f"EXTRACTING FOR: | {params.use_ICU.upper()} | {params.label.upper()} DUE TO {params.disease_label.upper()} | {str(params.time)} | "
            )
        else:
            print(
                f"EXTRACTING FOR: | {params.use_ICU.upper()} | {params.label.upper()} | {str(params.time)} |"
            )
    else:
        if len(params.disease_label):
            print(
                f"EXTRACTING FOR: | {params.use_ICU.upper()} | {params.label.upper()} DUE TO {params.disease_label.upper()} | ADMITTED DUE TO {params.icd_code.upper()} | {str(params.time)} |"
            )
        else:
            print(
                f"EXTRACTING FOR: | {params.use_ICU.upper()} | {params.label.upper()} | ADMITTED DUE TO {params.icd_code.upper()} | {str(params.time)} |"
            )

    return params


def extract_data_get_visit_pts(params: ExtractionParams):
    """Second part: Calls get_visit_pts to retrieve patient data.

    Args:
        params: ExtractionParams object with configuration

    Returns:
        DataFrame: Patient visit data from get_visit_pts
    """
    pts = get_visit_pts(
        mimic4_path=params.root_dir + "/mimiciv/3.0/",
        group_col=params.group_col,
        visit_col=params.visit_col,
        admit_col=params.admit_col,
        disch_col=params.disch_col,
        adm_visit_col=params.adm_visit_col,
        use_mort=params.use_mort,
        use_los=params.use_los,
        los=params.los,
        use_admn=params.use_admn,
        disease_label=params.disease_label,
        use_ICU=params.use_ICU_bool,
    )
    return pts


def extract_data_get_visit_pts_db(conn, params: ExtractionParams):
    """Second part: Calls get_visit_pts_db to retrieve patient data using DuckDB.

    Args:
        conn: DuckDB connection object with MIMIC-IV tables already imported
        params: ExtractionParams object with configuration

    Returns:
        DataFrame: Patient visit data from get_visit_pts_db
    """
    pts = get_visit_pts_db(conn, params)
    return pts


def extract_data_after_visit_pts(pts, params: ExtractionParams):
    """Third part: Processes visit patient data into final cohort and saves results.

    Parameters:
        pts: DataFrame containing visit patient data from get_visit_pts
        params: ExtractionParams object containing configuration

    Returns:
        str: cohort output filename
    """
    # cols to be extracted from get_case_ctrls
    cols = [
        params.group_col,
        params.visit_col,
        params.admit_col,
        params.disch_col,
        "Age",
        "gender",
        "ethnicity",
        "insurance",
        "label",
    ]

    if params.use_mort:
        cols.append(params.death_col)
        cohort, invalid = get_case_ctrls(
            pts,
            None,
            params.group_col,
            params.visit_col,
            params.admit_col,
            params.disch_col,
            "min_valid_year",
            params.death_col,
            use_mort=True,
            use_admn=False,
            use_los=False,
        )
    elif params.use_admn:
        interval = params.time
        cohort, invalid = get_case_ctrls(
            pts,
            interval,
            params.group_col,
            params.visit_col,
            params.admit_col,
            params.disch_col,
            "min_valid_year",
            params.death_col,
            use_mort=False,
            use_admn=True,
            use_los=False,
        )
    elif params.use_los:
        cohort, invalid = get_case_ctrls(
            pts,
            params.los,
            params.group_col,
            params.visit_col,
            params.admit_col,
            params.disch_col,
            "min_valid_year",
            params.death_col,
            use_mort=False,
            use_admn=False,
            use_los=True,
        )

    if params.use_ICU_bool:
        cols.append(params.adm_visit_col)

    # Apply disease filter if specified
    cohort_output = params.cohort_output
    summary_output = params.summary_output

    if params.use_disease:
        hids = disease_cohort.extract_diag_cohort(
            cohort["hadm_id"], params.icd_code, params.root_dir + "/mimiciv/3.0/"
        )
        cohort = cohort[cohort["hadm_id"].isin(hids["hadm_id"])]
        cohort_output = cohort_output + "_" + params.icd_code
        summary_output = summary_output + "_" + params.icd_code

    # save output
    cohort = cohort.rename(columns={"race": "ethnicity"})
    cohort[cols].to_csv(
        params.root_dir + "/data/cohort/" + cohort_output + ".csv.gz",
        index=False,
        compression="gzip",
    )
    print("[ COHORT SUCCESSFULLY SAVED ]")

    summary = "\n".join(
        [
            f"{params.label} FOR {params.use_ICU} DATA",
            f"# Admission Records: {cohort.shape[0]}",
            f"# Patients: {cohort[params.group_col].nunique()}",
            f"# Positive cases: {cohort[cohort['label']==1].shape[0]}",
            f"# Negative cases: {cohort[cohort['label']==0].shape[0]}",
        ]
    )

    # save basic summary of data
    with open(f"./data/cohort/{summary_output}.txt", "w") as f:
        f.write(summary)

    print("[ SUMMARY SUCCESSFULLY SAVED ]")
    print(summary)

    return cohort_output


def extract_data_db(
    conn,
    use_ICU: str,
    label: str,
    time: int,
    icd_code: str,
    root_dir,
    disease_label,
    cohort_output=None,
    summary_output=None,
):
    """DuckDB version: Extracts cohort data and summary from MIMIC-IV data based on provided parameters.

    Parameters:
    conn: DuckDB connection object with MIMIC-IV tables already imported
    cohort_output: name of labelled cohort output file
    summary_output: name of summary output file
    use_ICU: state whether to use ICU patient data or not
    label: Can either be '{day} day Readmission' or 'Mortality', decides what binary data label signifies"""

    # Part 1: Prepare extraction configuration
    params = extract_data_before_visit_pts(
        use_ICU,
        label,
        time,
        icd_code,
        root_dir,
        disease_label,
        cohort_output,
        summary_output,
    )

    # Part 2: Get visit patient data using DuckDB
    pts = extract_data_get_visit_pts_db(conn, params)

    # Part 3: Process the cohort data and save results
    return extract_data_after_visit_pts(pts, params)


def extract_data(
    use_ICU: str,
    label: str,
    time: int,
    icd_code: str,
    root_dir,
    disease_label,
    cohort_output=None,
    summary_output=None,
):
    """Extracts cohort data and summary from MIMIC-IV data based on provided parameters.

    Parameters:
    cohort_output: name of labelled cohort output file
    summary_output: name of summary output file
    use_ICU: state whether to use ICU patient data or not
    label: Can either be '{day} day Readmission' or 'Mortality', decides what binary data label signifies"""

    # Part 1: Prepare extraction configuration
    params = extract_data_before_visit_pts(
        use_ICU,
        label,
        time,
        icd_code,
        root_dir,
        disease_label,
        cohort_output,
        summary_output,
    )

    # Part 2: Get visit patient data
    pts = extract_data_get_visit_pts(params)

    # Part 3: Process the cohort data and save results
    return extract_data_after_visit_pts(pts, params)


if __name__ == "__main__":
    # use_ICU = input("Use ICU Data? (ICU/Non_ICU)\n").strip()
    # label = input("Please input the intended label:\n").strip()

    # extract(use_ICU, label)

    response = input("Extra all datasets? (y/n)").strip().lower()
    if response == "y":
        extract_data("ICU", "Mortality")
        extract_data("Non-ICU", "Mortality")

        extract_data("ICU", "30 Day Readmission")
        extract_data("Non-ICU", "30 Day Readmission")

        extract_data("ICU", "60 Day Readmission")
        extract_data("Non-ICU", "60 Day Readmission")

        extract_data("ICU", "120 Day Readmission")
        extract_data("Non-ICU", "120 Day Readmission")
