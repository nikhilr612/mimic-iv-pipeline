#!/usr/bin/env python3
"""
MIMIC-IV Length of Stay (LOS) Prediction Pipeline - Version 3 ICU Only
Single script that replicates the mainPipeline.ipynb functionality for MIMIC-IV v3 ICU data
"""

import os
import sys

import pandas as pd

# =============================================================================
# GLOBAL CONFIGURATION PARAMETERS
# =============================================================================

# Fixed Configuration (no longer configurable)
VERSION = "Version 3"  # Fixed to Version 3
DATA_SOURCE = "ICU"  # Fixed to ICU data

# LOS Configuration
LOS_THRESHOLD_TYPE = "Length of Stay ge 3"  # Options: 'Length of Stay ge 3', 'Length of Stay ge 7', 'Custom'
CUSTOM_LOS_THRESHOLD = 3  # Used if LOS_THRESHOLD_TYPE is 'Custom'

# Disease Filter Configuration
DISEASE_FILTER = "No Disease Filter"  # Options: 'No Disease Filter', 'Heart Failure', 'CKD', 'CAD', 'COPD'

# Feature Selection Configuration (Diagnoses always included for ICU)
FEATURE_FLAGS = {
    "diagnosis": True,  # Always True - required
    "procedures": True,
    "medications": True,
    "output_events": True,  # ICU specific
    "chart_events": True,  # ICU specific
}

# Time Series Configuration
TIME_WINDOW_TYPE = (
    "First 24 hours"  # Options: 'First 12 hours', 'First 24 hours', 'Custom'
)
CUSTOM_TIME_WINDOW = 24  # Used if TIME_WINDOW_TYPE is 'Custom'

BUCKET_SIZE_TYPE = (
    "1 hour"  # Options: '1 hour', '2 hour', '3 hour', '4 hour', '5 hour', 'Custom'
)
CUSTOM_BUCKET_SIZE = 1  # Used if BUCKET_SIZE_TYPE is 'Custom'

IMPUTATION_METHOD = "forward fill and mean"  # Options: 'No Imputation', 'forward fill and mean', 'forward fill and median'

# Directory Configuration
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

MIMIC_DB_PATH = "../mimiciv.duckdb"

# =============================================================================
# SETUP MODULE PATHS AND IMPORTS
# =============================================================================

# Add module paths to sys.path
module_paths = [
    os.path.join(ROOT_DIR, "preprocessing", "day_intervals_preproc"),
    os.path.join(ROOT_DIR, "utils"),
    os.path.join(ROOT_DIR, "preprocessing"),
    os.path.join(ROOT_DIR, "preprocessing", "hosp_module_preproc"),
    os.path.join(ROOT_DIR, "model"),
]

for module_path in module_paths:
    if module_path not in sys.path:
        sys.path.insert(0, module_path)


# Import required modules with error handling
def import_modules():
    """Import all required modules"""
    modules = {}

    # Import cohort module v3
    try:
        import day_intervals_cohort_v3

        modules["cohort"] = day_intervals_cohort_v3
        print("Successfully imported day_intervals_cohort_v3")
    except ImportError as e:
        print(f"Failed to import day_intervals_cohort_v3: {e}")
        print("Please ensure you're running from the project root directory")
        sys.exit(1)

    # Import feature selection ICU
    try:
        import feature_selection_icu

        modules["feature_icu"] = feature_selection_icu
        print("Successfully imported feature_selection_icu")
    except ImportError as e:
        print(f"Failed to import feature_selection_icu: {e}")
        sys.exit(1)

    # Import data generation ICU
    try:
        import data_generation_icu

        modules["data_gen_icu"] = data_generation_icu
        print("Successfully imported data_generation_icu")
    except ImportError as e:
        print(f"Failed to import data_generation_icu: {e}")
        sys.exit(1)

    return modules


# Import all modules
MODULES = import_modules()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def parse_los_threshold():
    """Parse LOS threshold from configuration"""
    if LOS_THRESHOLD_TYPE == "Custom":
        return CUSTOM_LOS_THRESHOLD
    else:
        return int(LOS_THRESHOLD_TYPE.split()[-1])


def parse_time_window():
    """Parse time window from configuration"""
    if TIME_WINDOW_TYPE == "Custom":
        return CUSTOM_TIME_WINDOW
    else:
        return int(TIME_WINDOW_TYPE.split()[1])


def parse_bucket_size():
    """Parse bucket size from configuration"""
    if BUCKET_SIZE_TYPE == "Custom":
        return CUSTOM_BUCKET_SIZE
    else:
        return int(BUCKET_SIZE_TYPE.split()[0])


def parse_imputation():
    """Parse imputation method"""
    if IMPUTATION_METHOD == "forward fill and mean":
        return "Mean"
    elif IMPUTATION_METHOD == "forward fill and median":
        return "Median"
    else:
        return False


def get_disease_icd_code():
    """Get ICD code for disease filter"""
    disease_mapping = {
        "Heart Failure": "I50",
        "CKD": "N18",
        "COPD": "J44",
        "CAD": "I25",
        "No Disease Filter": "No Disease Filter",
    }
    return disease_mapping.get(DISEASE_FILTER, "No Disease Filter")


def get_version_path():
    """Get version path for MIMIC-IV v3"""
    return "mimiciv/3.1"


# =============================================================================
# MAIN PIPELINE FUNCTIONS
# =============================================================================


def extract_cohort():
    """Extract cohort data using MIMIC-IV v3 ICU data"""
    print("=" * 60)
    print("STEP 1: COHORT EXTRACTION")
    print("=" * 60)

    # Parse parameters
    los_threshold = parse_los_threshold()
    icd_code = get_disease_icd_code()

    print(f"Configuration:")
    print(f"  - Version: {VERSION} (MIMIC-IV v3.1)")
    print(f"  - Data Source: {DATA_SOURCE}")
    print(f"  - LOS Threshold: {los_threshold} days")
    print(f"  - Disease Filter: {DISEASE_FILTER}")
    print(f"  - Diagnoses: Always included for ICU data")

    # Extract data using version 3 module
    try:
        import duckdb

        conn = duckdb.connect(MIMIC_DB_PATH)

        cohort_output = MODULES["cohort"].extract_data_db(
            conn,
            use_ICU=DATA_SOURCE,
            label="Length of Stay",
            time=los_threshold,
            icd_code=icd_code,
            root_dir=ROOT_DIR,
            disease_label="",
            cohort_output=None,
            summary_output=None,
        )
        print("Cohort extraction completed successfully")
        print(f"Cohort saved as: {cohort_output}")
        return cohort_output
    except Exception as e:
        print(f"Error during cohort extraction: {e}")
        raise


def feature_selection(cohort_output):
    """Perform ICU feature selection with diagnoses included"""
    print("=" * 60)
    print("STEP 2: ICU FEATURE SELECTION")
    print("=" * 60)

    version_path = get_version_path()

    print(f"Processing features for ICU data from {version_path}...")
    print(f"Selected features: {[k for k, v in FEATURE_FLAGS.items() if v]}")
    print("Diagnoses: ENABLED (required for ICU)")

    try:
        # ICU feature selection with diagnoses always enabled
        MODULES["feature_icu"].feature_icu(
            cohort_output=cohort_output,
            version_path=version_path,
            diag_flag=True,  # Always True for diagnoses
            proc_flag=FEATURE_FLAGS["procedures"],
            out_flag=FEATURE_FLAGS["output_events"],
            chart_flag=FEATURE_FLAGS["chart_events"],
            med_flag=FEATURE_FLAGS["medications"],
        )
        print("ICU feature selection completed successfully.")
    except Exception as e:
        print(f"Error during feature selection: {e}")
        raise


def generate_time_series(cohort_output):
    """Generate ICU time series data"""
    print("=" * 60)
    print("STEP 3: ICU TIME SERIES GENERATION")
    print("=" * 60)

    # Parse time series parameters
    include_hours = parse_time_window()
    bucket_size = parse_bucket_size()
    impute_method = parse_imputation()

    # Set task flags
    data_los = True
    data_mort = False
    data_admn = False

    print(f"Time series configuration:")
    print(f"  - Include: First {include_hours} hours")
    print(f"  - Bucket size: {bucket_size} hour(s)")
    print(f"  - Imputation: {impute_method}")
    print("  - Task: Length of Stay prediction")

    try:
        # Create ICU time series generator
        gen = MODULES["data_gen_icu"].Generator(
            cohort_output=cohort_output,
            if_mort=data_mort,
            if_admn=data_admn,
            if_los=data_los,
            feat_cond=True,  # Always True for diagnoses
            feat_proc=FEATURE_FLAGS["procedures"],
            feat_out=FEATURE_FLAGS["output_events"],
            feat_chart=FEATURE_FLAGS["chart_events"],
            feat_med=FEATURE_FLAGS["medications"],
            impute=impute_method,
            include_time=include_hours,
            bucket=bucket_size,
            predW=0,  # No prediction window for LOS
        )
        print("ICU time series generation completed successfully.")
        return gen
    except Exception as e:
        print(f"Error during time series generation: {e}")
        raise


def generate_summary(cohort_output):
    """Generate data summary and statistics"""
    print("=" * 60)
    print("STEP 4: DATA SUMMARY")
    print("=" * 60)

    # Load cohort data
    cohort_path = f"./data/cohort/{cohort_output}.csv.gz"
    if os.path.exists(cohort_path):
        try:
            cohort_df = pd.read_csv(cohort_path, compression="gzip")

            print("ICU Cohort Statistics:")
            print(f"  - Total patients: {cohort_df['subject_id'].nunique()}")
            print(f"  - Total ICU stays: {len(cohort_df)}")
            print(
                f"  - Positive cases (LOS > threshold): {(cohort_df['label'] == 1).sum()}"
            )
            print(
                f"  - Negative cases (LOS <= threshold): {(cohort_df['label'] == 0).sum()}"
            )
            print(
                f"  - Label distribution: {cohort_df['label'].value_counts().to_dict()}"
            )

            if "Age" in cohort_df.columns:
                print(f"  - Mean age: {cohort_df['Age'].mean():.1f}")
                print(
                    f"  - Age range: {cohort_df['Age'].min()}-{cohort_df['Age'].max()}"
                )

            if "gender" in cohort_df.columns:
                print(
                    f"  - Gender distribution: {cohort_df['gender'].value_counts().to_dict()}"
                )

            # Save summary
            os.makedirs("./data/summary", exist_ok=True)
            summary_path = f"./data/summary/{cohort_output}_summary.txt"
            with open(summary_path, "w") as f:
                f.write(f"MIMIC-IV v3 ICU Length of Stay Prediction Data Summary\n")
                f.write(f"Generated on: {pd.Timestamp.now()}\n")
                f.write(f"Configuration: ICU data, {LOS_THRESHOLD_TYPE}\n")
                f.write(f"Diagnoses: INCLUDED\n\n")
                f.write(f"Total patients: {cohort_df['subject_id'].nunique()}\n")
                f.write(f"Total ICU stays: {len(cohort_df)}\n")
                f.write(f"Positive cases: {(cohort_df['label'] == 1).sum()}\n")
                f.write(f"Negative cases: {(cohort_df['label'] == 0).sum()}\n")
            print(f"Summary saved to: {summary_path}")

        except Exception as e:
            print(f"Error generating summary: {e}")
    else:
        print(f"Warning: Cohort file not found at {cohort_path}")

    print("Data processing completed successfully!")
    print("Ready for model training with your preferred ML framework.")


def main():
    """Main pipeline execution"""
    print("MIMIC-IV v3 ICU Length of Stay Prediction Pipeline")
    print("=" * 60)
    print("Configuration:")
    print(f"  Version: {VERSION} (MIMIC-IV v3.1)")
    print(f"  Data Source: {DATA_SOURCE}")
    print(f"  LOS Threshold: {LOS_THRESHOLD_TYPE}")
    print(f"  Disease Filter: {DISEASE_FILTER}")
    print(f"  Features: {[k for k, v in FEATURE_FLAGS.items() if v]}")
    print(f"  Diagnoses: ALWAYS INCLUDED")
    print(f"  Time Window: {TIME_WINDOW_TYPE}")
    print(f"  Bucket Size: {BUCKET_SIZE_TYPE}")
    print(f"  Imputation: {IMPUTATION_METHOD}")
    print("=" * 60)

    try:
        # Step 1: Extract ICU cohort
        cohort_output = extract_cohort()

        # Step 2: ICU feature selection (with diagnoses)
        feature_selection(cohort_output)

        # Step 3: Generate ICU time series
        generate_time_series(cohort_output)

        # Step 4: Generate summary
        generate_summary(cohort_output)

        print("=" * 60)
        print("ICU DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Output files:")
        print(f"  - Cohort: ./data/cohort/{cohort_output}.csv.gz")
        print("  - Features: ./data/features/preproc_*.csv.gz")
        print("  - Time series: ./data/dict/ and ./data/csv/")
        print(f"  - Summary: ./data/summary/{cohort_output}_summary.txt")
        print("=" * 60)
        print("Data includes:")
        print("  - ICU stays from MIMIC-IV v3.1")
        print("  - Diagnosis codes (ICD-9/ICD-10)")
        print("  - Procedures, medications, chart events, output events")
        print("  - Time-series features with configurable bucketing")
        print("=" * 60)
        print("Next steps:")
        print("  - Use ./data/csv/labels.csv for training labels")
        print("  - Use ./data/csv/{patient_id}/ files for patient features")
        print("  - Load ./data/dict/ files for vocabulary and metadata")
        print("=" * 60)

    except Exception as e:
        print(f"ERROR: Pipeline failed with exception: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Ensure required directories exist
    required_dirs = [
        "./data/cohort",
        "./data/features",
        "./data/summary",
        "./data/dict",
        "./data/csv",
    ]

    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)

    # Verify MIMIC-IV v3.1 data path exists
    mimic_path = "./mimiciv/3.1"
    if not os.path.exists(mimic_path):
        print(f"WARNING: MIMIC-IV v3.1 data path not found: {mimic_path}")
        print("Please ensure MIMIC-IV v3.1 data is available at the expected location.")
        print("Expected structure:")
        print("  ./mimiciv/3.1/icu/icustays.csv.gz")
        print("  ./mimiciv/3.1/hosp/patients.csv.gz")
        print("  ./mimiciv/3.1/hosp/diagnoses_icd.csv.gz")
        print("  ... (other MIMIC-IV files)")
        print("")

    # Run main pipeline
    main()
