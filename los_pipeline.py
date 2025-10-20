#!/usr/bin/env python3
"""
MIMIC-IV Length of Stay (LOS) Prediction Pipeline - Version 3 ICU Only
Single script that replicates the mainPipeline.ipynb functionality for MIMIC-IV v3 ICU data
"""

import os
import sys
import logging
from datetime import datetime

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
    "procedures": False,
    "medications": False,
    "output_events": True,  # ICU specific
    "chart_events": True,  # ICU specific
}

# Database Configuration
USE_DATABASE_MODE = True  # Use DuckDB-based preprocessing with CSV.gz cohort files

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

# Item ID Lists for Feature Selection
CHART_ITEMIDS_FILE = os.path.join(ROOT_DIR, "utils", "chart_itemids.txt")
OUTPUT_ITEMIDS_FILE = os.path.join(ROOT_DIR, "utils", "output_itemids.txt")
MED_ITEMIDS_FILE = os.path.join(ROOT_DIR, "utils", "med_itemids.txt")
PROC_ITEMIDS_FILE = os.path.join(ROOT_DIR, "utils", "proc_itemids.txt")

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================


def setup_logging():
    """Setup logging configuration for the pipeline"""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(ROOT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"los_pipeline_{timestamp}.log")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


# Initialize logger
logger = setup_logging()

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
        logger.info("Successfully imported day_intervals_cohort_v3")
    except ImportError as e:
        logger.error(f"Failed to import day_intervals_cohort_v3: {e}")
        logger.error("Please ensure you're running from the project root directory")
        sys.exit(1)

    # Import feature selection ICU based on configuration
    if USE_DATABASE_MODE:
        try:
            import feature_selection_icu_db

            modules["feature_icu"] = feature_selection_icu_db
            logger.info(
                "Successfully imported feature_selection_icu_db (database mode)"
            )
        except ImportError as e:
            logger.warning(f"Failed to import feature_selection_icu_db: {e}")
            logger.info("Falling back to original feature_selection_icu...")
            try:
                import feature_selection_icu

                modules["feature_icu"] = feature_selection_icu
                logger.info("Successfully imported feature_selection_icu (fallback)")
            except ImportError as e2:
                logger.error(f"Failed to import both versions: {e}, {e2}")
                sys.exit(1)
    else:
        try:
            import feature_selection_icu

            modules["feature_icu"] = feature_selection_icu
            logger.info("Successfully imported feature_selection_icu (file mode)")
        except ImportError as e:
            logger.error(f"Failed to import feature_selection_icu: {e}")
            sys.exit(1)

    # Import data generation ICU
    try:
        import data_generation_icu

        modules["data_gen_icu"] = data_generation_icu
        logger.info("Successfully imported data_generation_icu")
    except ImportError as e:
        logger.error(f"Failed to import data_generation_icu: {e}")
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


def check_itemids_file(file_path):
    """Check if itemids file exists and has content"""
    if not os.path.exists(file_path):
        return False

    try:
        with open(file_path, "r") as f:
            content = f.read().strip()
            # Check if file has non-comment, non-empty lines
            lines = [
                line.strip()
                for line in content.split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]
            return len(lines) > 0
    except Exception as e:
        logger.warning(f"Error reading {file_path}: {e}")
        return False


def check_feature_availability():
    """Check which features are available based on itemids files"""
    features_available = {
        "chart_events": check_itemids_file(CHART_ITEMIDS_FILE),
        "output_events": check_itemids_file(OUTPUT_ITEMIDS_FILE),
        "medications": check_itemids_file(MED_ITEMIDS_FILE),
        "procedures": check_itemids_file(PROC_ITEMIDS_FILE),
    }

    logger.info("Feature availability check:")
    for feature, available in features_available.items():
        status = (
            "AVAILABLE" if available else "UNAVAILABLE (empty/missing itemids file)"
        )
        logger.info(f"  - {feature}: {status}")

    return features_available


# =============================================================================
# MAIN PIPELINE FUNCTIONS
# =============================================================================


def extract_cohort():
    """Extract cohort data using MIMIC-IV v3 ICU data"""
    logger.info("=" * 60)
    logger.info("STEP 1: COHORT EXTRACTION")
    logger.info("=" * 60)

    # Parse parameters
    los_threshold = parse_los_threshold()
    icd_code = get_disease_icd_code()

    logger.info(f"Configuration:")
    logger.info(f"  - Version: {VERSION} (MIMIC-IV v3.1)")
    logger.info(f"  - Data Source: {DATA_SOURCE}")
    logger.info(f"  - LOS Threshold: {los_threshold} days")
    logger.info(f"  - Disease Filter: {DISEASE_FILTER}")
    logger.info(f"  - Diagnoses: Always included for ICU data")

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
        logger.info("Cohort extraction completed successfully")
        logger.info(f"Cohort saved as: {cohort_output}")
        if USE_DATABASE_MODE:
            return cohort_output, conn
        else:
            conn.close()
            return cohort_output
    except Exception as e:
        logger.error(f"Error during cohort extraction: {e}")
        raise


def feature_selection(cohort_output, conn=None):
    """Perform ICU feature selection with diagnoses included"""
    logger.info("=" * 60)
    logger.info("STEP 2: ICU FEATURE SELECTION")
    logger.info("=" * 60)

    version_path = get_version_path()

    # Check feature availability
    features_available = check_feature_availability()

    # Update FEATURE_FLAGS based on availability
    updated_flags = FEATURE_FLAGS.copy()
    if not features_available["chart_events"]:
        updated_flags["chart_events"] = False
        logger.warning("Disabling chart events extraction - no itemids available")

    if not features_available["output_events"]:
        updated_flags["output_events"] = False
        logger.warning("Disabling output events extraction - no itemids available")

    if not features_available["medications"]:
        updated_flags["medications"] = False
        logger.warning("Disabling medications extraction - no itemids available")

    if not features_available["procedures"]:
        updated_flags["procedures"] = False
        logger.warning("Disabling procedures extraction - no itemids available")

    logger.info(f"Processing features for ICU data from {version_path}...")
    logger.info(f"Selected features: {[k for k, v in updated_flags.items() if v]}")
    logger.info("Diagnoses: ENABLED (required for ICU)")

    try:
        mode_str = "database" if USE_DATABASE_MODE else "file"
        logger.info(f"Running feature extraction in {mode_str} mode...")
        if USE_DATABASE_MODE:
            logger.info("Using DuckDB with CSV.gz cohort files for preprocessing")

        # ICU feature selection with diagnoses always enabled
        if USE_DATABASE_MODE and conn:
            # Use database-enabled feature extraction
            MODULES["feature_icu"].feature_icu(
                conn=conn,
                cohort_output=cohort_output,
                diag_flag=True,  # Always True for diagnoses
                proc_flag=updated_flags["procedures"],
                out_flag=updated_flags["output_events"],
                chart_flag=updated_flags["chart_events"],
                med_flag=updated_flags["medications"],
            )
        else:
            # Use original file-based feature extraction
            MODULES["feature_icu"].feature_icu(
                cohort_output=cohort_output,
                version_path=version_path,
                diag_flag=True,  # Always True for diagnoses
                proc_flag=updated_flags["procedures"],
                out_flag=updated_flags["output_events"],
                chart_flag=updated_flags["chart_events"],
                med_flag=updated_flags["medications"],
            )
        logger.info("Feature selection completed successfully")
    except Exception as e:
        logger.error(f"Feature selection failed: {e}")
        logger.error(f"Error during feature selection: {e}")
        raise


def generate_time_series(cohort_output):
    """Generate ICU time series data"""
    logger.info("=" * 60)
    logger.info("STEP 3: ICU TIME SERIES GENERATION")
    logger.info("=" * 60)

    # Parse time series parameters
    include_hours = parse_time_window()
    bucket_size = parse_bucket_size()
    impute_method = parse_imputation()

    # Set task flags
    data_los = True
    data_mort = False
    data_admn = False

    logger.info(f"Time series configuration:")
    logger.info(f"  - Include: First {include_hours} hours")
    logger.info(f"  - Bucket size: {bucket_size} hour(s)")
    logger.info(f"  - Imputation: {impute_method}")
    logger.info("  - Task: Length of Stay prediction")

    try:
        # Check feature availability for time series generation
        features_available = check_feature_availability()

        # Update flags for time series generation
        ts_flags = {
            "procedures": FEATURE_FLAGS["procedures"]
            and features_available["procedures"],
            "output_events": FEATURE_FLAGS["output_events"]
            and features_available["output_events"],
            "chart_events": FEATURE_FLAGS["chart_events"]
            and features_available["chart_events"],
            "medications": FEATURE_FLAGS["medications"]
            and features_available["medications"],
        }

        logger.info(
            f"Time series generation using features: {[k for k, v in ts_flags.items() if v]}"
        )

        # Create ICU time series generator
        gen = MODULES["data_gen_icu"].Generator(
            cohort_output=cohort_output,
            if_mort=data_mort,
            if_admn=data_admn,
            if_los=data_los,
            feat_cond=True,  # Always True for diagnoses
            feat_proc=ts_flags["procedures"],
            feat_out=ts_flags["output_events"],
            feat_chart=ts_flags["chart_events"],
            feat_med=ts_flags["medications"],
            impute=impute_method,
            include_time=include_hours,
            bucket=bucket_size,
            predW=0,  # No prediction window for LOS
        )
        logger.info("ICU time series generation completed successfully.")
        return gen
    except Exception as e:
        logger.error(f"Error during time series generation: {e}")
        raise


def generate_summary(cohort_output):
    """Generate data summary and statistics"""
    logger.info("=" * 60)
    logger.info("STEP 4: DATA SUMMARY")
    logger.info("=" * 60)

    # Load cohort data
    cohort_path = f"./data/cohort/{cohort_output}.csv.gz"
    if os.path.exists(cohort_path):
        try:
            cohort_df = pd.read_csv(cohort_path, compression="gzip")

            logger.info("ICU Cohort Statistics:")
            logger.info(f"  - Total patients: {cohort_df['subject_id'].nunique()}")
            logger.info(f"  - Total ICU stays: {len(cohort_df)}")
            logger.info(
                f"  - Positive cases (LOS > threshold): {(cohort_df['label'] == 1).sum()}"
            )
            logger.info(
                f"  - Negative cases (LOS <= threshold): {(cohort_df['label'] == 0).sum()}"
            )
            logger.info(
                f"  - Label distribution: {cohort_df['label'].value_counts().to_dict()}"
            )

            if "Age" in cohort_df.columns:
                logger.info(f"  - Mean age: {cohort_df['Age'].mean():.1f}")
                logger.info(
                    f"  - Age range: {cohort_df['Age'].min()}-{cohort_df['Age'].max()}"
                )

            if "gender" in cohort_df.columns:
                logger.info(
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
            logger.info(f"Summary saved to: {summary_path}")

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
    else:
        logger.warning(f"Warning: Cohort file not found at {cohort_path}")

    logger.info("Data processing completed successfully!")
    logger.info("Ready for model training with your preferred ML framework.")


def main():
    """Main pipeline execution"""
    logger.info("MIMIC-IV v3 ICU Length of Stay Prediction Pipeline")
    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info(f"  Version: {VERSION} (MIMIC-IV v3.1)")
    logger.info(f"  Data Source: {DATA_SOURCE}")
    logger.info(f"  LOS Threshold: {LOS_THRESHOLD_TYPE}")
    logger.info(f"  Disease Filter: {DISEASE_FILTER}")
    logger.info(f"  Features: {[k for k, v in FEATURE_FLAGS.items() if v]}")
    logger.info(f"  Diagnoses: ALWAYS INCLUDED")
    logger.info(f"  Time Window: {TIME_WINDOW_TYPE}")
    logger.info(f"  Bucket Size: {BUCKET_SIZE_TYPE}")
    logger.info(f"  Imputation: {IMPUTATION_METHOD}")
    logger.info("=" * 60)

    try:
        # Step 1: Extract ICU cohort
        if USE_DATABASE_MODE:
            cohort_output, conn = extract_cohort()
        else:
            cohort_output = extract_cohort()
            conn = None

        try:
            # Step 2: ICU feature selection (with diagnoses)
            feature_selection(cohort_output, conn)

            # Step 3: Generate ICU time series
            generate_time_series(cohort_output)

            # Step 4: Generate summary
            generate_summary(cohort_output)
        finally:
            # Clean up database connection if it exists
            if conn and USE_DATABASE_MODE:
                conn.close()
                logger.info("Database connection closed")

        logger.info("=" * 60)
        logger.info("ICU DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("Output files:")
        logger.info(f"  - Cohort: ./data/cohort/{cohort_output}.csv.gz")
        logger.info("  - Features: ./data/features/preproc_*.csv.gz")
        logger.info("  - Time series: ./data/dict/ and ./data/csv/")
        logger.info(f"  - Summary: ./data/summary/{cohort_output}_summary.txt")
        logger.info("=" * 60)
        logger.info("Data includes:")
        logger.info("  - ICU stays from MIMIC-IV v3.1")
        logger.info("  - Diagnosis codes (ICD-9/ICD-10)")
        logger.info("  - Procedures, medications, chart events, output events")
        logger.info("  - Time-series features with configurable bucketing")
        logger.info("=" * 60)
        logger.info("Next steps:")
        logger.info("  - Use ./data/csv/labels.csv for training labels")
        logger.info("  - Use ./data/csv/{patient_id}/ files for patient features")
        logger.info("  - Load ./data/dict/ files for vocabulary and metadata")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"ERROR: Pipeline failed with exception: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
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
        logger.warning(f"MIMIC-IV v3.1 data path not found: {mimic_path}")
        logger.warning(
            "Please ensure MIMIC-IV v3.1 data is available at the expected location."
        )
        logger.warning("Expected structure:")
        logger.warning("  ./mimiciv/3.1/icu/icustays.csv.gz")
        logger.warning("  ./mimiciv/3.1/hosp/patients.csv.gz")
        logger.warning("  ./mimiciv/3.1/hosp/diagnoses_icd.csv.gz")
        logger.warning("  ... (other MIMIC-IV files)")
        logger.warning("")

    # Run main pipeline
    main()
