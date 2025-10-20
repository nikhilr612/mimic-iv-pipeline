#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def drop_wrong_uom(data, cut_off):
    """
    Drop observations with wrong unit of measurement (UOM) based on frequency threshold.

    Args:
        data: DataFrame containing data with 'itemid' and 'valueuom' columns
        cut_off: Frequency threshold (0-1) for dominant UOM to trigger dropping

    Returns:
        DataFrame with non-dominant UOM observations removed
    """
    if data.empty:
        logger.warning("Input data is empty, returning empty DataFrame")
        return data

    if "itemid" not in data.columns or "valueuom" not in data.columns:
        logger.warning("Required columns 'itemid' or 'valueuom' not found in data")
        return data

    original_count = len(data)
    items_processed = 0
    items_with_multiple_uoms = 0
    items_with_drops = 0
    total_dropped = 0

    logger.info(
        f"Starting UOM cleaning on {original_count:,} records with cutoff {cut_off}"
    )

    grouped = data.groupby(["itemid"])["valueuom"]
    for id_number, uom in grouped:
        items_processed += 1
        value_counts = uom.value_counts()
        num_observations = len(uom)

        if value_counts.size > 1:
            items_with_multiple_uoms += 1
            most_frequent_measurement = value_counts.index[0]
            frequency = value_counts[0]
            frequency_ratio = frequency / num_observations

            logger.debug(
                f"Item {id_number}: {value_counts.size} different UOMs, "
                f"dominant UOM '{most_frequent_measurement}' appears in "
                f"{frequency}/{num_observations} ({frequency_ratio:.3f}) observations"
            )

            if frequency_ratio > cut_off:
                items_with_drops += 1
                values = uom
                index_to_drop = values[values != most_frequent_measurement].index
                records_to_drop = len(index_to_drop)
                total_dropped += records_to_drop

                logger.info(
                    f"Item {id_number}: Dropping {records_to_drop:,} records with "
                    f"non-dominant UOMs (keeping '{most_frequent_measurement}', "
                    f"dropping {value_counts.size - 1} other UOM types)"
                )

                # Log which UOMs are being dropped
                dropped_uoms = value_counts.drop(most_frequent_measurement)
                for uom_name, count in dropped_uoms.items():
                    logger.debug(
                        f"  - Dropping {count:,} records with UOM '{uom_name}'"
                    )

                data.drop(index_to_drop, axis=0, inplace=True)
            else:
                logger.debug(
                    f"Item {id_number}: Keeping all UOMs (dominant UOM frequency "
                    f"{frequency_ratio:.3f} <= threshold {cut_off})"
                )

    data = data.reset_index(drop=True)
    final_count = len(data)

    logger.info(f"UOM cleaning completed:")
    logger.info(f"  - Items processed: {items_processed:,}")
    logger.info(f"  - Items with multiple UOMs: {items_with_multiple_uoms:,}")
    logger.info(f"  - Items with records dropped: {items_with_drops:,}")
    logger.info(f"  - Total records dropped: {total_dropped:,}")
    logger.info(f"  - Records remaining: {final_count:,} (from {original_count:,})")
    logger.info(f"  - Drop percentage: {(total_dropped/original_count)*100:.2f}%")

    return data
