"""Various utilities"""

from __future__ import annotations

import os
import re

from datetime import datetime, timedelta

import geoutils as gu
import numpy as np
import xdem


def get_satellite_type(dem_path):
    """Parse the satellite type from the filename"""
    basename = os.path.basename(dem_path)
    if re.match("DEM\S*", basename):
        sat_type = "ASTER"
    elif re.match("\S*dem_mcf\S*", basename) is not None:
        sat_type = "TDX"
    else:
        raise ValueError("Could not identify satellite type")
    return sat_type


def get_aster_date(fname) -> datetime:
    """Parse the date of an ASTER DEM from the filename"""
    # Extract string containing decimal year
    basename = os.path.basename(fname)
    year_decimal = float(basename[4:17])

    # Get integer year and decimals
    year = int(np.trunc(year_decimal))
    decimals = year_decimal - year

    # Convert to date and time
    base = datetime(year, 1, 1)
    ndays = base.replace(year=base.year + 1) - base
    return base + timedelta(seconds=ndays.total_seconds() * decimals)


def get_tdx_date(fname: str) -> datetime:
    """Parse the date of a TDX DEM from the filename"""
    # Extract string containing date and time
    basename = os.path.basename(fname)
    datetime_str = basename[:17]

    # Convert to datetime
    return datetime.strptime(datetime_str, "%Y-%m-%d_%H%M%S")


def get_dems_date(dem_path_list: list[str]) -> list:
    """
    Returns a list of dates from a list of DEM paths.

    :param dem_path_list: List of path to DEMs

    :returns: The list of dates in datetime format
    """
    dates = []

    for dem_path in dem_path_list:
        basename = os.path.basename(dem_path)
        sat_type = get_satellite_type(dem_path)

        # Get date
        if sat_type == "ASTER":
            dates.append(get_aster_date(dem_path))
        elif sat_type == "TDX":
            dates.append(get_tdx_date(dem_path))

    return np.asarray(dates)


def select_dems_by_date(dem_path_list: list[str], date1: str, date2: str, sat_type: str) -> list:
    """
    Returns the list of files which date falls within date1 and date 2 (included)

    :param dem_path_list: List of path to DEMs
    :param date1: Start date in ISO format YYYY-MM-DD
    :param date1: End date in ISO format YYYY-MM-DD
    :param sat_type: Either 'ASTER' or 'TDX'

    :returns: The list of indexes that match the criteria
    """
    if sat_type == "ASTER":
        dates = np.asarray([get_aster_date(dem_file) for dem_file in dem_path_list])
    elif sat_type == "TDX":
        dates = np.asarray([get_tdx_date(dem_file) for dem_file in dem_path_list])
    else:
        raise ValueError("sat_type must be 'ASTER' or 'TDX'")

    date1 = datetime.fromisoformat(date1)
    date2 = datetime.fromisoformat(date2)
    return np.where((date1 <= dates) & (dates <= date2))[0]


def dems_selection(
        dem_path_list: list[str],
        mode: str = None,
        validation_dates: list[str] = None,
        dt: float = -1,
        months: list[int] = np.arange(12) + 1
) -> list[list[str]]:
    """
    Return a list of lists of DEMs path that fit the selection.
    Selection mode include None or 'temporal'. If None, return all DEMs. If 'temporal' is set, 
 `dt`, 'validation_dates` and optionally `months` must be set.

    :param dem_path_list: List containing path to all DEMs to be considered
    :param mode" Any of None or "temporal"
    :param validation_dates: List of validation dates for the experiment, dates expressed as 'yyyy-mm-dd'
    :param dt: Number of days allowed around each validation date
    :param months: A list of months to be selected (numbered 1 to 12). Default is all months.

    :returns: List of same length as validation dates, containing lists of DEM paths for each validation date.
    """
    if mode is None:
        print(f"Found {len(dem_path_list)} DEMs")
        return [dem_path_list]

    elif mode == "temporal":
        # check that optional arguments are set
        assert validation_dates is not None, "`validation_dates` must be set"
        assert dt >= 0, "dt must be set to >= 0 value"

        # Get input DEM dates
        dems_dates = get_dems_date(dem_path_list)
        dems_months = np.asarray([date.month for date in dems_dates])

        # Compare to each validation date
        output_list = []
        for date_str in validation_dates:
            date = datetime.fromisoformat(date_str)
            date1 = date - timedelta(dt)
            date2 = date + timedelta(dt)
            matching_dates = np.where((date1 <= dems_dates) & (dems_dates <= date2) & np.isin(dems_months, months))[0]
            output_list.append(dem_path_list[matching_dates])

        for date, group in zip(validation_dates, output_list):
            print(f"For date {date} found {len(group)} DEMs")

        return output_list
    else:
        raise ValueError(f"Mode {mode} not recognized")


def load_ref_and_masks(case_paths: dict) -> list:
    """
    Loads the reference xdem, outlines and masks of ROI and stable terrin, from the dictionary provided by files.get_data_paths.

    :returns:
    - ref_dem (xdem.DEM object), all_outlines (gu.Vector object), roi_outlines (gu.Vector object), roi_mask (np.ndarray, stable_mask (np.ndarray)
    """
    # Load reference DEM
    ref_dem = xdem.DEM(case_paths["raw_data"]["ref_dem_path"])

    # Load all outlines
    all_outlines = gu.geovector.Vector(case_paths["raw_data"]["rgi_path"])

    # Load selected glacier outline
    roi_outlines = gu.geovector.Vector(case_paths["raw_data"]["selected_path"])

    # Create masks
    roi_mask = roi_outlines.create_mask(ref_dem)
    stable_mask = ~all_outlines.create_mask(ref_dem)

    return ref_dem, all_outlines, roi_outlines, roi_mask, stable_mask
