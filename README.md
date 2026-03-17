# Sensor Validation Tools (3D-PAWS)

## Overview

This repository contains a suite of Python tools designed to convert raw instrument log outputs into a standardized CSV format and generate a variety of analytical plots. Supported visualizations include time-series, box-and-whisker, difference, and scatter plots, enabling systematic validation and comparison of sensor performance.

---

## File Summaries

### 1. `nesting_log_file_reformatter.py`

**Primary Function**  
Acts as the data ingestion tool that converts raw instrument logs into the project’s standard `formattedcsv` layout.

**Key Features**
- Contains specialized parsers for different vendor and reference formats
- Handles LabVIEW timestamp conversions to Unix epoch
- Resamples data into regular time intervals (e.g., 0.5-minute averages) for consistency across different sensors

**Output**
- Standardized CSV files saved in `data/formattedcsv/`

---

### 2. `graphMerger.py`

**Primary Function**  
Generates lightweight time-series plots from the formatted CSV files.

**Key Features**
- Identifies valid sensors and reference series based on naming conventions
- Automatically computes sensible y-axis limits and tick intervals, with support for manual axis presets for specific instruments (e.g., humidity or pressure)
- Defensively handles data by ignoring non-numeric values and aligning series lengths before plotting

**Output**
- PNG time-series images saved under `timeseriesPlots/`

---

### 3. `differencePlotter.py`

**Primary Function**  
Produces plots showing the difference between sensor values and a designated reference sensor (Sensor − Reference).

**Key Features**
- Includes special handling for humidity data, such as spike detection, masking, and interpolation to stabilize plateaus
- Detects hysteresis periods using configurable criteria (e.g., minimum duration and RH threshold)
- Exports per-sensor precision metrics alongside the plots

**Output**
- Difference plot PNGs and precision CSV files saved under `differencePlots/`

---

### 4. `boxwhiskerMerger.py`

**Primary Function**  
Generates box-and-whisker plots to visualize the distribution of sensor values for each folder of formatted CSVs.

**Key Features**
- Groups sensors by type based on the first three characters of their column names
- Provides boxed textual annotations for key statistics, including Min, Max, IQR, Q1, Median, and Mean
- Ensures consistent visual scaling across plots by applying a calculated `base_fontsize` to all labels and titles

**Output**
- PNG box-and-whisker plots saved under `boxPlots/`

---

### 5. `scatterMerger.py`

**Primary Function**  
Builds pairwise scatter plots for all sensor combinations within a CSV to analyze correlations.

**Key Features**
- Automatically identifies sensor pairs with sufficient overlapping data (minimum of three points) to generate valid plots
- Annotates reference sensors in plot labels and prefixes their filenames with `ref_`
- Uses a single-character suffix (e.g., `t`, `p`, `r`) to filter allowed sensors based on instrument type

**Output**
- PNG scatter plot files saved under `scatterPlots/`
