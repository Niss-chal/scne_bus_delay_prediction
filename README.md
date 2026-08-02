# Multi-Day Bus Delay Prediction for Stagecoach North East Using PySpark

## Overview

This project predicts stop-level bus arrival delay for Stagecoach North East services. It combines SIRI-VM vehicle-location data with GTFS timetable data, processes the data using PySpark, compares regression models, stores the final data in MySQL, and provides predictions through a Streamlit interface.

The project uses data from 26, 27 and 28 December 2025. After cleaning and matching, the final dataset contains **308,885 delay records**.

## Main Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 93.34 | 149.36 | 0.7110 |
| Decision Tree | 66.79 | 158.23 | 0.6756 |
| Random Forest | **61.95** | **137.57** | **0.7548** |
| Gradient-Boosted Trees | 66.55 | 152.51 | 0.6987 |

Random Forest gave the best overall result and was used in the Streamlit application.

## Technologies

- Python 3.11
- PySpark 3.5.8 and Spark SQL
- PySpark MLlib
- Pandas and SciPy
- MySQL and JDBC
- Streamlit
- Jupyter Notebook
- Parquet

## Project Structure

```text
.
├── data/
│   ├── raw/          # Original SIRI-VM and GTFS files
│   ├── interim/      # Extracted and filtered daily files
│   └── processed/    # Delay data and Spark Parquet datasets
├── database/
│   └── scne_bus_delay.sql
├── drivers/
│   └── mysql-connector-j-9.7.0.jar
├── models/
│   ├── random_forest_delay_model/
│   └── route_indexer_model/
├── notebooks/
│   ├── 01_SCNE_MultiDay_SIRIVM_Extraction.ipynb
│   ├── 02_SCNE_MultiDay_GTFS_Timetable_Extraction.ipynb
│   ├── 03_SCNE_MultiDay_Data_Cleaning_and_Delay_Creation.ipynb
│   ├── 04_SCNE_PySpark_Data_Integration.ipynb
│   ├── 05_SCNE_PySpark_SQL_and_EDA.ipynb
│   ├── 06_SCNE_PySpark_Feature_Engineering.ipynb
│   ├── 07_SCNE_PySpark_Regression_Models.ipynb
│   ├── 08_SCNE_Model_Evaluation_and_Conclusion.ipynb
│   ├── 09_SCNE_User_Interface_Preparation.ipynb
│   └── 10_SCNE_MySQL_Database_Integration.ipynb
├── outputs/
│   ├── figures/
│   └── logs/
├── ui/
│   └── app.py
├── .gitignore
└── README.md
```

## Notebook Guide

| Notebook | Purpose |
|---|---|
| 01 | Extract multi-day Stagecoach North East SIRI-VM observations |
| 02 | Extract active GTFS timetable data for each service date |
| 03 | Clean data, match journeys and stops, and calculate delay |
| 04 | Combine daily data using PySpark and save it as Parquet |
| 05 | Run Spark SQL, data profiling and exploratory analysis |
| 06 | Create machine-learning features and prevent target leakage |
| 07 | Train and compare four regression models |
| 08 | Evaluate the final model in detail |
| 09 | Save the model and prepare inputs for the user interface |
| 10 | Write Spark DataFrames to MySQL and run a parameterised query |

## Workflow

```text
SIRI-VM and GTFS data
        ↓
Extraction and cleaning
        ↓
Journey and nearest-stop matching
        ↓
Delay calculation
        ↓
PySpark integration and analysis
        ↓
Feature engineering
        ↓
Regression model comparison
        ↓
Random Forest evaluation
        ↓
MySQL storage and Streamlit interface
```

The prediction target is:

```text
delay_seconds = observed vehicle time - scheduled arrival time
```

## Running the Project

### Setup

#### 1. Create the environment

```bash
conda create -n pyspark311 python=3.11
conda activate pyspark311
```

#### 2. Install Python packages

```bash
pip install pyspark==3.5.8 pandas numpy scipy matplotlib plotly streamlit mysql-connector-python jupyter
```

#### 3. Add MySQL Connector/J

Download the platform-independent MySQL Connector/J archive and place the `.jar` file in:

```text
drivers/mysql-connector-j-9.7.0.jar
```

#### 4. Create the MySQL database

Run:

```text
scne_bus_delay.sql
```

in MySQL Workbench before running Notebook 10.

#### 5. Configure local paths

The notebooks were developed on Windows and contain local paths. Update the data, model, driver and database paths before running them on another computer.

## Notes

- The project uses only three Christmas holiday-period dates.
- Arrival time is estimated from the fresh GPS observation nearest to a stop.
- Weather, traffic, incidents and passenger demand are not included.
- The application is an academic demonstration and is not connected to live bus data.
