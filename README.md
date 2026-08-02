# Multi-Day Bus Delay Prediction for Stagecoach North East Using PySpark

## Overview

This project predicts stop-level bus arrival delays for Stagecoach North East services using SIRI-VM vehicle-location data and GTFS timetable data.

The data pipeline was developed with PySpark and includes data extraction, cleaning, journey and stop matching, feature engineering, regression modelling, MySQL storage, and a Streamlit prediction interface.

The project uses data from **26, 27 and 28 December 2025** and produced **308,885 matched delay records**.

## Model Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 93.34 | 149.36 | 0.7110 |
| Decision Tree | 66.79 | 158.23 | 0.6756 |
| **Random Forest** | **61.95** | **137.57** | **0.7548** |
| Gradient-Boosted Trees | 66.55 | 152.51 | 0.6987 |

Random Forest gave the best overall performance and was used in the Streamlit application.

## Technologies

- Python 3.11
- PySpark 3.5.8
- Spark SQL and MLlib
- Pandas and SciPy
- MySQL and JDBC
- Streamlit
- Jupyter Notebook
- Parquet

## Repository Structure

```text
.
├── database/
│   └── scne_bus_delay.sql
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

The large `data/raw`, `data/interim`, and `data/processed` folders are kept locally and excluded from GitHub.

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
PySpark processing and analysis
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

## How to Run

### 1. Install the main packages

```bash
pip install pyspark==3.5.8 pandas numpy scipy matplotlib plotly streamlit mysql-connector-python jupyter
```

Java, MySQL, and MySQL Connector/J are also required.

### 2. Run the notebooks

Open the `notebooks` folder and run Notebooks `01` to `10` in order.

Update the local data, model, JDBC driver, and database paths where required.

Before running Notebook 10, create the database using:

```text
database/scne_bus_delay.sql
```

### 3. Run the Streamlit app

From the project root:

```bash
streamlit run ui/app.py
```

## Notes

- The project uses only three Christmas holiday-period dates.
- Arrival time is estimated from the fresh GPS observation nearest to a stop.
- Weather, traffic, incidents, roadworks, and passenger demand are not included.
- The application is an academic demonstration and is not connected to live bus data.
