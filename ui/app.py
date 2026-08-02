import streamlit as st
import pandas as pd
import zipfile

from pyspark.sql import SparkSession, Row
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.ml.feature import StringIndexerModel, VectorAssembler


# PAGE SETUP
st.set_page_config(
    page_title="SCNE Bus Delay Predictor",
    page_icon="🚌",
    layout="wide"
)

st.title("🚌 SCNE Bus Delay Predictor")

st.caption(
    "Stagecoach North East | PySpark Random Forest Regression"
)

st.write(
    "Enter journey information below to estimate the expected delay "
    "for a bus service."
)


# LOAD SPARK + SAVED MODEL
@st.cache_resource
def load_resources():

    spark = (
        SparkSession.builder
        .appName("SCNE Streamlit Prediction")
        .master("local[*]")
        .getOrCreate()
    )

    model_path = (
        r"D:\Big Data Programming Project\Final Assignment"
        r"\models\random_forest_delay_model"
    )

    encoder_path = (
        r"D:\Big Data Programming Project\Final Assignment"
        r"\models\route_indexer_model"
    )

    model = RandomForestRegressionModel.load(model_path)
    encoder = StringIndexerModel.load(encoder_path)

    return spark, model, encoder


spark, model, encoder = load_resources()

route_options = sorted(list(encoder.labels))


# LOAD ROUTE DESTINATIONS FROM GTFS
GTFS_PATH = (
    r"D:\Big Data Programming Project\Final Assignment"
    r"\data\raw\timetable\2025-12-28"
    r"\itm_north_east_gtfs_20251228.zip"
)

with zipfile.ZipFile(GTFS_PATH, "r") as gtfs_zip:

    with gtfs_zip.open("routes.txt") as file:
        routes_df = pd.read_csv(file)

    with gtfs_zip.open("trips.txt") as file:
        trips_df = pd.read_csv(file)


routes_df["route_short_name"] = (
    routes_df["route_short_name"]
    .astype(str)
    .str.strip()
)

route_trip_df = trips_df.merge(
    routes_df[["route_id", "route_short_name"]],
    on="route_id",
    how="left"
)

route_destination_map = {}

for route_code in route_options:

    for direction_value in [0, 1]:

        matches = route_trip_df[
            (route_trip_df["route_short_name"] == str(route_code).strip()) &
            (route_trip_df["direction_id"] == direction_value)
        ]

        if not matches.empty:

            headsigns = (
                matches["trip_headsign"]
                .dropna()
                .astype(str)
                .str.strip()
            )

            if len(headsigns) > 0:
                destination = headsigns.value_counts().index[0]
                route_destination_map[(route_code, direction_value)] = destination

# MODEL INFORMATION
st.subheader("Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("MAE", "61.95 sec")
col2.metric("RMSE", "137.57 sec")
col3.metric("R²", "0.7548")

st.caption(
    "Random Forest was selected because it achieved the lowest MAE and RMSE "
    "and the highest R² among the four regression models tested."
)

st.divider()


# JOURNEY DETAILS
st.subheader("Journey Details")

left, right = st.columns(2)

with left:

    direction = st.selectbox(
        "Direction",
        [0, 1],
        format_func=lambda x: "First Direction" if x == 0 else "Return Direction"
    )

    available_routes = [
        route_code
        for route_code in route_options
        if (route_code, direction) in route_destination_map
    ]

    route = st.selectbox(
        "Bus Route",
        available_routes,
        format_func=lambda x: (
            f"{x} — {route_destination_map[(x, direction)]}"
        )
    )

    stop_sequence = st.number_input(
        "Stop Sequence",
        min_value=0,
        max_value=100,
        value=10,
        step=1
    )

    journey_progress = st.slider(
        "Journey Progress",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.01
    )

with right:

    hour = st.slider(
        "Scheduled Hour",
        min_value=0,
        max_value=23,
        value=18
    )

    minute = st.slider(
        "Scheduled Minute",
        min_value=0,
        max_value=59,
        value=30
    )

    day_of_week = st.selectbox(
        "Day of Week",
        [
            ("Sunday", 1),
            ("Monday", 2),
            ("Tuesday", 3),
            ("Wednesday", 4),
            ("Thursday", 5),
            ("Friday", 6),
            ("Saturday", 7)
        ],
        format_func=lambda x: x[0]
    )[1]

    public_holiday = st.checkbox(
        "Public Holiday"
    )


# DERIVED VALUES
is_weekend = 1 if day_of_week in [1, 7] else 0
is_public_holiday = 1 if public_holiday else 0


# RECENT JOURNEY DELAY INFORMATION
st.divider()

st.subheader("Recent Journey Delay Information")

st.caption(
    "Use this section only when delay information from earlier stops "
    "in the same journey is available."
)

history_available = st.checkbox(
    "Previous journey delay information available"
)

if history_available:

    history_col1, history_col2 = st.columns(2)

    with history_col1:

        previous_stop_delay = st.number_input(
            "Previous Stop Delay (seconds)",
            min_value=-900.0,
            max_value=1800.0,
            value=0.0,
            step=10.0,
            help="Delay recorded at the immediately previous stop."
        )

    with history_col2:

        rolling_previous_delay = st.number_input(
            "Average Delay of Previous Stops (seconds)",
            min_value=-900.0,
            max_value=1800.0,
            value=0.0,
            step=10.0,
            help="Average delay from the most recent previous stops."
        )

    has_previous_delay = 1

else:

    previous_stop_delay = 0.0
    rolling_previous_delay = 0.0
    has_previous_delay = 0

    st.info(
        "No recent delay history will be used. "
        "The prediction will rely on route, time, stop and journey information."
    )


# PREDICTION
st.divider()

if st.button(
    "Predict Bus Delay",
    type="primary",
    use_container_width=True
):

    input_row = Row(
        published_line_name=route,
        direction_id=int(direction),
        stop_sequence=int(stop_sequence),
        hour=int(hour),
        minute=int(minute),
        day_of_week=int(day_of_week),
        is_weekend=int(is_weekend),
        is_public_holiday=int(is_public_holiday),
        journey_progress=float(journey_progress),
        previous_stop_delay=float(previous_stop_delay),
        rolling_previous_delay=float(rolling_previous_delay),
        has_previous_delay=int(has_previous_delay)
    )

    input_df = spark.createDataFrame([input_row])

    indexed_df = encoder.transform(input_df)

    feature_columns = [
        "route_index",
        "direction_id",
        "stop_sequence",
        "hour",
        "minute",
        "day_of_week",
        "is_weekend",
        "is_public_holiday",
        "journey_progress",
        "previous_stop_delay",
        "rolling_previous_delay",
        "has_previous_delay"
    ]

    assembler = VectorAssembler(
        inputCols=feature_columns,
        outputCol="tree_features"
    )

    ready_df = assembler.transform(indexed_df)

    prediction_df = model.transform(ready_df)

    prediction = prediction_df.select("prediction").first()[0]

    prediction_minutes = prediction / 60


    # RESULT STATUS
    if prediction < -60:

        status = "Early"
        status_icon = "🟢"
        message = "The service is predicted to arrive early."

    elif prediction <= 300:

        status = "On Time"
        status_icon = "✅"
        message = (
            "The service is predicted to operate "
            "within the normal delay range."
        )

    else:

        status = "Delayed"
        status_icon = "⚠️"
        message = (
            "The service is predicted to experience "
            "a noticeable delay."
        )


    # DISPLAY RESULT
    st.subheader("Prediction Result")

    result_col1, result_col2, result_col3 = st.columns(3)

    result_col1.metric(
        "Predicted Delay",
        f"{prediction:.0f} sec"
    )

    result_col2.metric(
        "Delay in Minutes",
        f"{prediction_minutes:.2f} min"
    )

    result_col3.metric(
        "Service Status",
        f"{status_icon} {status}"
    )

    st.info(message)


# ABOUT SECTION
st.divider()

with st.expander("About this prediction system"):

    st.markdown(
        """
        ### Project Overview

        This application predicts bus delay for Stagecoach North East services
        using a PySpark Random Forest regression model.

        The project combines:

        - SIRI-VM vehicle location data
        - GTFS timetable data
        - multi-day delay observations
        - engineered journey and delay-history features

        ### Data Used

        The model was developed using data from:

        - 26 December 2025
        - 27 December 2025
        - 28 December 2025

        Delay was calculated as the difference between the observed bus arrival
        time and the scheduled GTFS arrival time.

        ### Machine Learning Models

        Four regression models were evaluated:

        - Linear Regression
        - Decision Tree Regressor
        - Random Forest Regressor
        - Gradient-Boosted Tree Regressor

        Random Forest produced the best overall test performance and was selected
        for the final prediction system.

        ### Final Model Performance

        - MAE: 61.95 seconds
        - RMSE: 137.57 seconds
        - R²: 0.7548

        ### Prediction Inputs

        The prediction uses information such as route, direction, stop position,
        scheduled time, day of week, journey progress and recent delay information.

        The application is intended as a demonstration of how PySpark, public
        transport data and machine learning can be combined to support bus delay
        prediction.
        """
    )