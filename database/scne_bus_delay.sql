DROP DATABASE IF EXISTS scne_bus_delay;

CREATE DATABASE scne_bus_delay;
USE scne_bus_delay;

CREATE TABLE routes (
    route_id BIGINT PRIMARY KEY,
    route_name VARCHAR(50) NOT NULL
);

CREATE TABLE service_dates (
    service_date DATE PRIMARY KEY,
    day_of_week INT,
    is_weekend INT,
    is_public_holiday INT
);

CREATE TABLE stops (
    stop_id VARCHAR(100) PRIMARY KEY
);

CREATE TABLE trips (
    trip_id VARCHAR(100),
    service_date DATE,
    route_id BIGINT,
    direction_id INT,
    PRIMARY KEY (trip_id, service_date),

    FOREIGN KEY (route_id)
        REFERENCES routes(route_id),

    FOREIGN KEY (service_date)
        REFERENCES service_dates(service_date)
);

CREATE TABLE journey_observations (
    observation_id BIGINT PRIMARY KEY,
    trip_id VARCHAR(100),
    service_date DATE,
    stop_id VARCHAR(100),
    stop_sequence INT,
    hour INT,
    minute INT,
    journey_progress DOUBLE,
    previous_stop_delay DOUBLE,
    rolling_previous_delay DOUBLE,
    has_previous_delay INT,
    delay_seconds DOUBLE,

    FOREIGN KEY (trip_id, service_date)
        REFERENCES trips(trip_id, service_date),

    FOREIGN KEY (stop_id)
        REFERENCES stops(stop_id)
);