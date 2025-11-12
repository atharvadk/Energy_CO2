CREATE DATABASE IF NOT EXISTS energy_project;

USE energy_project;

CREATE TABLE energy_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    country VARCHAR(100),
    year INT,
    energy_consumption FLOAT,
    co2_emission FLOAT
);
