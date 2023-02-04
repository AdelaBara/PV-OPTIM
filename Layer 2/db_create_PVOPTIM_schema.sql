CREATE SCHEMA `pv_optim` ;

CREATE TABLE pv_optim.`T_APPLIANCE_LOGS` (
  `ID_APPLIANCE` varchar(10) NOT NULL,
  `TIMESTAMP_R` datetime DEFAULT NULL,
  `COMMAND` varchar(5) DEFAULT NULL,
  `PT` decimal(10,0) DEFAULT NULL,
  `PF` decimal(10,0) DEFAULT NULL,
  `SOC` decimal(10,0) DEFAULT NULL,
  `PG` decimal(10,0) DEFAULT NULL,
  `P_APP` decimal(10,0) DEFAULT NULL
) ;

CREATE TABLE pv_optim.`T_APPLIANCE_OPTIMAL_SCHEDULE` (
  `ID_APPLIANCE` varchar(20) NOT NULL,
  `START_TIME` timestamp NULL DEFAULT NULL,
  `END_TIME` timestamp NULL DEFAULT NULL,
  `PMAX` decimal(10,0) DEFAULT NULL,
  `MAX_END_TIME` timestamp NULL DEFAULT NULL,
  `ACTIVE` varchar(10) DEFAULT NULL
) ;

CREATE TABLE pv_optim.`T_APPLIANCE_READINGS` (
  `ID_APPLIANCE` varchar(128) DEFAULT NULL,
  `STATUS` varchar(10) DEFAULT NULL,
  `READING_DATE` timestamp NULL DEFAULT NULL,
  `VOLTAGE_MV` decimal(13,2) DEFAULT NULL,
  `CURRENT_MA` decimal(13,2) DEFAULT NULL,
  `ACTIVE_POWER_CONS` decimal(13,2) DEFAULT NULL,
  `TOTAL_WH` decimal(13,2) DEFAULT NULL,
  `ERR_CODE1` decimal(10,0) DEFAULT NULL
) ;

CREATE TABLE pv_optim.`T_APPLIANCE_SCHEDULE` (
  `ID_APPLIANCE` varchar(20) NOT NULL,
  `OPERATION` decimal(2,0) DEFAULT NULL,
  `START_TIME` timestamp NULL DEFAULT NULL,
  `END_TIME` timestamp NULL DEFAULT NULL,
  `PRIORITY` decimal(10,0) DEFAULT NULL,
  `PMAX` decimal(10,0) DEFAULT NULL,
  `NO_OPERATIONS` decimal(2,0) DEFAULT NULL,
  `DURATION` decimal(5,0) DEFAULT NULL
) ;

CREATE TABLE pv_optim.`T_APPLIANCES` (
  `ID_APPLIANCE` varchar(128) NOT NULL,
  `NAME` varchar(50) DEFAULT NULL,
  `DEVICE_TYPE` varchar(50) DEFAULT NULL,
  `DESCRIPTION` varchar(100) DEFAULT NULL,
  `START_DATE` date DEFAULT NULL,
  `STATUS` varchar(20) DEFAULT NULL,
  `ID_CONSUMER_PLACE` varchar(20) DEFAULT NULL,
  `REQUIRED_OPERATION_TIME` decimal(10,0) DEFAULT NULL,
  `TIME_BETWEEN_OPER` decimal(10,0) DEFAULT NULL,
  `ACTIONABLE` varchar(50) DEFAULT NULL,
  `CAPACITY` decimal(10,0) DEFAULT NULL,
  `POWER_MAX` decimal(10,0) DEFAULT NULL,
  `POWER_MIN` decimal(10,0) DEFAULT NULL,
  `SOC_MAX` decimal(10,0) DEFAULT NULL,
  `SOC_MIN` decimal(10,0) DEFAULT NULL,
  `FLEX_TYPE` varchar(10) DEFAULT NULL,
  `MAX_OPER_TIME` decimal(5,0) DEFAULT NULL,
  `PRIORITY` decimal(3,0) DEFAULT NULL
) ;

CREATE TABLE pv_optim.`T_CONSUMER_TARIFFS` (
  `ID_CONSUMER_PLACE` varchar(100) DEFAULT NULL,
  `CONS_PRICE` decimal(5,2) DEFAULT NULL,
  `START_HOUR` decimal(2,0) DEFAULT NULL,
  `END_HOUR` decimal(2,0) DEFAULT NULL
) ;

CREATE TABLE pv_optim.`T_INVERTER_FORECAST` (
  `ID_METER` varchar(50) NOT NULL,
  `READING_DATE` datetime NOT NULL,
  `POWER_FORECAST` decimal(10,4) DEFAULT NULL,
  `P1` decimal(10,4) DEFAULT NULL,
  `P2` decimal(10,4) DEFAULT NULL,
  `P3` decimal(10,4) DEFAULT NULL,
  `P4` decimal(10,4) DEFAULT NULL,
  `P5` decimal(10,4) DEFAULT NULL,
  `MODELS` varchar(1000) DEFAULT NULL,
  `P_stormw_noaa` decimal(10,4) DEFAULT NULL,
  `P_stormw_sg` decimal(10,4) DEFAULT NULL,
  `P_accuw` decimal(10,4) DEFAULT NULL,
  `p_openw` decimal(10,4) DEFAULT NULL,
  `p_clear_sky` decimal(10,4) DEFAULT NULL,
  `p_w` decimal(10,4) DEFAULT NULL
) ;

CREATE TABLE pv_optim.`T_INVERTER_READINGS` (
  `ID_METER` varchar(50) DEFAULT NULL,
  `POWER_LOAD` decimal(10,2) DEFAULT NULL,
  `POWER_GEN` decimal(10,2) DEFAULT NULL,
  `STATUS` varchar(50) DEFAULT NULL,
  `POWER_BAT` decimal(10,2) DEFAULT NULL,
  `POWER_GRID` decimal(10,2) DEFAULT NULL,
  `SD_CAPACITY` decimal(3,0) DEFAULT NULL,
  `VPV` decimal(6,2) DEFAULT NULL,
  `IPV` decimal(6,2) DEFAULT NULL,
  `TIMESTAMP_R` timestamp(6) NULL DEFAULT NULL,
  `E_GEN` decimal(8,2) DEFAULT NULL,
  `E_LOAD` decimal(8,2) DEFAULT NULL,
  `E_GRID` decimal(8,2) DEFAULT NULL,
  `E_BAT_CHARGE` decimal(8,2) DEFAULT NULL,
  `E_BAT_DISCHARGE` decimal(8,2) DEFAULT NULL
) ;

CREATE TABLE pv_optim.`T_SASM_PARAMS` (
  `param_name` text,
  `param_value` double DEFAULT NULL,
  `param_description` text
) ;

CREATE TABLE pv_optim.`T_STORM_WEATHER` (
  `reading_date` datetime DEFAULT NULL,
  `id_location` varchar(10) DEFAULT NULL,
  `storm_weather` json DEFAULT NULL
) ;

CREATE TABLE pv_optim.`T_USERS` (
  `id` int NOT NULL,
  `username` varchar(30) DEFAULT NULL,
  `email` varchar(50) DEFAULT NULL,
  `password_hash` varchar(60) DEFAULT NULL
) ;

CREATE TABLE pv_optim.`T_WEATHER_FORECAST` (
  `reading_date` datetime NOT NULL,
  `open_weather` json DEFAULT NULL,
  `accu_weather` json DEFAULT NULL,
  `id_location` varchar(10) DEFAULT NULL
) ;

CREATE TABLE pv_optim.`T_WEATHER_READINGS` (
  `reading_date` datetime DEFAULT NULL,
  `open_weather` json DEFAULT NULL,
  `accu_weather` json DEFAULT NULL,
  `id_location` varchar(10) DEFAULT NULL
) ;

GRANT ALL PRIVILEGES ON pv_optim.* TO pv_optim@localhost;