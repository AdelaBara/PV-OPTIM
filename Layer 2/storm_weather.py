'''Get weather and solar data from stormglass.io - 10 calls per day'''

'''Docs API weather:
    https://docs.stormglass.io/?_gl=1*13pre5q*_ga*MzY1NzU0MTQzLjE2NjIyNzM4ODA.*_ga_79XDW52F27*MTY2MjI3Mzg3OS4xLjEuMTY2MjI3NDk0My4wLjAuMA..&_ga=2.225000858.1833362386.1662273880-365754143.1662273880#/weather'''


'''Docs API solar: UVIIndex, GHI
    https://docs.stormglass.io/?_gl=1*13pre5q*_ga*MzY1NzU0MTQzLjE2NjIyNzM4ODA.*_ga_79XDW52F27*MTY2MjI3Mzg3OS4xLjEuMTY2MjI3NDk0My4wLjAuMA..&_ga=2.225000858.1833362386.1662273880-365754143.1662273880#/solar'''

import requests
import time
import mysql.connector
import json
import datetime as dt
import pytz, dateutil.parser

import sys

db_local_connection = mysql.connector.connect(
  host="localhost", #replace with localhost on RP
  user="pv_optim",
  passwd="pv_optim1234", #MeterPass pe RP
  database="pv_optim", port=3306, auth_plugin='mysql_native_password'
)
ins_wf="""INSERT INTO T_STORM_WEATHER 
    (ID_LOCATION, READING_DATE, STORM_WEATHER) 
    VALUES (%s, DATE_FORMAT(%s,'%Y-%m-%d %H:%i'), %s )"""
del_wf="""delete from T_STORM_WEATHER 
    where reading_date=DATE_FORMAT(%s,'%Y-%m-%d %H:%i') and id_location=%s"""

def dml_db(params, sql, db_local_connection):
    db_local_cursor = db_local_connection.cursor()
    db_local_cursor.execute(sql, params)
    db_local_connection.commit()
    print(str(db_local_cursor.rowcount) + " weather records affected" )
    db_local_cursor.close()


def get_storm_weather(plat, plon,id_location, API_key, time_zone):
    #weather data
    response = requests.get(
      'https://api.stormglass.io/v2/weather/point',
      params={
        'lat': plat,
        'lng': plon,
        'params': ','.join(['pressure', 'airTemperature', 'cloudCover', 'humidity', 'precipitation','windDirection','windSpeed', 'visibility']),
      },
      headers={
        'Authorization': API_key
      }
    )
    json_data = response.json()
    
    #solar data: UVIndex, GHI
    response = requests.get(
      'https://api.stormglass.io/v2/solar/point',
      params={
        'lat': plat,
        'lng': plon,
        'params': ','.join(['uvIndex', 'downwardShortWaveRadiationFlux']),
      },
      headers={
        'Authorization':  API_key
      }
    )
    json_data_solar = response.json()
    
    #concatenate weather and solar parameters
    zip_list=list(zip(json_data['hours'], json_data_solar['hours']))
    for z in zip_list:
        z[0].update(z[1]) 
        rec_storm=json.dumps(z[0])
        utctime = dateutil.parser.parse(z[0]['time']) #convert UTC time to datetime
        reading_date=utctime.astimezone(pytz.timezone(time_zone)) #convert datetime form UTC to locat time
        params=(reading_date,id_location,)
        dml_db(params, del_wf,db_local_connection )
        params=(id_location, reading_date, rec_storm,  )
        dml_db(params, ins_wf,db_local_connection )

'''____________PARAMETERS____________'''
# Set your latitude, longitude and local time
plat=45.943
plon=25.021
time_zone="Europe/Bucharest"
#Provide your API_key for storm weather API
API_key='xxxx-xxx-xxx-xxx-xxx-xxx-xxx'
#modify id_location ONLY if you have multiple locations
id_location=289883

get_storm_weather(plat, plon, id_location, API_key, time_zone )


