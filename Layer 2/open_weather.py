'''Read the current weather data and forecast from openweather and save the records into the local db as json'''
import urllib.request, json
import time
import mysql.connector
import datetime as dt

def dml_db(params, sql, db_local_connection):
    db_local_cursor = db_local_connection.cursor()
    db_local_cursor.execute(sql, params)
    db_local_connection.commit()
    print(str(db_local_cursor.rowcount) + " weather records affected" )
    db_local_cursor.close()
    
def weather_readings(plat, plon, id_location, API_key):    
    '''Openweather API'''
    with urllib.request.urlopen("https://api.openweathermap.org/data/2.5/onecall?lat="+str(plat)+"&lon="+str(plon)+"&units=metric&appid="+API_key) as url:
        data_open = json.loads(url.read().decode())
    rec_open=data_open['current'] #current readings
    rec_open['dt']=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(rec_open['dt'])))
    rec_open=json.dumps(rec_open)
    forw=data_open['hourly'] #forecast 48h
    for i in range (48):
        forw[i]['dt']=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(forw[i]['dt'])))
    '''Current readings'''
    now= dt.datetime.now().strftime('%Y-%m-%d %H:%M')
    db_local_connection = mysql.connector.connect(
      host="localhost", #replace with localhost on RP
      user="pv_optim",
      passwd="pv_optim1234", #MeterPass pe RP
      database="pv_optim", port=3306, auth_plugin='mysql_native_password'
    )
    #insert into local DB
    ins_wr="""INSERT INTO T_WEATHER_READINGS 
        (ID_LOCATION, READING_DATE, OPEN_WEATHER) 
        VALUES (%s, DATE_FORMAT(%s,'%Y-%m-%d %H:%i'), %s ) """
    params=(id_location, now, rec_open, )
    dml_db(params, ins_wr,db_local_connection )
    
    '''Weather forecast'''
    ins_wf="""INSERT INTO T_WEATHER_FORECAST 
        (ID_LOCATION, READING_DATE, OPEN_WEATHER) 
        VALUES (%s, DATE_FORMAT(%s,'%Y-%m-%d %H:%i'), %s )"""
    del_wf="""delete from T_WEATHER_FORECAST 
        where reading_date=DATE_FORMAT(%s,'%Y-%m-%d %H:%i') and id_location=%s"""
    for i in range (48):
        rec_open=json.dumps(forw[i])
        reading_date=forw[i]['dt']
        params=(reading_date,id_location,)
        dml_db(params, del_wf,db_local_connection )
        params=(id_location, reading_date, rec_open, )
        dml_db(params, ins_wf,db_local_connection )
    
'''____________PARAMETERS____________'''
# Set your latitude, longitude and local time
plat=45.943
plon=25.021
#Provide your API_key for openweather API
API_key='xxxx-xxx-xxx-xxx-xxx-xxx-xxx'
#modify id_location ONLY if you have multiple locations
id_location=289883
weather_readings(plat, plon, id_location, API_key)
