'''Read the app schedule from t_appliance_schedule and switch on/off the appliances'''
import sys
import mysql.connector
import json
import datetime as dt
import PVControl
try:
    connection = mysql.connector.connect(
      host="localhost", 
      user="pv_optim",
      passwd="pv_optim1234", 
      database="pv_optim", port=3306, auth_plugin='mysql_native_password'
    )
except:
    print('App warning! Could not connect to local DB!')    

#retrieve the list of appliances to control
sql_schedule="""select s.ID_APPLIANCE, DATE_FORMAT(START_TIME,'%Y-%m-%d %H:%i') START_TIME, DATE_FORMAT(END_TIME,'%Y-%m-%d %H:%i') END_TIME
, PMAX, ACTIONABLE, MAX_OPER_TIME, PRIORITY 
from T_APPLIANCE_OPTIMAL_SCHEDULE s, T_APPLIANCES a
where s.id_appliance=a.id_appliance
and (DATE_FORMAT(START_TIME,'%Y-%m-%d %H:%i') =%s OR DATE_FORMAT(END_TIME,'%Y-%m-%d %H:%i')=%s) 
and active='Yes'
ORDER BY PRIORITY"""
db_local_cursor=connection.cursor()
str_now=dt.datetime.now().strftime('%Y-%m-%d %H:%M')
#str_now='2022-03-08 12:20'
params=(str_now,str_now,)
db_local_cursor.execute(sql_schedule,params)
res_schedule = db_local_cursor.fetchall()
lapp_check=[]
check_control, df_params=PVControl.SASM_check_option(connection)
for row in res_schedule:
    if row[1]==str_now:
        if row[4]=='Yes' and check_control==1: #checks the PV availability for appliance Actionable=Yes
            lapp_check.append([row[0],row[3], row[5]])
        else: 
            print(row[0], ":on")
    elif row[2]==str_now:
        print(row[0], ":off")

if len(lapp_check)>0: 
    PVControl.SASM(lapp_check, 'ON',str_now, connection, df_params)
connection.close()


