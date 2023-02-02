'''Read the appliances conumption and save it to the local DB'''
import sys
import mysql.connector
import json
import datetime as dt
# data_in="""{"voltage_mv":216075,"current_ma":33,"power_mw":1444,"total_wh":31,"err_code":0,"current":0.033,"power":1.444,
# "total":0.031,"voltage":216.075,"timestamp":"2020-11-30T12:30:33+02:00","id_appliance":"EV_1010"}"""
#print(sys.stdin.readline())
data_in=sys.stdin.readline()
dd=json.loads(data_in)
#print(dd)
print('APPLIANCES READINGS AT: ', dt.datetime.now().strftime('%Y-%m-%d %H:%M'))
try:
    db_local_connection = mysql.connector.connect(
      host="localhost", #replace with localhost on RP
      user="pv_optim",
      passwd="pv_optim1234", #MeterPass pe RP
      database="pv_optim", port=3306, auth_plugin='mysql_native_password'
    )
    #retrieve the appliances connected to the sensors
    sql_app="""select ID_APPLIANCE, capacity  from T_APPLIANCES"""
    cursor=db_local_connection.cursor()
    cursor.execute(sql_app)
    app_list = cursor.fetchall()
    cursor.close()
    
except:
    print('App warning! Could not connect to local DB!')
for key, d in dd.items():
    for app in app_list:
        #print(app[0], app[1], app[2])
        if d['id_sensor']==app[0]:
            id_appliance=app[0] #id_appliance asociat cu senzorul
            err_code=d["err_code"]
            power=d["power_mw"]/1000
            voltage=d["voltage_mv"]/1000
            current=d["current_ma"]/1000
            totalP=d["total_wh"]/1000
            if power==0:
                status=0
            else: 
                status=1
            str_now=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            params = (id_appliance,status,str_now,voltage, current, power,totalP,err_code)
            #print(params)
            try:
                 #insert into local DB
                ins_local="""INSERT INTO T_APPLIANCE_READINGS 
                    (ID_APPLIANCE, STATUS,READING_DATE, VOLTAGE_MV, CURRENT_MA, ACTIVE_POWER_CONS, TOTAL_WH, ERR_CODE1) 
                    VALUES (%s,%s,TIMESTAMP(%s), %s, %s, %s, %s, %s) """
                db_local_cursor = db_local_connection.cursor()
                db_local_cursor.execute(ins_local, params)
                db_local_connection.commit()
                print(str(db_local_cursor.rowcount) + " record inserted in local db " +id_appliance)
                db_local_cursor.close()
            except:
                print('App warning! Could not insert into local DB', params)
