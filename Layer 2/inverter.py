'''Read the data from Growatt invertor and save the readings into the local DB'''
'''The script is working with on-grid Growatt inverters'''

'''For a complete API reference of growatServer please see: 
    https://github.com/indykoning/PyPi_GrowattServer
    including examples:
    https://github.com/indykoning/PyPi_GrowattServer/blob/master/examples/mix_example.py
    
Also, another Growatt API can be found at:
    https://github.com/Sjord/growatt_api_client'''

import datetime
import mysql.connector
import growattServer

db_local_connection = mysql.connector.connect(
  host="localhost",
  user="pv_optim",
  passwd="pv_optim1234",
  database="pv_optim", port=3306, auth_plugin='mysql_native_password'
)
ins_local="""INSERT INTO T_INVERTER_READINGS 
    (ID_METER, POWER_LOAD, POWER_GEN, STATUS, POWER_BAT, POWER_GRID, SD_CAPACITY,   TIMESTAMP_R,E_GEN, E_LOAD, E_GRID, E_BAT_CHARGE, E_BAT_DISCHARGE ) 
    VALUES ( %s, %s, %s, %s, %s, %s, %s, %s, %s, TIMESTAMP(%s),%s,%s,%s) """




'''____________PARAMETERS____________'''
'''PROVIDE YOUR USERNAME AND PASSOWRD FOR GROWATT.SERVER login page'''
username='test'
password='test'


timestamp_r=datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
api = growattServer.GrowattApi(True)
api.server_url='https://server.growatt.com/'
login_result=api.login(username, password)
userId=login_result['userId']
plant_info = api.plant_list(userId)
plant_id = plant_info["data"][1]["plantId"]
dev_list=api.device_list(plant_id)
device_sn=dev_list[0]['deviceSn']
#or check the growatt dashboard for Device Serial Number
#device_sn = "xxxxF" 
edata=api.storage_energy_overview (plant_id, device_sn)
actual_values=api.storage_detail(device_sn)
mix_status = api.mix_system_status(device_sn,plant_id)

params=(plant_id,mix_status['pLocalLoad'],float(mix_status['pPv1']) + float(mix_status['pPv2']), 'ACTUAL', mix_status['chargePower']-mix_status['pdisCharge1'],
        mix_status['pactouser'], actual_values['capacity'], timestamp_r,
        edata['epvTotal'],edata['useEnergyTotal'],edata['eToUserTotal'],edata['eChargeTotal'],edata['eDischargeTotal'],)
try:
    db_local_cursor = db_local_connection.cursor()
    db_local_cursor.execute(ins_local, params)
    db_local_connection.commit()
    print(str(db_local_cursor.rowcount) + " record inserted in local db")
    db_local_cursor.close()
    
except:
    print('Could not insert into local DB', params)