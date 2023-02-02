''''PV Control'''
'''All operation are logged into T_APPLIANCE_LOGS'''
import sys
import mysql.connector
import json
import datetime as dt
import pandas as pd
import numpy as np

def logs_insert(id_appliance, str_now, command, pt, pf, soc, pg, p_app, pbat,db_local_connection):
    ins_logs="""INSERT INTO T_APPLIANCE_LOGS 
        (ID_APPLIANCE, TIMESTAMP_R, COMMAND, PT, PF, SOC, PG, P_APP, PBAT) 
        VALUES (%s, DATE_FORMAT(%s,'%Y-%m-%d %H:%i'), %s, %s, %s, %s, %s, %s, %s) """
    params=(id_appliance, str_now, command, pt, pf, soc, pg,p_app, pbat )
    cursor = db_local_connection.cursor()
    cursor.execute(ins_logs, params)
    db_local_connection.commit()
    cursor.close()

def SASM_check_option(db_local_connection):
    '''Params load_rate, soc din T_SASM_PARAMS'''
    sql_params="""select param_name, param_value from T_SASM_PARAMS"""
    df_params = pd.read_sql(sql_params, con=db_local_connection)
    sasm_control=int(df_params.loc[df_params['param_name']=='sasm_control', 'param_value'])
    if sasm_control==0:
        return 0,df_params
    else:
        return 1,df_params
def reschedule(app,str_now,db_local_connection):
    day=dt.datetime.now().strftime('%Y-%m-%d')
    id_app=app[0]
    papp=app[1]
    max_oper_time=app[2]
    shift=False
    '''Check for other schedules and reschedule'''
    #retrieve the initial and optimal schedule - appliances of type I that will operate in the next period
    sql_opt_schedule="""select s.ID_APPLIANCE, s.PMAX, DATE_FORMAT(s.START_TIME,'%Y-%m-%d %H:%i') oSTART_TIME
    , DATE_FORMAT(s.END_TIME,'%Y-%m-%d %H:%i') oEND_TIME, DATE_FORMAT(s.MAX_END_TIME,'%Y-%m-%d %H:%i') iEND_TIME
    from T_APPLIANCE_OPTIMAL_SCHEDULE s
    where ID_APPLIANCE=%s
    and DATE_FORMAT(s.END_TIME,'%Y-%m-%d %H:%i')>=%s
    and DATE_FORMAT(s.START_TIME,'%Y-%m-%d') =%s"""
    df_all = pd.read_sql(sql_opt_schedule, con=db_local_connection, params=(id_app, str_now,day,))
    df_all['oSTART_TIME']=pd.to_datetime(df_all['oSTART_TIME'])
    df_all['oEND_TIME']=pd.to_datetime(df_all['oEND_TIME'])
    df_all['iEND_TIME']=pd.to_datetime(df_all['iEND_TIME'])
    if len(df_all)>0: #daca are programari   
        df_all['READING_DATE']=pd.to_datetime(str_now)
        df_all['deltaT']=dt.timedelta(minutes=0)
        df_all.loc[(df_all['READING_DATE'] >=df_all['oSTART_TIME'])&(df_all['READING_DATE'] <df_all['oEND_TIME']), 'deltaT']=df_all.loc[(df_all['READING_DATE'] >=df_all['oSTART_TIME'])&(df_all['READING_DATE'] <df_all['oEND_TIME']), 'oEND_TIME']-df_all.loc[(df_all['READING_DATE'] >=df_all['oSTART_TIME'])&(df_all['READING_DATE'] <df_all['oEND_TIME']), 'READING_DATE']
        df_all['nSTART_TIME']=df_all['oSTART_TIME']+dt.timedelta(seconds=1)
        df_all['nEND_TIME']=df_all['oEND_TIME']
        df_all.loc[df_all['deltaT']>dt.timedelta(minutes=0), 'nSTART_TIME']=df_all['READING_DATE']+dt.timedelta(minutes=10)
        df_all.loc[df_all['deltaT']>dt.timedelta(minutes=0), 'nEND_TIME']=df_all['nSTART_TIME']+df_all['deltaT']
        while shift==False:
            #exit if nSTART_TIME>iEND_TIME end time-> shift=False
            if df_all.loc[df_all['deltaT']>dt.timedelta(minutes=0), 'iEND_TIME'].max()<=df_all.loc[df_all['deltaT']>dt.timedelta(minutes=0), 'nSTART_TIME'].max():
                print('Cannot exceed maximum end time') #set time at initial values
                df_all.loc[df_all['deltaT']>dt.timedelta(minutes=0), 'nSTART_TIME']=df_all.loc[df_all['deltaT']>dt.timedelta(minutes=0), 'oSTART_TIME']
                df_all.loc[df_all['deltaT']>dt.timedelta(minutes=0), 'nEND_TIME']=df_all.loc[df_all['deltaT']>dt.timedelta(minutes=0), 'oEND_TIME']
                df_all.loc[df_all['deltaT']>dt.timedelta(minutes=0), 'deltaT']=dt.timedelta(minutes=0)
                break
            else: #check for overlaps
               l_overlap=df_all.set_index(pd.IntervalIndex.from_arrays(df_all['nSTART_TIME'], df_all['nEND_TIME'], closed='both')).groupby('ID_APPLIANCE').apply(lambda df_all: df_all.index.is_overlapping)
               l_overlap=l_overlap.reset_index().rename(columns={0: "OVERLAP"})
               id_overlap=l_overlap.loc[l_overlap['OVERLAP']==True, 'ID_APPLIANCE'].tolist()
               if len(id_overlap)>0: #if there are overlaps then check again
                   print('overlaps)')
                   print('Reschedule at', df_all.loc[(df_all['deltaT']>dt.timedelta(minutes=0)), 'nSTART_TIME']+dt.timedelta(minutes=5))
                   df_all.loc[df_all['deltaT']>dt.timedelta(minutes=0), 'nSTART_TIME']=df_all.loc[df_all['deltaT']>dt.timedelta(minutes=0), 'nSTART_TIME']+dt.timedelta(minutes=5)
                   df_all.loc[df_all['deltaT']>dt.timedelta(minutes=0), 'nEND_TIME']=df_all.loc[df_all['deltaT']>dt.timedelta(minutes=0), 'nEND_TIME']+dt.timedelta(minutes=5)
               else: #no overlaps and nSTART_TIME>iEND_TIME-> shift=True
                   shift=True
        #if shift=True check  iEND_TIME 
        #if nEND_TIME>iEND_TIMEset endtime to iEND_TIME.
        if df_all.loc[df_all['deltaT']>dt.timedelta(minutes=0), 'iEND_TIME'].max()<=df_all.loc[df_all['deltaT']>dt.timedelta(minutes=0), 'nEND_TIME'].max():
            df_all.loc[df_all['deltaT']>dt.timedelta(minutes=0), 'nEND_TIME']=df_all.loc[df_all['deltaT']>dt.timedelta(minutes=0), 'iEND_TIME']
        lshift=df_all.loc[df_all['deltaT']>dt.timedelta(minutes=0), ['ID_APPLIANCE','nSTART_TIME', 'nEND_TIME','PMAX', 'iEND_TIME']].values.tolist()
       
    else:#if there are no other schedules then schedule it after 5 minutes with max_oper_time and build lshift
        shift=True    
        nSTART_TIME=dt.datetime.now()+dt.timedelta(minutes=5)
        nEND_TIME=nSTART_TIME+dt.timedelta(minutes=int(max_oper_time))
        iEND_TIME=nEND_TIME
        lshift=[[id_app, nSTART_TIME, nEND_TIME, papp, iEND_TIME]]
        
    if shift==True:
        #insert the new schedule into T_APPLIANCE_OPTIMAL_SCHEDULE 
        ins_sch="""INSERT INTO T_APPLIANCE_OPTIMAL_SCHEDULE  (ID_APPLIANCE, START_TIME, END_TIME, PMAX, MAX_END_TIME, ACTIVE) VALUES
        (%s,DATE_FORMAT(%s,'%Y-%m-%d %H:%i'),DATE_FORMAT(%s,'%Y-%m-%d %H:%i'),%s,DATE_FORMAT(%s,'%Y-%m-%d %H:%i'), 'Yes')"""
        cursor=db_local_connection.cursor()
        for l in lshift:
            params=(l[0], str(l[1]), str(l[2]), l[3], str(l[4]))
            cursor.execute(ins_sch,params)
        cursor.close()
        
        #update the actual schedule set END_TIME=now
        up_sch="""update T_APPLIANCE_OPTIMAL_SCHEDULE 
        set end_time=DATE_FORMAT(%s,'%Y-%m-%d %H:%i'),
        active='RESCHEDULED'
        where DATE_FORMAT(%s,'%Y-%m-%d %H:%i') between START_TIME and END_TIME
        and id_appliance=%s"""
        cursor=db_local_connection.cursor()
        params=(str_now,str_now, id_app)
        cursor.execute(up_sch,params )
        cursor.close()
        db_local_connection.commit()

    return shift


'''START/STOP AUTOMATICALLY'''
def SASM(lapp, operation,str_now,db_local_connection,df_params):
    load_rate=int(df_params.loc[df_params['param_name']=='sasm_load_rate', 'param_value']) #load excedent
    soc_discharged=int(df_params.loc[df_params['param_name']=='sasm_soc_discharged', 'param_value'])
    soc_charged=int(df_params.loc[df_params['param_name']=='sasm_soc_charged', 'param_value'])
    pbat_discharge=int(df_params.loc[df_params['param_name']=='pbat_discharge', 'param_value'])
    papp_to_control=int(df_params.loc[df_params['param_name']=='papp_to_control', 'param_value'])
    #retrieve the meter readings and forecast
    ora=(dt.datetime.now()-dt.timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M')
    
    sql_MR="""select power_load pt, power_gen pg, power_bat, power_grid pgrid
    , sd_capacity soc, POWER_FORECAST pf
    from T_INVERTER_READINGS MR, T_INVERTER_FORECAST MF
    where DATE_FORMAT(timestamp_r,'%Y-%m-%d %H:%i')>=%s 
    AND DATE_FORMAT(timestamp_r,'%Y-%m-%d %H')=DATE_FORMAT(READING_DATE,'%Y-%m-%d %H')
    order by timestamp_r desc limit 1"""
    cursor=db_local_connection.cursor()
    params=(ora,)
    cursor.execute(sql_MR,params)
    m_r = cursor.fetchall()
    cursor.close()
    for m in m_r:
        print('pt:', m[0], 'pg:',m[1], 'pbat:', m[2],'Grid:', m[3], 'SOC:', m[4], 'pf:', m[5])
        pt=float(m[0]) #total load
        pg=float(m[1]) #generated power
        pbat=m[2] #power from battery
        soc=int(m[4])  #soc
        pf=float(m[5]) #forecasted power
    
    #RULES: bpat<0 CHARGE; pbat>0 DISCHARGE   !!!!!!!!  
    if operation=='OFF': #Control start if papp>=papp_to_control:
        for app in lapp:
            if  app[1]>=papp_to_control:
                if soc<=soc_discharged and pbat>=pbat_discharge: #soc<=93 and pbat>=150
                    print (app[0], ":off") #send OFF command to interrupt
                    #call reschedule 
                    shift=reschedule(app, str_now,db_local_connection)
                    if shift==True: 
                        print('The appliance was rescheduled!')
                        logs_insert(app[0],str_now,'off-rescheduled', pt, pf, soc, pg,app[1],pbat,db_local_connection)
                    else:
                        print('No options to reschedule the appliance!')
                        logs_insert(app[0],str_now,'off-norescheduled', pt, pf, soc, pg,app[1],pbat,db_local_connection)
                    pt=pt-float(app[1])
                else:
                    print('Appliance', app[0],  'will not be interrupted')
    elif operation=='ON': #Control interruptions in case SASM receives lapp from the app_scheduler with operation=ON
        for app in lapp:
            pdif=pt+float(app[1])-max(pf,pg)
            if pdif<=load_rate and ((soc>=soc_charged and pbat<-pdif) or(soc>=soc_discharged and pbat<pbat_discharge)):
                   print (app[0],":on") #send ON command to switch on
                   logs_insert(app[0],str_now,'on', pt, pf, soc, pg,app[1],pbat,db_local_connection)
                   pt=pt+float(app[1])
            else:
                    print('The appliance wiill not start. Trying to reschedule it.......')
                    shift=reschedule(app, str_now,db_local_connection)
                    if shift==True: 
                        print('The appliance was rescheduled!')
                        logs_insert(app[0],str_now,'rescheduled', pt, pf, soc, pg,app[1],pbat, db_local_connection)
                    else:
                        print('No options to reschedule the appliance!')
                        logs_insert(app[0],str_now,'norescheduled', pt, pf, soc, pg,app[1], pbat, db_local_connection)

if __name__ == '__main__':
    try:
        db_local_connection = mysql.connector.connect(
          host="localhost", 
          user="pv_optim",
          passwd="pv_optim1234", 
          database="pv_optim", port=3306, auth_plugin='mysql_native_password'
        )
    except:
        print('App warning! Could not connect to local DB!')    
    check_control, df_params=SASM_check_option(db_local_connection)
    if check_control==1:
        #retrieve the appliances readings for type I and actionable='Yes'
        str_now=dt.datetime.now().strftime('%Y-%m-%d %H:%M')
        sql_AR="""select a.ID_APPLIANCE, avg(r.ACTIVE_POWER_CONS) P_App, a.max_oper_time MAX_OPER_TIME, a.priority PRIORITY
        from T_APPLIANCE_READINGS r, T_APPLIANCES a
        where r.ID_APPLIANCE=a.ID_APPLIANCE
        and substr(a.device_type,1,1)='I' and actionable='Yes'
        and r.status =1 AND DATE_FORMAT(r.READING_DATE,'%Y-%m-%d %H:%i') =%s
        group by a.ID_APPLIANCE, DATE_FORMAT(r.READING_DATE,'%Y-%m-%d %H:%i'),a.max_oper_time,a.priority"""
        df_app = pd.read_sql(sql_AR, con=db_local_connection, params=(str_now,))
        lapp=df_app[df_app['P_App']>0].sort_values(by='PRIORITY', ascending=False).values.tolist()
        if len(lapp)>0:
            SASM(lapp, 'OFF',str_now, db_local_connection, df_params)    
    else:
        print('PVControl is not activated! Set param SASM_control=1')
    db_local_connection.close()    

    