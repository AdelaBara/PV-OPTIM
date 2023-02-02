'''PVOptim - Optimizes the day-ahead schedule based on PV forecast'''

import pandas as pd
import mysql.connector
import matplotlib as plt
import datetime as dt
import numpy as np

'''Generates H columns'''
def generate_hicols(interval,hinterval):
    hicols=[]
    for h in range (0,24):
        for hi in range(0, hinterval):
            col='H'+str(h)+'_'+str(hi)
            hicols.append(col)
    return hicols

'''Load user preferences and build constraints matrix'''
def generate_app_constraints(day, interval,hinterval):
    '''retrieve the user preferences for the day'''
    df_AS=pd.read_csv("data_folder/app_constraints.csv")
    df_AS['istart_time']=pd.to_datetime(df_AS['istart_time'])
    df_AS['iend_time']=pd.to_datetime(df_AS['iend_time'])
    df_AS['schedule_date']=pd.to_datetime(df_AS['istart_time']).dt.date
    #transform the user preferences into constraints
    df_AR=df_AS[['ID_APPLIANCE', 'OPERATION']]
    df_AS['start_hour']=(df_AS['istart_time']).dt.hour
    df_AS['end_hour']=(df_AS['iend_time']).dt.hour
    df_AS['start_i']=round((df_AS['istart_time']).dt.minute/interval)
    df_AS['end_i']=round((df_AS['iend_time']).dt.minute/interval)
    for i in range(0, len(df_AS)):   
        sh=df_AS.iloc[i,9] #start hour
        eh=df_AS.iloc[i,10] #stop hour
        shi=df_AS.iloc[i,11] #interval start
        ehi=df_AS.iloc[i,12] #interval stop
        id_app=df_AS.iloc[i,0]
        oper=df_AS.iloc[i,5]
        for h in range (0,24):
            for hi in range(0, hinterval):
                col='H'+str(h)+'_'+str(hi)
                
                if h > sh and h<eh:
                   df_AR.loc[(df_AR['ID_APPLIANCE']==id_app) & (df_AR['OPERATION']==oper),col]=1
                elif (h==sh and hi >= shi) or (h==eh and hi<=ehi):
                    df_AR.loc[(df_AR['ID_APPLIANCE']==id_app) & (df_AR['OPERATION']==oper),col]=1
                else:
                   df_AR.loc[(df_AR['ID_APPLIANCE']==id_app) & (df_AR['OPERATION']==oper),col]=0
    df_AS=df_AS.merge(df_AR, on=['ID_APPLIANCE', 'OPERATION'])
    return df_AS


'''Load PV forecast from inverter_forecast.csv'''
def load_PV (day, hicols):
    pvg=pd.read_csv('data_folder/inverter_forecast.csv', usecols=[0,1])
    pvg.rename(columns={'PG_PRED':'PV'}, inplace=True)
    if pvg['PV'].max() <10: pvg['PV']=pvg['PV']*1000 #in kWh
    pvg['Hour']=pd.to_datetime(pvg['reading_date']).dt.hour
    pvg['Day']=pd.to_datetime(pvg['reading_date']).dt.strftime('%Y-%m-%d')
    #pvg=pvg.pivot(index='Day', columns='Hour', values='PV')
    pvg=pvg.loc[pvg['Day']==day, :].pivot_table(index='Day', columns='Hour', values='PV', aggfunc='mean')
    cols=pvg.columns
    for col in cols:
        pvg.rename(columns={col:'H'+str(col)+'_0'}, inplace=True)
    clist=[] #rearange columns in order H0...H23 and fill with 0 during night
    for h in range (0,24):
        col='H'+str(h)+'_0'
        clist.append(col)
        if h not in cols: 
            pvg[col]=0
    pvg=pvg[clist]
    for col in hicols:
        if col not in clist:
            pvg[col]=np.nan
    pvg=pvg[hicols]
    pvg.interpolate(axis=1, inplace=True)
    return pvg

'''Load tariffs from tariffs.csv'''
def load_ToU (hicols,hinterval ):
    tfh = pd.DataFrame()
    for col in hicols:
        tfh[col]=[0] 
    dftf=pd.read_csv("data_folder/tariffs.csv")
    for i in range(0, len(dftf)):
         start=int(dftf.iloc[i,1])
         end=int(dftf.iloc[i,2])
         #tranform dftf into 24 vector->tfh
         for h in range(start,end+1):
             for hi in range(0, hinterval):
                 col='H'+str(h)+'_'+str(hi)
                 tfh[col]=float(dftf.iloc[i,0])
    return tfh

'''Schedule appliance at hmin'''
def muta_app (app, dfch, dfsch, hmin, hicols,loptschedule, hinterval):
    #ROT
    if dfch.loc[dfch['UID_APPLIANCE']==app, 'TYPE'].iloc[0] in ('I', 'B'):
        req_ot=1
    else: #TYPE='S'
        req_ot=int(dfch.loc[dfch['UID_APPLIANCE']==app, 'ROT'])
    starth=dfch.columns.get_loc(hmin) #index of the start hour in dfch
    endh=starth+req_ot #index of the start hour in dfch
    print('Mutam: ', app, 'intre orele:', dfch.columns[starth], 'si', dfch.columns[endh])
    #increased no of SCHEDULED app and decreased NO_OPERATIONS
    dfch.loc[dfch['UID_APPLIANCE']==app, 'SCHEDULED']=dfch.loc[dfch['UID_APPLIANCE']==app, 'SCHEDULED']+1
    dfch.loc[dfch['UID_APPLIANCE']==app, 'NO_OPERATIONS']=dfch.loc[dfch['UID_APPLIANCE']==app, 'NO_OPERATIONS']-1
    no_op=int(dfch.loc[dfch['UID_APPLIANCE']==app, 'NO_OPERATIONS'])
    if no_op==0: #eliminate the appliance from the list and set dfch[H0..H23]=0
        for col in hicols: #range (0,24):
            dfch.loc[dfch['UID_APPLIANCE']==app, col]=0
    #for appliance type S set the optimal load for the next intervals of ROT
    if dfch.loc[dfch['UID_APPLIANCE']==app, 'TYPE'].iloc[0] in ('S'):
        for h in range (starth, min(endh,24*hinterval)):
            col=dfch.columns[h]
            dfsch.loc[dfsch['UID_APPLIANCE']==app, col]=dfch.loc[dfch['UID_APPLIANCE']==app, 'CAPACITY']
            dfch.loc[dfch['UID_APPLIANCE']==app, col]=0
    else:  #for app type I and B at hmin set optimal load=CAPACITY in dfsch  
        dfsch.loc[dfsch['UID_APPLIANCE']==app, hmin]=dfch.loc[dfch['UID_APPLIANCE']==app, 'CAPACITY']
        dfch.loc[dfch['UID_APPLIANCE']==app, hmin]=0
    loptschedule.append([app,dfch.columns[starth], dfch.columns[endh]])
    return dfch, dfsch,loptschedule

'''OPTIMIZE DAY AHEAD - MINIMIZE COST'''
def optim_das (day, interval, hinterval, pcmax, hicols, pvg, tfh):
    dfch=pd.read_csv("data_folder/app_list.csv")
    #build constraints 
    df_AS=generate_app_constraints(day, interval,hinterval)
    dfch=dfch.merge(df_AS, on='ID_APPLIANCE', copy=False)
    dfch['CAPACITY']=dfch['CAPACITY']*dfch['COEF']
    dfch.drop('COEF', axis=1, inplace=True)
    #order apps
    dfch.sort_values(by=['TYPE','CAPACITY'], ascending=False,inplace=True)
    #unique ID for ID_APPLIANCE_OPERATION
    dfch['UID_APPLIANCE']=dfch['ID_APPLIANCE'].astype(str)+'_'+dfch['OPERATION'].astype(str)
    
    '''Initializare OPTIMAL SCHEDULED ->dfsch'''
    dfsch=pd.DataFrame()
    dfsch[['ID_APPLIANCE', 'OPERATION', 'UID_APPLIANCE']]=dfch[['ID_APPLIANCE', 'OPERATION','UID_APPLIANCE']]
    for col in hicols:
        dfch[col]=dfch[col]*dfch['CAPACITY']
        dfsch[col]=0 #initialize the optimal load
    dfsch['SCHEDULE_DATE']=day    
    dfch['START_TIME'] =0 #no of devices to be scheduled (START_TIME<0). Initial START_TIME=-1
    dfch['SCHEDULED'] =0
    dfch['MAX_END_TIME']=pd.to_datetime(dfch['iend_time'])+pd.to_timedelta(dfch['ROT'], unit='m')
    dfch['ROT']=np.ceil(dfch['ROT']/interval)
    dfch.loc[dfch['TYPE'].isin(['I','B']), 'NO_OPERATIONS']=dfch.loc[dfch['TYPE'].isin(['I','B']), 'NO_OPERATIONS']*dfch.loc[dfch['TYPE'].isin(['I','B']), 'ROT']
    dfch['RESCHEDULED'] =0 
    
    '''Initialization'''
    PGav=pvg.copy() #PV available - updated after each schedule
    rest=dfch['NO_OPERATIONS'].sum() #no of app to schedule
    loptschedule=[]  #list of scheduled apps (start, end) from muta_app()
    
    while rest>0:
        lapp=dfch.loc[dfch['NO_OPERATIONS']>0, 'UID_APPLIANCE'].values.tolist()
        for app in lapp:
            capp=dfch.loc[dfch['UID_APPLIANCE']==app, 'CAPACITY'].max() #Capacity of app
            if dfch.loc[dfch['UID_APPLIANCE']==app, 'TYPE'].iloc[0] in ('I', 'B'): #ROT =1 for B and I
                req_ot=1
            else: #TYPE='S' ROT
                req_ot=int(dfch.loc[dfch['UID_APPLIANCE']==app, 'ROT'])
            hicolsop=[] #operating hours for app
            for col in hicols: #range (0,24):
               if dfch.loc[dfch['UID_APPLIANCE']==app, col].max()>0: #if app can operate at col
                    hicolsop.append(col)
            cmin=pow(dfch['CAPACITY'].sum()*tfh.iloc[0,0].max(),2) #initializing cmin with sum of all the app
            for hicol in hicolsop:
                no_resch=int(dfch.loc[dfch['UID_APPLIANCE']==app, 'RESCHEDULED'])
                #check not to exceed pcmax
                starth=dfch.columns.get_loc(hicol) #index of start hour from dfch
                endh=starth+req_ot #index of end hour from dfch
                costapp=0
                to_sch=True
                for h in range (starth, min(endh,24*hinterval)):
                    col=dfch.columns[h]
                    ctot=dfsch[col].sum()
                    if capp-PGav[col].sum()>0:
                        costapp=costapp+(capp-PGav[col].sum())*tfh[col].sum()
                    else:
                        costapp=costapp+0 #add feed-in tariffs
                    if (ctot+capp>pcmax)  and no_resch<5: 
                        to_sch=False
                        print(app, 'cannot operate at hour:', col, 'exceeds pcmax')
                        break
                if (to_sch==True and costapp<cmin):
                    hmin=hicol
                    cmin=costapp
            if to_sch==True:#if app can be scheduled
                print('Schedule:',app, 'at hour:', hmin)
                dfch, dfsch,loptschedule=muta_app(app, dfch, dfsch, hmin, hicols,loptschedule, hinterval)
                for col in hicols: 
                     PGav[col]=pvg[col]-dfsch[col].sum()
                rest_restrictii=(dfch.loc[:,'H1_0':'H23'+'_'+str(hinterval-1)].sum()).sum()
                rest=min(rest_restrictii, dfch['NO_OPERATIONS'].sum())
                break #check if app needs more intervals
            else: #increase no of reschedules in case app could not be scheduled at current iteration
              dfch.loc[dfch['UID_APPLIANCE']==app, 'RESCHEDULED']=dfch.loc[dfch['UID_APPLIANCE']==app, 'RESCHEDULED']+1
            print('No of appliances that needs to be scheduled:', rest)
    return dfsch, dfch, loptschedule

'''Generate and save the optimal schedule in optimal_schedule.csv'''
def save_app_schedule(day, dfch,dfsch,interval, loptschedule):
    dfoptsch=pd.DataFrame(loptschedule, columns=['UID_APPLIANCE', 'HSTART', 'HEND'])
    dfoptsch=dfoptsch.merge(dfch[['ID_APPLIANCE','UID_APPLIANCE', 'CAPACITY', 'MAX_END_TIME']], on='UID_APPLIANCE')
    dfoptsch['START']=dfoptsch['HSTART'].str.strip('H')
    dfoptsch['START']=dfoptsch['START'].str.split(pat="_")
    dfoptsch['START_TIME']=day +' '+ dfoptsch['START'].str.get(0)+':'+(dfoptsch['START'].str.get(1).astype(int)*interval).astype(str)
    dfoptsch['START_TIME']= pd.to_datetime(dfoptsch['START_TIME'], format='%Y-%m-%d %H:%M').astype(str)
    dfoptsch['END']=dfoptsch['HEND'].str.strip('H')
    dfoptsch['END']=dfoptsch['END'].str.split(pat="_")
    dfoptsch['END_TIME']=day +' '+ dfoptsch['END'].str.get(0)+':'+(dfoptsch['END'].str.get(1).astype(int)*interval).astype(str)
    dfoptsch['END_TIME']= pd.to_datetime(dfoptsch['END_TIME'], format='%Y-%m-%d %H:%M').astype(str)
    dfoptsch[['ID_APPLIANCE','START_TIME', 'END_TIME','CAPACITY', 'MAX_END_TIME']].to_csv("data_folder/optimal_schedule.csv", index=False)
    return dfoptsch[['ID_APPLIANCE','START_TIME', 'END_TIME','CAPACITY']]

'''Transform columns H in timestamp'''
def mapH_timestamp(df, interval, day):
    dfT=df.T
    dfT['SCH_TIME']=dfT.index.values
    dfT['SCH_TIME']=dfT['SCH_TIME'].str.strip('H')
    dfT['SCH_TIME']=dfT['SCH_TIME'].str.split(pat="_")
    dfT['SCH_TIME']=day +' '+ dfT['SCH_TIME'].str.get(0)+':'+(dfT['SCH_TIME'].str.get(1).astype(int)*interval).astype(str)
    dfT['SCH_TIME']= pd.to_datetime(dfT['SCH_TIME'], format='%Y-%m-%d %H:%M')
    dfT['SCH_TIME']= dfT['SCH_TIME'].dt.time.astype(str)
    dfT.set_index('SCH_TIME', inplace=True)
    return dfT

def app_id_mapper(dfsch):
    l_id_app=dfsch['ID_APPLIANCE'].to_list()
    l_uid_app=dfsch['UID_APPLIANCE'].to_list()
    zip_iterator = zip(l_uid_app, l_id_app)
    col_mapper=dict(zip_iterator) 
    return col_mapper


'''PARAMETERS'''
interval=30 #interval for optimization in minutes (20, 30, 60)
pcmax=4500 #maximum load per interval
# day=dt.datetime.now().strftime('%Y-%m-%d %H:%M') 
id_meter='2060015235'
day='2022-08-02'
TNP_load=200

'''____________MAIN__________________'''
def main_optim(day, interval, pcmax, TNP_load):
    hinterval=int(60/interval)  #no intervals/hour
    print(day, interval, pcmax)
    hicols=generate_hicols(interval,hinterval)
    '''Load ToU -> tfh, PV FORECAST -> pvg'''
    tfh=load_ToU (hicols, hinterval)
    pvg=load_PV(day, hicols)
    #Optimize schedule 
    dfsch, dfch, loptschedule=optim_das( day, interval, hinterval, pcmax, hicols, pvg, tfh)
    #Save schedule in T_APPLIANCE_SCHEDULE
    dfoptsch=save_app_schedule(day,dfch, dfsch,interval, loptschedule)
    '''Tranform the optimal schedule for charts -> dfoptschT'''
    dfschplot=dfsch[hicols]
    dfschplot['UID_APPLIANCE']=dfsch['UID_APPLIANCE']
    dfschplot.set_index('UID_APPLIANCE', inplace=True)
    dfschplotT=mapH_timestamp(dfschplot, interval, day)
    #add PV at dfschplotT
    pvgT=mapH_timestamp(pvg, interval, day)
    pvgT.rename(columns={pvgT.columns[0]:'PV'}, inplace=True)
    dfschplotT['PV']=pvgT['PV'].astype(float)
    #replace UID_APPLIANCE  
    col_mapper=app_id_mapper(dfsch)
    dfschplotT.rename(columns=col_mapper, inplace=True)
    #add TNP_load
    dfschplotT['TNP_load']=TNP_load
    '''Hourly costs -> costh'''
    costh = pd.DataFrame()
    for col in hicols:
        costh[col]=[0]
    cth = pd.DataFrame() 
    for col in hicols: #range (0,24):
        #col='H'+str(h)
        cth[col]=[dfsch[col].sum()]
        costh[col]=(cth[col]+TNP_load)*tfh[col]/1000
    costhT=mapH_timestamp(costh, interval, day)
    cthT=mapH_timestamp(cth, interval, day)
    CT_day=((cthT/1000).sum()/hinterval).sum()
    PG_day=((pvgT/1000).sum()/hinterval).sum()
    return dfschplotT, CT_day, PG_day

dfschplotT, CT_day, PG_day=main_optim(day, interval, pcmax,TNP_load)   
cols=dfschplotT.columns.to_list()
cols.remove('PV')
dfschplotT['Deficit/Surplus']=dfschplotT['PV']-dfschplotT[cols].sum(axis=1)
dfschplotT.to_csv("data_folder/deficit_surplus.csv")
cols.append('PV')
plt.rcParams.update({'font.size': 22})
ax =dfschplotT['PV'].plot(x=None, linewidth=5, fontsize=22) 
ax.set_ylabel('kw',fontdict={'fontsize':24})
cols.remove('PV')
dfschplotT[cols].plot(x=None,figsize=(30,18), kind='bar',stacked=True,legend=True, ax=ax, fontsize=22)
ax.set_title("Daily optimal schedule",pad=20, fontdict={'fontsize':24})
ax.legend(loc=1,fontsize=20, title="")
