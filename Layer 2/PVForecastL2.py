''''PV-OPTIM Forecast LAYER 2'''
import mysql.connector
mysql.connector.conversion.MySQLConverter._timestamp_to_mysql = mysql.connector.conversion.MySQLConverter._datetime_to_mysql
import json
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option("display.max_columns",100)
pd.set_option("display.max_rows",100)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

def percentage_error(actual, predicted): #calculate MAPE
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred))))

def prelWR(dfm):
    dfm['reading_date']=pd.to_datetime(dfm['dt'])
    dfm['reading_date']=dfm['reading_date'].dt.round('60min')
    dfm['weather']=dfm['weather'].apply(pd.Series)[0]
    dfm=pd.concat([dfm.drop(['weather'], axis=1), dfm['weather'].apply(pd.Series)], axis=1)
    if 'rain' in dfm.columns:
        dfm['rain']=dfm['rain'].apply(pd.Series)['1h']
        dfm['rain']=dfm['rain'].fillna(0)
    else:
        dfm['rain']=0
    dfm.drop('dt', axis=1, inplace=True)   
    cols=dfm.select_dtypes(include=np.number).columns.tolist()
    for col in cols:
        dfm[col]=dfm[col].interpolate(method='linear',limit_direction='both' )
        
    cols=dfm.select_dtypes(exclude=np.number).columns.tolist()
    for col in cols:
        dfm[col]=dfm[col].interpolate(method='pad')
    return dfm

def save_forecast(data_test):
    try:
        del_forecast="""delete from T_INVERTER_FORECAST where reading_date=DATE_FORMAT(%s,'%Y-%m-%d %H:%i')"""
        ins_forecast="""INSERT INTO T_INVERTER_FORECAST 
            (READING_DATE, POWER_FORECAST,P1,P2, P3, P4, P5, ID_METER) 
            VALUES (DATE_FORMAT(%s,'%Y-%m-%d %H:%i'),%s, %s, %s, %s, %s, %s, %s) """
        for d in data_test.to_numpy():
            d[0]=d[0].strftime('%Y-%m-%d %H:%M')
            params=tuple(d)
            param_del=(d[0],)
            cursor = pvoptim_connection.cursor()
            cursor.execute(del_forecast,param_del )
            cursor.close()
            pvoptim_connection.commit()
            cursor = pvoptim_connection.cursor()
            cursor.execute(ins_forecast, params)
            cursor.close()
            pvoptim_connection.commit()
            
        print("PV Forecast inserted in PV-OPTIM db")
    except:
        print('App warning! Could not insert into PV-OPTIM DB', params)
        
def get_storm_weather(start, end):
    '''Retrive STORM_WEATHER >= day'''
    sql_swr="""select DATE_FORMAT(READING_DATE,'%Y-%m-%d %H:%i') reading_date, STORM_WEATHER
    from T_STORM_WEATHER
    where reading_date >=DATE_FORMAT(%s,'%Y-%m-%d %H:%i') 
    and reading_date <=DATE_FORMAT(%s,'%Y-%m-%d %H:%i')
    ORDER BY reading_date"""
    df_SW = pd.read_sql(sql_swr, con=pvoptim_connection, params=(start,end,))
    df_SW['STORM_WEATHER']=df_SW['STORM_WEATHER'].apply(json.loads)
    df_SW=pd.concat([df_SW.drop(['STORM_WEATHER'], axis=1), df_SW['STORM_WEATHER'].apply(pd.Series)], axis=1)
    df_SW.drop('time', axis=1, inplace=True)
    #print(df_SW.columns.to_list)
    for colSW in df_SW.columns:
        if colSW !='reading_date':
            dfwc=df_SW[colSW].apply(pd.Series)
            for col in dfwc.columns:
                dfwc.rename(columns={col: colSW+'_'+str(col)}, inplace=True)
            df_SW=pd.concat([df_SW.drop([colSW], axis=1), dfwc], axis=1)
    df_SW['reading_date']=pd.to_datetime(df_SW['reading_date'], format='%Y-%m-%d %H:%M')
    return df_SW        

def get_readings(day):
    '''Retrieve the meter readings from BD'''
    sql_MR="""select timestamp_r reading_date, power_gen*1000 PG_AVG
    from T_INVERTER_READINGS
    where datediff(DATE_FORMAT(%s,'%Y-%m-%d %H:%i'), timestamp_r)<=30
    and timestamp_r <DATE_FORMAT(%s,'%Y-%m-%d %H:%i')
    and id_meter='2060015235'
    """
    df_MR = pd.read_sql(sql_MR, con=pvoptim_connection, params=(day,day,))
    df_MR['reading_date']=pd.to_datetime(df_MR['reading_date'], format='%d-%m-%Y %H:%M:%S')
    df_MR.sort_values(by='reading_date', inplace=True)
    df_MR.drop_duplicates(subset='reading_date', keep="last", inplace=True)
    start=df_MR['reading_date'].min()
    end=df_MR['reading_date'].max()

    '''Retrive WR from accuweather and openweather'''
    sql_wr="""select DATE_FORMAT(READING_DATE,'%Y-%m-%d %H:%i') READING_DATE, OPEN_WEATHER
    from T_WEATHER_READINGS
    where READING_DATE >=DATE_FORMAT(%s,'%Y-%m-%d %H:%i')
    and READING_DATE <=DATE_FORMAT(%s,'%Y-%m-%d %H:%i')
    ORDER BY READING_DATE"""
    df_WR = pd.read_sql(sql_wr, con=pvoptim_connection, params=(start,end,))
    df_WR['OPEN_WEATHER']=df_WR['OPEN_WEATHER'].apply(json.loads)
    df_WR=pd.concat([df_WR.drop(['OPEN_WEATHER'], axis=1), df_WR['OPEN_WEATHER'].apply(pd.Series)], axis=1)
    df_WR=prelWR(df_WR)
    
    '''Get stormweather data'''
    df_SW=get_storm_weather(start,end)
    dfm=pd.merge(df_WR, df_SW, how='inner', on ='reading_date')

    '''set  train: meter readings + weather readings'''
    df=pd.merge(dfm, df_MR, how='right', on ='reading_date')
    df.reset_index(inplace=True)
    df.drop(['index'], axis=1, inplace=True)
    df.sort_values(by='reading_date', inplace=True)
    df.drop_duplicates(subset='reading_date', keep="last", inplace=True)
    df['reading_date']=pd.DatetimeIndex(df['reading_date'],tz=tz, ambiguous='NaT')
    df=df.loc[df['reading_date'].notnull()].reset_index()
    
    #set time to continuous 15 min interval
    start=df['reading_date'].min()
    end=df['reading_date'].max()
    time = pd.date_range(start=start, end=end, freq='15min', tz = 'Europe/Bucharest').to_frame(name='reading_date')
    df=df.merge(time, on='reading_date', how='right')
    
    #interpolate weather variables
    colsn=list(df.select_dtypes(include=[np.number]).columns.values)
    for col in colsn:
        df[col].interpolate(method='linear', limit_direction='both' ,inplace=True)
    colsn=list(df.select_dtypes(exclude=[np.number]).columns.values)
    for col in colsn:
        df[col].interpolate(method='pad', inplace=True)
    return df, df_WR, dfm 

def get_weather_forecast(day, weather_attributes):
    # retrieve the forecast of the weather readings for the next max 10 days
    end=dt.datetime.strptime(day, '%Y-%m-%d')+dt.timedelta(days=10)
    sql_wr="""select DATE_FORMAT(READING_DATE,'%Y-%m-%d %H:%i') READING_DATE, OPEN_WEATHER
    from T_WEATHER_FORECAST
    where READING_DATE > DATE_FORMAT(%s,'%Y-%m-%d')  
    and READING_DATE<=DATE_FORMAT(%s,'%Y-%m-%d')
    ORDER BY READING_DATE"""
    df_WF = pd.read_sql(sql_wr, con=pvoptim_connection, params=(day,end))
    df_WF['OPEN_WEATHER']=df_WF['OPEN_WEATHER'].apply(json.loads)
    df_WF=pd.concat([df_WF.drop(['OPEN_WEATHER'], axis=1), df_WF['OPEN_WEATHER'].apply(pd.Series)], axis=1)
    dfmf=prelWR(df_WF)
    dfmf.drop(['pop'], axis=1, inplace=True)
    colsf=dfmf.columns.tolist()
    for col in weather_attributes:
        if col not in colsf:
            dfmf[col]=0
    dfmf=dfmf[weather_attributes]
    
    '''Get stormweather forecast'''
    df_SW=get_storm_weather(day, end)
    dfmf=pd.merge(dfmf, df_SW, how='inner', on ='reading_date')
    dfmf.drop_duplicates(subset='reading_date', keep="last", inplace=True)
    dfmf['reading_date']=pd.DatetimeIndex(dfmf['reading_date'],tz=tz, ambiguous='NaT')
    dfmf=dfmf.loc[dfmf['reading_date'].notnull()].reset_index()
    #set time to continuous 15 min interval
    start=dfmf['reading_date'].min()
    end=dfmf['reading_date'].max()
    time = pd.date_range(start=start, end=end, freq='15min', tz = 'Europe/Bucharest').to_frame(name='reading_date')
    dfmf=dfmf.merge(time, on='reading_date', how='right')
    
    #interpolate weather variables
    colsn=list(dfmf.select_dtypes(include=[np.number]).columns.values)
    for col in colsn:
        dfmf[col].interpolate(method='linear', limit_direction='both' ,inplace=True)
    colsn=list(dfmf.select_dtypes(exclude=[np.number]).columns.values)
    for col in colsn:
        dfmf[col].interpolate(method='pad', inplace=True)
    return dfmf

def feature_engineering(df, dfmf):
    '''FEATURE ENGINEERING'''
    '''Add combination variables'''
    #train data
    df['time']=df['reading_date'].dt.strftime('%H')
    df['uvi_bin']=(df['uvi']).round(1)
    df['clouds_w']=(df['clouds']+df['cloudCover_noaa'])/2    
    combi_features=['uvi_bin', 'clouds_w']
    df['combination_variables']=df[combi_features].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    
    #forecast data   
    dfmf['time']=dfmf['reading_date'].dt.strftime('%H')
    dfmf.reset_index(inplace=True, drop=True)
    dfmf['uvi_bin']=(dfmf['uvi']).round(1)
    dfmf['clouds_w']=(dfmf['clouds']+dfmf['cloudCover_noaa'])/2
    dfmf['combination_variables']=dfmf[combi_features].apply(lambda row: '_'.join(row.values.astype(str)), axis=1) 
    #dfmf['timeSR_features']=dfmf[timeSR_features].apply(lambda row: '_'.join(row.values.astype(str)), axis=1) 
    
    '''Add aggregated values'''
    #train data
    colg=['clouds_w']
    for cg in colg:
        dfg=df.groupby([cg], as_index=False).agg({'PG_AVG':['max', 'mean', 'min', 'std']})
        dfg.columns = [''.join(col).strip() for col in dfg.columns.values]
        dfg.rename(columns={"PG_AVGmean": cg+"_mean", "PG_AVGmax": cg+"_max","PG_AVGmin": cg+"_min","PG_AVGstd": cg+"_std" }, inplace=True)
        df=pd.merge(df, dfg, on=[cg])
        df[cg+'_range']=df[cg+'_max']-df[cg+'_min']
    
    #forecast data 
    for cg in colg:
        dfmf[cg+'_mean']=0
        dfmf[cg+'_max']=0
        dfmf[cg+'_min']=0
        dfmf[cg+'_std']=0
        for i, row in dfmf.iterrows():
            cg_value=dfmf.iloc[i, :][cg]
            dfmf.iloc[i, -4]=float(df.loc[df[cg]==cg_value, cg+'_mean'].mean())
            dfmf.iloc[i, -3]=float(df.loc[df[cg]==cg_value, cg+'_max'].max())
            dfmf.iloc[i, -2]=float(df.loc[df[cg]==cg_value, cg+'_min'].min())
            dfmf.iloc[i, -1]=float(df.loc[df[cg]==cg_value, cg+'_std'].mean())
        dfmf[cg+'_range']=dfmf[cg+'_max']-dfmf[cg+'_min']    
        dfmf[cg+'_max']=dfmf[cg+'_max'].interpolate(method='linear',limit_direction='both')
        dfmf[cg+'_mean']=dfmf[cg+'_mean'].interpolate(method='linear',limit_direction='both')
        dfmf[cg+'_min']=dfmf[cg+'_min'].interpolate(method='linear',limit_direction='both')
        dfmf[cg+'_std']=dfmf[cg+'_std'].interpolate(method='linear',limit_direction='both')
        vmax=df[cg+'_max'].max()
        imax=df.loc[df[cg+'_max']==vmax, cg].max()
        dfmf.loc[dfmf[cg]>=imax, cg+'_max']=vmax
    return df, dfmf


def prepare_data(df, dfmf, day, cols_selector):
    '''Prepare X,y for ML models'''
    from sklearn.preprocessing import LabelEncoder
    labelEncoder = LabelEncoder()
    cols=df.select_dtypes(exclude=np.number).columns.tolist()
    cols.remove('reading_date')
    for col in cols:
        labelEncoder.fit(list(df[col].astype(str))+list(dfmf[col].astype(str)))
        df[col]= labelEncoder.transform(df[col].astype(str))
        dfmf[col]= labelEncoder.transform(dfmf[col].astype(str))
    # corr = df.corr(method='pearson')
    # corr.sort_values(["PG_AVG"], ascending = False, inplace = True)
    #print(corr.PG_AVG)
    '''Extract X and Y '''
    X=df.loc[df['reading_date']<=day, :].copy()
    X.drop(["PG_AVG"], axis=1, inplace=True)
    X.drop("reading_date", axis=1, inplace=True)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)
    y=df.loc[df['reading_date']<=day, 'PG_AVG']
    #y = np.log(y)
    y.replace([np.inf, -np.inf], np.nan, inplace=True)
    y.fillna(0, inplace=True)
    #y = df['PG_AVG']/df['PG_AVG'].max()
    X_pred=dfmf.copy()
    X_pred.drop("reading_date", axis=1, inplace=True)
    X_pred.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_pred.fillna(0, inplace=True)
    X=X[cols_selector]
    X_pred=X_pred[cols_selector]
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X_pred = scaler.transform(X_pred)
    y=y.to_numpy()
    return X,y, X_pred

def scale_y(y):
    y = np.log(y)
    #y.replace([np.inf, -np.inf], np.nan, inplace=True)
    #y.fillna(0, inplace=True)
    y=np.nan_to_num(y, nan=0, posinf=0, neginf=0)
    return y

def noise_filter(y):
    from scipy.signal import savgol_filter
    y = savgol_filter(y, 21, 3) 
    return y

def train_ML_models(X,y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=.1, shuffle=True)
        
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
    from xgboost import XGBRegressor
    from sklearn import tree
    from sklearn.ensemble import VotingRegressor
    #from lightgbm import LGBMRegressor 
    
    '''Cross validation pentru modele'''
    acc_df = pd.DataFrame(columns = ['Model','Model name','Train Score', 'Test Score', 'R2_Test', 'MAPE'])
    lacc=[]
    models = []
    models.append(("Gradient Boosting:",GradientBoostingRegressor(n_estimators=200, criterion='squared_error',learning_rate=0.05,max_depth=5, loss='squared_error')))
    models.append(("Random Forest:", RandomForestRegressor(n_estimators = 200, criterion='squared_error',max_depth=10)))
    models.append(("XGBoost:", XGBRegressor(n_estimators=200, n_jobs=8, random_state = 10 ,max_depth=5,learning_rate=0.05)))
    #models.append(("LGBMR:", LGBMRegressor(n_estimators=200, n_jobs=8, random_state = 10 ,max_depth=10,learning_rate=0.05)))
    models.append(("HistGBR:", HistGradientBoostingRegressor(random_state = 10 ,max_depth=10,learning_rate=0.09, l2_regularization=0.5, max_iter=150)))
    models.append(("DT:",tree.DecisionTreeRegressor()))
    reg1=GradientBoostingRegressor(n_estimators=200, criterion='squared_error',learning_rate=0.05,max_depth=5, loss='squared_error')
    reg2=RandomForestRegressor(n_estimators = 200, criterion='squared_error',max_depth=10)
    reg3=XGBRegressor(n_estimators=200, n_jobs=8, random_state = 10 ,max_depth=5,learning_rate=0.05)
    reg4= HistGradientBoostingRegressor(random_state = 10 ,max_depth=10,learning_rate=0.09, l2_regularization=0.5, max_iter=150)
    models.append(("VotingR:",VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('xgb', reg3), ('hist', reg4)])))
    
    #train models
    from sklearn.model_selection import KFold, RepeatedKFold
    from sklearn.model_selection import cross_val_score
    for name,model in models:
        rskf = RepeatedKFold(n_splits=2, n_repeats=2)
        for train, test in rskf.split(X_train, y_train):
            xt=X_train[train]
            yt=y_train[train]
            model.fit(xt,yt)
        kfold = KFold(n_splits=5)
        cv_result = cross_val_score(model,X_train,y_train.ravel(), cv = kfold,scoring = "neg_mean_squared_error")
        y_pred= model.predict(X_test)
        lacc.append([model, name,abs(cv_result.mean()),mean_squared_error(y_test, y_pred),r2_score(y_test, y_pred),mean_absolute_percentage_error(y_test, y_pred)*100])
    acc_df=pd.DataFrame (lacc, columns = ['Model','Model name','Train Score', 'Test Score', 'R2_Test', 'MAPE'])
    acc_df.sort_values(by =['Test Score'], ascending = True, inplace = True)
    return acc_df, models


'''MAIN:'''
'''Train on previous 30 days and forecast for the next 10 days'''

try:
    pvoptim_connection = mysql.connector.connect(
      host="localhost", 
      user="pv_optim",
      passwd="pv_optim1234",
      database="pv_optim", port=3306, auth_plugin='mysql_native_password'
    )
except:
    print('App warning! Could not connect to PV-OPTIM DB!')  

'''____________PARAMETERS____________'''
tz = 'Europe/Bucharest'
#day='2022-08-01'
day=dt.datetime.now().strftime('%Y-%m-%d')



df, df_WR,dfm=get_readings(day)
weather_attributes=df_WR.columns.tolist()
dfmf=get_weather_forecast(day, weather_attributes)
df, dfmf=feature_engineering(df, dfmf)
df = df.sort_values(by='reading_date',ignore_index=True)
dfmf = dfmf.sort_values(by='reading_date',ignore_index=True)

# =============================================================================
# '''Save dataframes in csv'''
# cols_selector=['reading_date', 'temp', 'uvi',  'humidity', 'wind_speed',
#                     'cloudCover_noaa','clouds',
#                     'combination_variables'
#                  ,'icon','visibility','pressure','uvIndex_noaa','wind_speed','precipitation_noaa',
#                  'downwardShortWaveRadiationFlux_sg', 'downwardShortWaveRadiationFlux_noaa']
# df[cols_selector].to_csv('weather_readings.csv', index=False)
# dfmf[cols_selector].to_csv('weather_forecast.csv', index=False)
# df[['reading_date','PG_AVG']].to_csv('inverter_readings.csv', index=False)
# =============================================================================
'''Feature selection'''
cols_selector=['time', 'temp', 'uvi',  'humidity', 'wind_speed',
                    'cloudCover_noaa','clouds_w','clouds',
                    'combination_variables'
                 ,'icon','visibility','pressure','uvIndex_noaa','wind_speed','precipitation_noaa',
                 'downwardShortWaveRadiationFlux_sg', 'downwardShortWaveRadiationFlux_noaa']
X, y, X_pred=prepare_data(df, dfmf, day,cols_selector)
y=noise_filter(y)
y=scale_y(y)
'''_____ML_____dfmf'''
acc_df, models=train_ML_models(X,y) 
print(acc_df[['Model name','Train Score', 'Test Score', 'R2_Test', 'MAPE' ]])
cols=['reading_date', 'PG_PRED']
l=acc_df.iloc[0:5, 0:2].values.tolist()
dfmf['PG_PRED']=0
i=1
for  model, name in l:
    y_pred=model.predict(X_pred)
    dfmf['P'+str(i)]=np.exp(y_pred)
    #dfmf['P'+str(i)]=y_pred
    cols.append('P'+str(i))
    dfmf['PG_PRED']=dfmf['PG_PRED']+dfmf['P'+str(i)]
    i=i+1
dfmf['PG_PRED']=dfmf['PG_PRED']/5


# =============================================================================
# '''Apply ML on df'''
# l=acc_df.iloc[0:5, 0:2].values.tolist()
# df['PG_PRED']=0
# i=1
# for  model, name in l:
#     y_pred=model.predict(X)
#     df['P'+str(i)]=np.exp(y_pred)
#     #df['P'+str(i)]=y_pred
#     #cols.append('P'+str(i))
#     df['PG_PRED']=df['PG_PRED']+df['P'+str(i)]
#     i=i+1
# df['PG_PRED']=df['PG_PRED']/5
# =============================================================================

'''-----------Save forecast into DB--------------'''
cols_p=['PG_PRED','P1', 'P2', 'P3','P4', 'P5']
for c in cols_p:
    dfmf[c]= dfmf[c].round(1)
    dfmf.loc[dfmf[c]<0, c]=0
data_test = dfmf[['reading_date', 'PG_PRED','P1', 'P2', 'P3','P4', 'P5']] 
data_test['reading_date']=pd.to_datetime(data_test['reading_date'])
data_test['id_meter']='2060015235'
save_forecast(data_test)
#pvoptim_connection.close()   

# =============================================================================
# '''______Comparisson___________'''
# start_day=day
# end_day=dt.datetime.strptime(start_day, '%Y-%m-%d')+dt.timedelta(days=10)
# sql_ma="""select timestamp_r reading_date,power_gen*1000 P
# from T_INVERTER_READINGS
# where DATE_FORMAT(timestamp_r,'%Y-%m-%d') >=%s and DATE_FORMAT(timestamp_r,'%Y-%m-%d') <=%s
# and id_meter='2060015235'
# ORDER BY READING_DATE"""
# df_ma = pd.read_sql(sql_ma, con=pvoptim_connection, params=(start_day,end_day,))
# df_ma['reading_date']=pd.to_datetime(df_ma['reading_date'], format='%Y-%m-%d %H:%M')
# cols=['reading_date','PG_PRED','P1','P2','P3', 'P4', 'P5']
# dfmf['reading_date']=dfmf['reading_date'].apply(lambda x: x.replace(tzinfo=None))
# df_all = pd.merge(df_ma, dfmf[cols], on='reading_date')
# #metrici
# cols_m=['PG_PRED']
# for c in cols_m:
#     print('Rezultate forecast pe perioada',start_day, ' - ', end_day , 'model:', c)
#     print('Mean error:',mean_absolute_error(df_all['P'],df_all[c])) #MAE
#     print('MAPE:', mean_absolute_percentage_error(df_all['P'],df_all[c])) #MAPE
#     print('RMSE:', mean_squared_error(df_all['P'],df_all[c],  squared=False)) #RMSE
#     print('R2:', r2_score(df_all['P'],df_all[c])) #R2
# 
# #plot
# start=df_all['reading_date'].min()
# end=df_all['reading_date'].max()
# time = pd.date_range(start=start, end=end, freq='D').to_frame(name='reading_date')
# for d in time['reading_date']:
#     fday=d.strftime('%Y-%m-%d')
#     cols=['reading_date','P',  'PG_PRED']
#     data_plot = df_all.loc[(df_all['reading_date'].dt.strftime('%Y-%m-%d')==fday),cols]
#     data_plot.plot(x='reading_date',xlabel='Hour', ylabel='kW', title=fday).legend(loc='upper right')
#     plt.show()
# =============================================================================



