''''PV-OPTIM Forecast LAYER 1 - use csv templates'''
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
    '''ANTRENARE MODELE'''
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
tz = 'Europe/Bucharest'
df=pd.read_csv('data_folder/inverter_readings.csv')
df_WR=pd.read_csv('data_folder/weather_readings.csv')
df=df.merge(df_WR, on= 'reading_date')
df['reading_date']=pd.to_datetime(df['reading_date'])
dfmf=pd.read_csv('data_folder/weather_forecast.csv')
dfmf['reading_date']=pd.to_datetime(dfmf['reading_date'])
day=dfmf['reading_date'].min().strftime("%Y-%m-%d")

df, dfmf=feature_engineering(df, dfmf)
df = df.sort_values(by='reading_date',ignore_index=True)
dfmf = dfmf.sort_values(by='reading_date',ignore_index=True)
'''Feature selection'''
cols_selector=['time', 'temp', 'uvi',  'humidity', 'wind_speed',
                    'cloudCover_noaa','clouds_w','clouds',
                    'combination_variables'
                 ,'icon','visibility','pressure','uvIndex_noaa','wind_speed','precipitation_noaa',
                 'downwardShortWaveRadiationFlux_sg', 'downwardShortWaveRadiationFlux_noaa']
X, y, X_pred=prepare_data(df, dfmf, day,cols_selector)
y=noise_filter(y)
y=scale_y(y)
'''_____ML_____'''
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


'''-----------Save forecast into csv file--------------'''
cols_p=['PG_PRED','P1', 'P2', 'P3','P4', 'P5']
for c in cols_p:
    dfmf[c]= dfmf[c].round(1)
    dfmf.loc[dfmf[c]<0, c]=0
data_test = dfmf[['reading_date', 'PG_PRED','P1', 'P2', 'P3','P4', 'P5']] 
data_test['reading_date']=pd.to_datetime(data_test['reading_date'])
data_test['id_meter']='2060015235'
data_test.to_csv('data_folder/inverter_forecast.csv',index=False)
  





