import pandas_datareader as web
import pandas as pd
import numpy as np
import datetime as dt
import math
import warnings
import random as rnd
from datetime import datetime
from datetime import timedelta
import sys, os
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

ETF = str(sys.argv[1])
START_DATE = str(sys.argv[2])
END_DATE = str(sys.argv[3])

# Disable print function
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore print function
def enablePrint():
    sys.stdout = sys.__stdout__

blockPrint()
stock_df = web.DataReader(ETF,'yahoo',start = '2001-1-1',end = END_DATE)
stock_df['avg_price'] = stock_df[['High','Low','Open','Close','Adj Close']].mean(axis=1)
stock_df['avg_price_5ma'] = stock_df['avg_price'].rolling(window=5).mean()
stock_df['avg_price_10ma'] = stock_df['avg_price'].rolling(window=10).mean()
stock_df['avg_price_30ma'] = stock_df['avg_price'].rolling(window=30).mean()
stock_df['avg_price_60ma'] = stock_df['avg_price'].rolling(window=60).mean()
stock_df['avg_price_100ma'] = stock_df['avg_price'].rolling(window=100).mean()
stock_df['avg_price_180ma'] = stock_df['avg_price'].rolling(window=180).mean()
stock_df['avg_price_360ma'] = stock_df['avg_price'].rolling(window=360).mean()

stock_df['price_chg'] = stock_df['avg_price'].pct_change()*100

stock_df['price_chg_5sum'] = stock_df['price_chg'].rolling(window=5).sum()
stock_df['price_chg_10sum'] = stock_df['price_chg'].rolling(window=10).sum()
stock_df['price_chg_30sum'] = stock_df['price_chg'].rolling(window=30).sum()


stock_df = stock_df.copy().drop(['Adj Close','Close','High','Low','Open','Volume'],axis=1)
stock_df = stock_df.copy().dropna()

dates = stock_df.index.unique()
dates_rel = stock_df[stock_df.index>START_DATE].index.unique()



model = load_model('RL_agent_model_Neurips_2020before_vF.h5')

X=[]
df_for_price=[]
period_label = []
st_idx = len(dates) - len(dates[(dates > START_DATE)])
for i in range(math.ceil(len(dates_rel)/30)):
  try:
    loop_df = stock_df[(stock_df.index>=dates[st_idx+30*i]) & (stock_df.index < dates[st_idx+30*(i+1)])].copy()
    # create scaler
    scaler = StandardScaler()
    # fit scaler on data
    # print(sample)
    scaler.fit(stock_df[(stock_df.index>=dates[st_idx-30+30*i]) & (stock_df.index < dates[st_idx+30*(i)])].copy())
    # apply transform
    norm_df = scaler.transform(loop_df)
    X.append(norm_df)
    df_for_price.append(stock_df[(stock_df.index>=dates[st_idx+30*i]) & (stock_df.index < dates[st_idx+30*(i+1)])]['avg_price'].reset_index())
    period_label.append([np.min(stock_df[(stock_df.index>=dates[st_idx+30*i])].index)]*30)
  except:
    pass

X=np.concatenate(X)
predicted_x= np.concatenate(model.predict_classes(X))*1.

df_with_predictions  = pd.DataFrame(np.concatenate(df_for_price))
df_with_predictions['Action'] = predicted_x
df_with_predictions['Period_Start'] = np.concatenate(period_label)
df_with_predictions = df_with_predictions.rename(columns={0:'Date',1:'Price'}).copy()
df_with_predictions = df_with_predictions.set_index('Date').copy()
df_with_predictions['Price'] = pd.to_numeric(df_with_predictions['Price'])

df_summary = df_with_predictions[['Price','Period_Start']].groupby('Period_Start').mean()

df_summary['Agent_Price'] = df_with_predictions[df_with_predictions['Action']==1][['Price','Period_Start']].groupby('Period_Start').mean()
df_summary['Daily_Count'] = 30
df_summary['Agent_Count'] = df_with_predictions[df_with_predictions['Action']==1][['Price','Period_Start']].groupby('Period_Start').count()*2
df_summary = df_summary.rename(columns={'Price':'Daily_Price'}).copy()
df_summary['Agent_Return_Over_Daily'] = (df_summary['Daily_Price']/df_summary['Agent_Price']-1)*100
df_summary['Agent_Count_Over_Daily'] = (df_summary['Agent_Count'] - df_summary['Daily_Count'])
summary_final = df_summary.sort_index()

enablePrint()

print("")
print("")
print("")
print("----------------------------------------------------------------------------")
print("----------------------Results from Validation of Model----------------------")
print("----------------------------------------------------------------------------")
print("")
print(df_summary.sort_index())
print("")
a = round(np.sum(summary_final['Daily_Price'])/len(summary_final),2)
print('Average Daily Price is',a)
b = round(np.sum(summary_final['Agent_Price']*summary_final['Agent_Count'])/np.sum(summary_final['Agent_Count']),2)
print('Average Agent Price is',b)
c = round((a/b-1)*100,2)
print('Total Agent Return over Daily Purchase is',c,'%') 
d = np.sum(summary_final['Agent_Count']) - np.sum(summary_final['Daily_Count'])
print('Total Agent Purchased Count over Daily Purchase Count is',d,'Units')                 
print("")
print("")


# df_with_predictions.to_csv('Agents_actions.csv')
