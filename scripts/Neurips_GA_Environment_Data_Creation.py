
import pandas_datareader as web
import pandas as pd
import numpy as np
import datetime as dt
import math
import warnings
import random as rnd
from datetime import datetime
from datetime import timedelta
from genetic_algo_package import geneticalgorithm as ga
import sys, os

#Configure ETF and Date
ETF = 'VTI'
END_DATE = '2019-12-31'

# Disable print function
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore print function
def enablePrint():
    sys.stdout = sys.__stdout__



#Download data from Yahoo Finance
stock_df = web.DataReader(ETF,'yahoo',start = '2000-1-1',end = END_DATE)


#Feature engineering that will be used later
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


# Create funtions
list_of_dates = stock_df.iloc[:-30].index.unique()

#empty dfs
envr = []
sclr =[]
data_compiled = []


#Create 3,000 episodes and identify optimal action for each
for i in range(3000):
    blockPrint()
    def environment_sampler(inputlist=list_of_dates, lengthsample=30):
        '''
        Sampler to take list of dates and return sample environment of a specific length
        '''
        sample_date = rnd.choice(inputlist)
        environment_sample = stock_df[(stock_df.index >= sample_date)].iloc[:30]
        environment_scaler = stock_df[(stock_df.index < sample_date)].iloc[-30:]
        return environment_sample, environment_scaler
        
    # Create buy functions 
    def buy_1():
        return 0
    def buy_2():
        return 2

    action_list = [buy_1(),buy_2()]
    environment, environment_scaler = environment_sampler()

    #Define loss function to solve each episode
    def fitness(gene,env_price = environment['avg_price'].values):
        mp_env = np.mean(env_price)
        len_gene = np.sum(gene)
        pdt_gene_price = np.dot(gene,env_price)
        mkt_return = 10/1200
        size_gene = 30
        return ((pdt_gene_price/len_gene) - mp_env)*(2*len_gene) + (mp_env*(mkt_return)*((size_gene-2*len_gene)/size_gene))

    #configure parameters of genetic algorithm
    algorithm_param = {'max_num_iteration': 200,                  
    'population_size':100,                 
    'mutation_probability':0.2,               
        'elit_ratio': 0.3,                 
            'crossover_probability': 0.4,           
                    'parents_portion': 0.3,                
                        'crossover_type':'uniform',              
                            'max_iteration_without_improv':30}


    #solve each episode to arrive at optimal action, given the loss function
    model=ga(function=fitness,dimension=30,variable_type='bool',algorithm_parameters=algorithm_param)
    env_price = environment['avg_price'].values
    model.run()

    #store optimal action vector
    environment['action'] = model.output_dict['variable']


    def pur_price(row):
        if float(row['action']) > 0:
            return float(row['action'])*float(row['avg_price'])

    #Create columns in dataset for Nueral Network Training
    environment['pur_price'] = environment.apply(lambda row: pur_price(row), axis=1)
    environment['state'] = environment['pur_price'].rolling(30,min_periods = 1).mean()
    environment['sample_id'] = i

    environment_scaler['sample_id'] = i
    environment_scaler['state'] = -9999

    envr.append(environment)
    sclr.append(environment_scaler)
    enablePrint()
    print(i,end ='\n')

data_compiled = pd.concat([pd.concat(envr),pd.concat(sclr)],axis = 0, ignore_index=True)

#Export Data for Neural Network Training
data_compiled.to_csv('Modeling_data_for_'+ETF+'_'+END_DATE+'_.csv')

