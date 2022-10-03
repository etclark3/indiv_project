# Standard imports
import numpy as np
import pandas as pd

# For statistical modeling
import scipy.stats as stats
from math import sqrt 

# For data visualization
import seaborn as sns
import matplotlib.pyplot as plt

# To avoid deprecation and other warnings
import warnings
warnings.filterwarnings('ignore')

# For modeling
# to evaluated performance using rmse
from sklearn.metrics import mean_squared_error
# holt's linear trend model
from statsmodels.tsa.api import Holt
import statsmodels.api as sm

from matplotlib.ticker import StrMethodFormatter
from matplotlib.dates import DateFormatter

# working with dates
from datetime import datetime

import os

#def get_data():

def pred(df, pop, model):
    '''
    Creates predictions based on an pop is the prediction value
    '''
    #print(f'{model}: {pop}')
    yhat_df = pd.DataFrame({'total_population': [pop]},index=df.index)
    return yhat_df.head(2)
    
    
    
def prep_mp(df):
    '''
    This function preps the midyear_population.csv for further exploration by year and modeling
    '''
    df = pd.read_csv('/Users/everettclark/Downloads/archive (3)/midyear_population.csv')
    df.drop(columns='country_code', inplace=True)
    df = pd.pivot_table(df, index='country_name', columns='year',  values='midyear_population').T
    df.reset_index(inplace=True)
    df.year = pd.to_datetime(df.year, format='%Y')
    df.set_index('year', inplace=True)
    df['total_population'] = df.T.sum()
    return df
    
#def :    
    
def verify_split(train, validate, test):    
    for col in train[['total_population']]:
        plt.figure(figsize=(16,9))
        ax = plt.axes()
        plt.plot(train[col], linewidth=5)
        plt.plot(validate[col], linewidth=5)
        plt.plot(test[col], linewidth=5)
        plt.ylabel('World Population')
        plt.xlabel('Year')
        plt.title('World Population: 1950 to 2050')

        plt.text(datetime(1970,1,1), 4500000000, 'Train', fontsize = 25)
        plt.text(datetime(2005,1,1), 7500000000, 'Validate', fontsize = 25)
        plt.text(datetime(2030,1,1), 9000000000, 'Test', fontsize = 25)
        #plt.scatter(x=datetime(1970,1,1),y=800000000, marker='o')
        #plt.legend().remove()
        plt.ticklabel_format(style='plain', axis='y')
        #plt.xlim()
        #plt.ylim()
        ax.set_xticks([datetime(1950,1,1), datetime(1960,1,1), datetime(1970,1,1), datetime(1980,1,1), datetime(1990,1,1), 
                       datetime(2000,1,1), datetime(2010,1,1), datetime(2020,1,1), datetime(2030,1,1), datetime(2040,1,1),
                       datetime(2050,1,1)])

        # setting labels for x tick
        ax.set_xticklabels(['1950','1960','1970','1980','1990','2000','2010','2020','2030','2040','2050'])

        # setting ticks for y-axis
        ax.set_yticks([0, 1000000000, 2000000000, 3000000000, 4000000000, 5000000000, 6000000000, 7000000000, 8000000000
                      , 9000000000, 10000000000])

        # setting labels for y tick
        ax.set_yticklabels(['0','1,000,000,000', '2,000,000,000','3,000,000,000', '4,000,000,000', '5,000,000,000', 
                            '6,000,000,000', '7,000,000,000','8,000,000,000', '9,000,000,000', '10,000,000,000'])
        ax.set_facecolor("lightgrey")
        ax.patch.set_alpha(0.2)
        plt.grid()
        plt.show()
    
    
def total_pop(df): 
    '''
    This function plots the total world population from 1950 to 2050
    '''
    plt.figure(figsize=(16,9))
    ax = plt.axes()
    plt.plot(df.index.to_pydatetime(), df.total_population, color='black',marker="D")
    #plt.ticklabel_format(useOffset=False, style='plain')
    plt.ticklabel_format(style='plain', axis='y')
    plt.xlabel('Year')
    plt.ylabel('Population (Billions)')
    #plt.xlim()
    #plt.ylim()
    # setting ticks for x-axis
    ax.set_xticks([datetime(1950,1,1), datetime(1960,1,1), datetime(1970,1,1), datetime(1980,1,1), datetime(1990,1,1), 
                   datetime(2000,1,1), datetime(2010,1,1), datetime(2020,1,1), datetime(2030,1,1), datetime(2040,1,1),
                   datetime(2050,1,1)])

    # setting labels for x tick
    ax.set_xticklabels(['1950','1960','1970','1980','1990','2000','2010','2020','2030','2040','2050'])

    # setting ticks for y-axis
    ax.set_yticks([0, 1000000000, 2000000000, 3000000000, 4000000000, 5000000000, 6000000000, 7000000000, 8000000000
                  , 9000000000, 10000000000])

    # setting labels for y tick
    ax.set_yticklabels(['0','1,000,000,000', '2,000,000,000','3,000,000,000', '4,000,000,000', '5,000,000,000', 
                        '6,000,000,000', '7,000,000,000','8,000,000,000', '9,000,000,000', '10,000,000,000'])
    ax.set_facecolor("lightgrey")
    ax.patch.set_alpha(0.2)
    plt.title('Population Growth (Real and Estimated): 1950 to 2050')
    plt.grid()
    plt.show()
    
def pop_plot(df):    
    '''
    This function plots the country population from 1950 to 2050 (Top 3 countries labeled)
    '''
    plt.figure(figsize=(16,9))
    ax = plt.axes()
    plt.plot(df.drop(columns='total_population').resample('Y').max())
    plt.text(datetime(1963,1,1), 800000000, 'China', fontsize = 15)
    plt.text(datetime(1963,1,1), 550000000, 'India', fontsize = 15)
    plt.text(datetime(1963,1,1), 210000000, 'US', fontsize = 15)
    #plt.scatter(x=datetime(1970,1,1),y=800000000, marker='o')
    #plt.legend().remove()
    plt.ticklabel_format(style='plain', axis='y')
    plt.xlabel('Year')
    plt.ylabel('Population (Hundreds of Millions)')
    #plt.xlim()
    #plt.ylim()
    ax.set_xticks([datetime(1950,1,1), datetime(1960,1,1), datetime(1970,1,1), datetime(1980,1,1), datetime(1990,1,1), 
                   datetime(2000,1,1), datetime(2010,1,1), datetime(2020,1,1), datetime(2030,1,1), datetime(2040,1,1),
                   datetime(2050,1,1)])
    # setting labels for x tick
    ax.set_xticklabels(['1950','1960','1970','1980','1990','2000','2010','2020','2030','2040','2050'])

    # setting ticks for y-axis
    ax.set_yticks([0, 100000000, 200000000, 300000000, 400000000, 500000000, 600000000, 700000000, 800000000
                  ,900000000, 1000000000, 1100000000, 1200000000, 1300000000, 1400000000, 1500000000, 1600000000])

    # setting labels for y tick
    ax.set_yticklabels(['0','100,000,000', '200,000,000', '300,000,000', '400,000,000', '500,000,000', '600,000,000', 
                        '700,000,000', '800,000,000', '900,000,000', '1,000,000,000', '1,100,000,000', '1,200,000,000',
                        '1,300,000,000', '1,400,000,000', '1,500,000,000', '1,600,000,000'])
    ax.set_facecolor("lightgrey")
    ax.patch.set_alpha(0.2)
    plt.title('Population Growth (Real and Estimated): 1950 to 2050')
    plt.grid()
    plt.show()  
    
#------------------------------------
    
def evaluate(validate, target_var):
    '''
    This function will take the actual values of the target_var from validate, 
    and the predicted values stored in yhat_df, 
    and compute the, rounding to 0 decimal places. 
    it will return the rmse. 
    '''
    rmse = round(sqrt(mean_squared_error(validate[target_var], yhat_df[target_var])), 0)
    return rmse


def plot_eval(train, validate, test, yhat_df, target_var):
    '''
    This function takes in the target var name (string), and returns a plot
    of the values of train for that variable, validate, and the predicted values from yhat_df. 
    it will als lable the rmse. 
    '''
    plt.figure(figsize = (14,6))
    ax = plt.axes()
    plt.plot(train[target_var], label='Train', linewidth=5)
    plt.plot(validate[target_var], label='Validate', linewidth=5)
    plt.plot(yhat_df[target_var], linewidth=5)
    plt.text(datetime(1970,1,1), 4500000000, 'Train', fontsize = 25)
    plt.text(datetime(2005,1,1), 7500000000, 'Validate', fontsize = 25)
    plt.ticklabel_format(style='plain', axis='y')
    ax.set_facecolor("lightgrey")
    ax.patch.set_alpha(0.2)
    plt.title('Population: 1950-2050')
    plt.xlabel('Year')
    plt.ylabel('Population')
    rmse = round(sqrt(mean_squared_error(validate[target_var], yhat_df[target_var])), 0)
    print('\033[1m' + 'RMSE: {:.1f}'.format(rmse) + '\033[0m')
    plt.grid()
    plt.show()
    
    
def val(validate, yhat_df, target_var):
    '''
    This function takes in the target var name (string), and returns a plot
    of the values of train for that variable, validate, and the predicted values from yhat_df. 
    it will als lable the rmse. 
    '''
    plt.figure(figsize = (14,6))
    ax = plt.axes()
    plt.plot(validate[target_var], label='Validate', linewidth=5)
    plt.plot(yhat_df[target_var], linewidth=5)
    plt.text(datetime(2020,1,1), 7000000000, 'Holt (Blue)', fontsize = 25)
    plt.text(datetime(2005,1,1), 7500000000, 'Validate (Orange)', fontsize = 25)
    plt.ticklabel_format(style='plain', axis='y')
    ax.set_facecolor("lightgrey")
    ax.patch.set_alpha(0.2)
    plt.title('Population: 2001-2030')
    plt.xlabel('Year')
    plt.ylabel('Population')
    rmse = round(sqrt(mean_squared_error(validate[target_var], yhat_df[target_var])), 0)
    #print('\033[1m' + 'RMSE: {:.1f}'.format(rmse) + '\033[0m')
    plt.xlim(datetime(2000,1,1),datetime(2030,1,1))
    plt.ylim(6000000000,8500000000)
    plt.grid()
    plt.show()
    
    
# function to store the rmse so that we can compare
def append_eval_df(model_type, target_var):
    '''
    this function takes in as arguments the type of model run, and the name of the target variable. 
    It returns the eval_df with the rmse appended to it for that model and target_var. 
    '''
    rmse = evaluate(target_var)
    d = {'model_type': [model_type], 'target_var': [target_var],
        'rmse': [rmse]}
    d = pd.DataFrame(d)
    return eval_df.append(d, ignore_index = True)


def predictions(validate, amount=None):
    yhat_df = pd.DataFrame({'total_population': [amount]},
                          index=validate.index)
    return yhat_df

def evalu(train, validate, target_var, model_type):
    for col in train.columns:
        #eval_df = pd.DataFrame(columns=['model_type', 'target_var', 'rmse'])
        rmse = round(sqrt(mean_squared_error(validate[target_var], yhat_df[target_var])), 0)
        #return rmse
        d = {'model_type': [model_type], 'target_var': [target_var],'rmse': [rmse]}
        d = pd.DataFrame(d)
        return eval_df.append(d, ignore_index = True)

    
def known(df):
    '''
    Plots country population over known/experienced time (1950-2022)
    '''
    plt.figure(figsize=(16,9))
    ax = plt.axes()
    plt.plot(df[df.index < datetime(2023,1,1)].drop(columns='total_population').resample('Y').max())
    plt.text(datetime(1963,1,1), 800000000, 'China', fontsize = 15)
    plt.text(datetime(1963,1,1), 550000000, 'India', fontsize = 15)
    plt.text(datetime(1963,1,1), 210000000, 'US', fontsize = 15)
    #plt.legend().remove()
    plt.ticklabel_format(style='plain', axis='y')
    plt.xlabel('Year')
    plt.ylabel('Population (Hundreds of Millions)')
    plt.xlim(datetime(1950,1,1), datetime(2023,1,1))
    plt.ylim()
    ax.set_xticks([datetime(1950,1,1), datetime(1960,1,1), datetime(1970,1,1), datetime(1980,1,1), datetime(1990,1,1), 
                   datetime(2000,1,1), datetime(2010,1,1), datetime(2020,1,1)])
    # setting labels for x tick
    ax.set_xticklabels(['1950','1960','1970','1980','1990','2000','2010','2020'])

    # setting ticks for y-axis
    ax.set_yticks([0, 100000000, 200000000, 300000000, 400000000, 500000000, 600000000, 700000000, 800000000
                  ,900000000, 1000000000, 1100000000, 1200000000, 1300000000, 1400000000, 1500000000])

    # setting labels for y tick
    ax.set_yticklabels(['0','100,000,000', '200,000,000', '300,000,000', '400,000,000', '500,000,000', '600,000,000', 
                        '700,000,000', '800,000,000', '900,000,000', '1,000,000,000', '1,100,000,000', '1,200,000,000',
                        '1,300,000,000', '1,400,000,000', '1,500,000,000'])
    ax.set_facecolor("lightgrey")
    ax.patch.set_alpha(0.2)
    plt.title('Population Growth: 1950 to 2022')
    plt.grid()
    plt.show() 
    
    
def exclude(df):
    '''
    Plots all countries population except US, China, and India over experienced time (1950-2022) 
    to better view those countries with more closely related population counts
    '''
    plt.figure(figsize=(16,9))
    ax = plt.axes()
    plt.plot(df[df.index < datetime(2023,1,1)].drop(columns='total_population').resample('Y').max())
    #plt.legend().remove()
    plt.ticklabel_format(style='plain', axis='y')
    plt.xlabel('Year')
    plt.ylabel('Population (Hundreds of Millions)')
    plt.xlim(datetime(1950,1,1), datetime(2023,1,1))
    plt.ylim(0,300000000)
    ax.set_xticks([datetime(1950,1,1), datetime(1960,1,1), datetime(1970,1,1), datetime(1980,1,1), datetime(1990,1,1), 
                   datetime(2000,1,1), datetime(2010,1,1), datetime(2020,1,1)])
    # setting labels for x tick
    ax.set_xticklabels(['1950','1960','1970','1980','1990','2000','2010','2020'])

    # setting ticks for y-axis
    ax.set_yticks([0, 100000000, 200000000, 300000000])

    # setting labels for y tick
    ax.set_yticklabels(['0','100,000,000', '200,000,000', '300,000,000'])
    ax.set_facecolor("lightgrey")
    ax.patch.set_alpha(0.2)
    plt.title('Country Population Growth (Exc. China, India, US): 1950 to 2022')
    plt.grid()
    plt.show() 
    
def plot_all(train, validate, test, yhat_df, target_var):
    plt.figure(figsize = (16,7))
    ax = plt.axes()
    plt.plot(train[target_var], label='train', linewidth=5)
    plt.plot(validate[target_var], label='validate', linewidth=5)
    plt.plot(test[target_var], label='test', linewidth=5)
    plt.plot(yhat_df[target_var], linewidth=5)
    plt.text(datetime(2033,1,1), 9400000000, 'Holt', fontsize = 25)
    plt.text(datetime(2044,1,1), 8400000000, 'Test', fontsize = 25)
    plt.ticklabel_format(style='plain', axis='y')
    ax.set_facecolor("lightgrey")
    ax.patch.set_alpha(0.2)
    plt.title('Population: 1950-2050')
    plt.xlabel('Year')
    plt.ylabel('World Population')
    #plt.xlim()
    #plt.ylim()
    plt.grid()
    plt.show()
    

    
def test_plot(test, yhat_df, target_var):
    plt.figure(figsize = (14,6))
    ax = plt.axes()
    #plt.plot(train[target_var], label='Train', linewidth=5)
    plt.plot(test[target_var], label='test', linewidth=5)
    plt.plot(yhat_df[target_var], linewidth=5)
    plt.text(datetime(2038,1,1), 9400000000, 'Holt', fontsize = 25)
    plt.text(datetime(2044,1,1), 8800000000, 'Test', fontsize = 25)
    plt.ticklabel_format(style='plain', axis='y')
    ax.set_facecolor("lightgrey")
    ax.patch.set_alpha(0.2)
    plt.title('Population (Test): 2031-2050')
    plt.xlabel('Year')
    plt.ylabel('Population')
    #rmse = round(sqrt(mean_squared_error(validate[target_var], yhat_df[target_var])), 0)
    #print('\033[1m' + 'RMSE: {:.1f}'.format(rmse) + '\033[0m')
    plt.xlim(datetime(2031,1,1),datetime(2050,1,1))
    #plt.ylim(6000000000,8500000000)
    plt.grid()
    plt.show()
    
def rmse(df, yhat_df, target_var):
    rmse = round(sqrt(mean_squared_error(df[target_var], yhat_df[target_var])), 0)
    return rmse

def growth(df, y, w):
    plt.figure(figsize=(16,9))
    ax = plt.axes()
    plt.plot(df.index.to_pydatetime(), df.total_population, color='black',marker="D")
    plt.plot(y, linewidth=4, color='orange')
    plt.plot(w, linewidth=4, color='purple')
    #plt.ticklabel_format(useOffset=False, style='plain')
    plt.text(datetime(1960,1,1), 5500000000, 'Average Increase 1950-2022', fontsize = 15)
    plt.text(datetime(1960,1,1), 5100000000, 'Known time (73yrs)', fontsize = 15)
    plt.text(datetime(2010,1,1), 6000000000, 'Average Increase 1950-2050', fontsize = 15)
    plt.text(datetime(2010,1,1), 5600000000, 'Estimated (101yrs)', fontsize = 15)
    plt.ticklabel_format(style='plain', axis='y')
    plt.xlabel('Year')
    plt.ylabel('Population (Billions)')
    plt.xlim(datetime(1949,1,1),datetime(2051,1,1))
    #plt.ylim()
    # setting ticks for x-axis
    ax.set_xticks([datetime(1950,1,1), datetime(1960,1,1), datetime(1970,1,1), datetime(1980,1,1), datetime(1990,1,1), 
                   datetime(2000,1,1), datetime(2010,1,1), datetime(2020,1,1), datetime(2030,1,1), datetime(2040,1,1),
                   datetime(2050,1,1)])

    # setting labels for x tick
    ax.set_xticklabels(['1950','1960','1970','1980','1990','2000','2010','2020','2030','2040','2050'])

    # setting ticks for y-axis
    ax.set_yticks([2000000000, 3000000000, 4000000000, 5000000000, 6000000000, 7000000000, 8000000000
                  , 9000000000, 10000000000])

    # setting labels for y tick
    ax.set_yticklabels(['2,000,000,000','3,000,000,000', '4,000,000,000', '5,000,000,000', 
                        '6,000,000,000', '7,000,000,000','8,000,000,000', '9,000,000,000', '10,000,000,000'])
    ax.set_facecolor("lightgrey")
    ax.patch.set_alpha(0.2)
    plt.title('Population Growth (Real and Estimated): 1950 to 2050')
    plt.grid()
    plt.show()