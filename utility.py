# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 15:30:16 2023

@author: azarf
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time



def normalize(data):
    data_min = data.min()
    data_max = data.max()
    data = (data - data_min)*0.8/(data_max-data_min)+0.1
    data[np.isnan(data)] = 0
    return data

def normalize_categorized(data):
    categories =  data.unique()
    data = data.replace('MyNaN',0) 
    categories = np.delete(categories,np.where(categories == 'MyNaN'))
    new_categories = np.arange(1, len(categories)+1, 1, dtype=float)
    n = 0
    for cat in categories:
        data = data.replace(cat, new_categories[n]) 
        n = n + 1
    return data.values



def data_stat(data, step, xlabel):
    filled_percentage = len(data[data.notna()])/len(data)*100
    data = data[data.notna()]
    data = data.values
    data_min = np.min(data)
    data_max = np.max(data)
    data_mean = np.mean(data)
    data_var = np.var(data)
    mybin = range(int(np.floor(data_min)),int(np.floor(data_max))+2, step)
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.figure()
    plt.hist(data,mybin)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel('Count', fontsize=20)
    plt.xlim((data_min-1, data_max+2)) 
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('min = {:.0f}'.format(data_min) + ', max = {:.0f}'.format(data_max) +', mean = {:.2f}'.format(data_mean) + ', var = {:.2f}'.format(data_var) + ', filled = {:.0f}'.format(filled_percentage) + '%',fontsize=20)
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.show()
    plt.savefig( xlabel +'.png')
    return data_min, data_max, data_mean, data_var,filled_percentage


def data_pichart(data, mytitle):
    filled_percentage = len(data[data.notna()])/len(data)*100
    data = data[data.notna()]
    mylabels =  data.unique()
    y = data.value_counts()
    percentage = y[mylabels]/np.sum(y[mylabels])*100
    plt.figure()
    plt.title(mytitle +', filled = {:.0f}'.format(filled_percentage) + '%')
    plt.pie(y[mylabels], labels = mylabels)
    plt.show() 
    if mytitle == 'Re-transplant?': 
        mytitle = 'Re-transplant'
    elif mytitle == 'Does the patient have chronic kidney disease (defined as eGFR < 30)':
        mytitle = 'Does the patient have chronic kidney disease'
    elif mytitle == 'Did the patient receive a different vaccine for the second dose?':
        mytitle ='Did the patient receive a different vaccine for the second dose'
    elif mytitle == 'Was pre-vaccine (serum) sample obtained?':
        mytitle ='Was pre-vaccine (serum) sample obtained'       
    plt.savefig(mytitle +'.png')
    print(percentage)
    return 0


def days_to_month_year(data, xlabel):
    filled_percentage = len(data[data.notna()])/len(data)*100
    data = data[data.notna()]
    data_min = np.min(data)
    data_max = np.max(data)
    data_mean = np.mean(data)
    data_var = np.var(data)
    data = data.values
    months_year = len(data[np.where((data <= 135))])
    months_year = np.reshape(months_year,(1,1))
    months_year.shape
    #axis_title = ['<3 month','3 month','6 month','9 month','1 year', ]
    #print('[less than ' + str(135) + ' days, or ' + str(135/30) + ' months]')
    for i in range (1,4):
        tmp =  len(data[np.where((data > 90*i+45) & (data <= 90*(i+1)+45))])
        months_year = np.append(months_year, np.reshape(tmp,(1,1)), axis = 0)
        #print('[day interval is: ' + str(90*i+45) + ' to ' + str(90*(i+1)+45) + ']  ' + '[3 month interval is: ' + str((90*i+45)/90) + ' to ' + str((90*(i+1)+45)/90) + ']' )

    tmp = len(data[np.where((data > 90*(3+1)+45) & (data <= 182*(2+1)+90))])
    months_year = np.append(months_year, np.reshape(tmp,(1,1)), axis = 0)
    #print('[day interval is: ' + str(90*(3+1)+45) + ' to ' + str(182*(2+1)+90) + ']  ' + '[6 month interval is: ' + str((90*(3+1)+45)/182) + ' to ' + str((182*(2+1)+90)/182) + ']' )
    for i in range(3,10):
        tmp =  len(data[np.where((data > 182*i+90) & (data <= 182*(i+1)+90))])
        months_year = np.append(months_year,  np.reshape(tmp,(1,1)), axis = 0)
        #print('[day interval is: ' + str(182*i+90) + ' to ' + str(182*(i+1)+90) + ']  ' + '[6 month interval is: ' + str((182*i+90)/182) + ' to ' + str((182*(i+1)+90)/182) + ']' )
    
    tmp =  len(data[np.where((data > 182*(9+1)+90) & (data <= 365*6))])
    months_year = np.append(months_year, np.reshape(tmp,(1,1)), axis = 0) 
    #print('[day interval is: ' + str(182*(9+1)+90) + ' to ' + str(365*6) + ']  ' + '[1 year interval is: ' + str((182*(9+1)+90)/365) + ' to ' + str(365*6/365) + ']' )
    
    for i in range(6,10):
        tmp =  len(data[np.where((data > 365*i) & (data <= 365*(i+1)))])
        months_year = np.append(months_year, np.reshape(tmp,(1,1)), axis = 0)
        #print('[day interval is: ' + str(365*i) + ' to ' + str(365*(i+1)) + ']  ' + '[1 year interval is: ' + str((365*i)/365) + ' to ' + str((365*(i+1))/365) + ']' )
    tmp =  len(data[np.where((data > 365*(9+1)) & (data <= np.max(data)))])
    months_year = np.append(months_year, np.reshape(tmp,(1,1)), axis = 0)

    axis_legends = ['< 3 month', '6 months', '9 months', '1 year', '1.5 years', '2 years', '2.5 years', '3 years', '3.5 years', '4 years', '4.5 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10 years', '> 10 years' ]
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.figure()
    plt.bar(axis_legends,months_year.flatten())
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=20)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel('count', fontsize=20)
    plt.title('min = {:.0f}'.format(data_min) + ', max = {:.0f}'.format(data_max) +', mean = {:.2f}'.format(data_mean) + ', var = {:.2f}'.format(data_var) + ', filled = {:.0f}'.format(filled_percentage) + '%', fontsize=20)
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.savefig( xlabel +'.png')
    return axis_legends, months_year


def select_patients_SOD(data, SOD, label):
    MyBolninx = SOD.isin(['Checked'])
    patients = data.to_dict('list')
    patients = pd.DataFrame(patients, index = MyBolninx)
    patients = patients.loc[True];
    patients= patients.set_index(np.array(range(0,len(patients))))
    return patients

def days_between_dates(df, header_date1, header_date2, day_column_name):
    date_format = "%Y_%m_%d"
    
    date1 = df[header_date1[0],header_date1[1]]
    date2 = df[header_date2[0],header_date2[1]]
    count = 0
    for row in df.loc[(df[header_date2[0],header_date2[1]].notna()) & (df[header_date1[0],header_date1[1]].notna())].index:
        count = count + 1
        dt1 = date1[row].strftime(date_format)
        dt2 = date2[row].strftime(date_format)
        a = time.mktime(time.strptime(dt1, date_format))
        b = time.mktime(time.strptime(dt2, date_format))
        delta = np.array(int((b - a)/86400))
        #df[day_column_name[0],day_column_name[1]][row] = delta
        df.loc[:, (day_column_name[0],day_column_name[1])][row] = delta
    return df, count