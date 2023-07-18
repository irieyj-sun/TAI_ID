# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 12:00:52 2023

@author: Ghazal Azarfar
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
# %matplotlib qt 

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
        df[day_column_name[0],day_column_name[1]][row] = delta
    return df, count
 


path2data = "D:\\UHN\\Covid19 Vaccination\\"
fname = 'REDCap Extraction June 12 2023 with RBD data.xlsx'

#Read Data
df = pd.read_excel(path2data + fname, header = [0,1])


#Age Distribution
age_min, age_max, age_mean, age_var, age_filled = data_stat(df['Characteristics','Age'], 1,'Age')

#Height Distribution
Height_min, Height_max, Height_mean, Height_var, Height_filled = data_stat(df['Characteristics','Height (cm)'], 1,'Height (cm)')
Height_min, Height_max, Height_mean, Height_var, Height_filled = data_stat(df.loc[df['Characteristics','Height (cm)'] > 100]['Characteristics','Height (cm)'], 1,'Height (cm)')

#Weight Distribution
Weight_min, Weight_max, Weight_mean, Weight_var, Weight_filled = data_stat(df['Characteristics','Weight (kg)'], 1,'Weight (kg)')
Weight_min, Weight_max, Weight_mean, Weight_var, Weight_filled = data_stat(df.loc[df['Characteristics','Height (cm)'] > 100]['Characteristics','Weight (kg)'], 1,'Weight (kg)')

#BMI Distribution
BMI_min, BMI_max, BMI_mean, BMI_var, BMI_filled = data_stat(df['Characteristics','BMI'], 1,'BMI')
BMI_min, BMI_max, BMI_mean, BMI_var, BMI_filled = data_stat(df.loc[df['Characteristics','Height (cm)'] > 100]['Characteristics','BMI'], 1,'BMI')


#Data Access Group Distribution
data_pichart(df['Characteristics','Data Access Group'],'Data Access Group')

#Sex Distribution
data_pichart(df['Characteristics','Sex'],'Sex')




#Distribution of Days from Transplant to Enrollement 
day_column_name = ['Characteristics','Tran_to_enrol_days']
df[day_column_name[0],day_column_name[1]] = None

header_date1 = ['Characteristics','Transplant date']
header_date2 = ['Characteristics','Enrolment date']
df, count = days_between_dates(df, header_date1, header_date2, day_column_name)

day_min, day_max, day_mean, day_var, day_filled = data_stat(df['Characteristics','Tran_to_enrol_days'], 90,'SOT day to enrollmen (90 days)')

axis_legends, months_year = days_to_month_year(df['Characteristics','Tran_to_enrol_days'], 'SOD to enrollment(day)')





#classify patients based on organ transplanted
Lung_patients=select_patients_SOD(df, df['Characteristics','Organ Transplanted (check all that apply) (choice=Lung)'])
Heart_patients=select_patients_SOD(df, df['Characteristics','Organ Transplanted (check all that apply) (choice=Heart)'])
Liver_patients=select_patients_SOD(df, df['Characteristics','Organ Transplanted (check all that apply) (choice=Liver)'])
Kidney_patients=select_patients_SOD(df, df['Characteristics','Organ Transplanted (check all that apply) (choice=Kidney)'])
Pancreas_patients=select_patients_SOD(df, df['Characteristics','Organ Transplanted (check all that apply) (choice=Pancreas)'])
Stem_Cell_patients=select_patients_SOD(df, df['Characteristics','Organ Transplanted (check all that apply) (choice=Stem Cell)'])
Islet_Cell_patients=select_patients_SOD(df, df['Characteristics','Organ Transplanted (check all that apply) (choice=Islet cell)'])

plt.figure()
plt.pie([81,1,58,1,120,262,3,14,1,1,163], labels = ['Lung', 'Lung & Heart', 'Heart', 'Heart & Liver', 'Liver', 'Kidney','Kidney & Liver','Kidney & Pancreas','Kidney & Liver & Pancreas','Kidney & Islet_cell','Stem_Cell'])
plt.show() 

column_name = ['Re-transplant?','Transplant Induction ','Drug of induction','Treatment for rejection (in the past 3 months)','Drug for rejection',
               'Does the patient have chronic kidney disease (defined as eGFR < 30)',
               'Does the patient have any other immunosuppressive conditions (i.e. HIV, concurrent chemotherapy, etc)',
               'Type of HSCT transplant  (choice=Allogeneic- matched sibling)',
               'Type of HSCT transplant  (choice=Allogeneic- matched unrelated)',
               'Type of HSCT transplant  (choice=Allogeneic- haploidentical)',
               'Type of HSCT transplant  (choice=Allogeneic- mismatched (non-haplo))',
               'Type of HSCT transplant  (choice=Autologous)',
               #'Indication for HSCT ', 'Other indication','Type of conditioning','Conditioning regimen','Graft Source', 'Graft Manipulation', 'Other '
               ]

for i in range(0,len(column_name)):
    data_pichart(df['Characteristics',column_name[i]],column_name[i])
    print('*************************************')

column_name = ['GVHD Prophylaxis (choice=Anti-thymocyte globulin)','GVHD Prophylaxis (choice=Post-transplant cyclophosphamide)',
               'GVHD Prophylaxis (choice=Cyclosporine)','GVHD Prophylaxis (choice=Tacrolimus)','GVHD Prophylaxis (choice=Tacrolimus)',
               'GVHD Prophylaxis (choice=MMF)','GVHD Prophylaxis (choice=Sirolimus)','GVHD Prophylaxis (choice=Other)',
               'Prior AlloHSCT','Prior Auto HSCT','Vaccine administered first dose']

for i in range(0,len(column_name)):
    data_pichart(df['Characteristics',column_name[i]],column_name[i])
    print('*************************************')
    
    
    
column_name = ['Did the patient receive a different vaccine for the second dose?',
               'Vaccine administered if diff from first',
               'Was pre-vaccine (serum) sample obtained?',
               'Pre-vaccine blood sample date ',
               'Date of COVID-19 Vaccine Dose 1 date']

for i in range(0,len(column_name)):
    data_pichart(df['Characteristics',column_name[i]],column_name[i])
    print('*************************************')
##############################################################################
# Time Diagram
##############################################################################
#Distribution of Days from Transplant to First Covid dose
day_column_name = ['Characteristics','Tran_to_first_dose_days']
df[day_column_name[0],day_column_name[1]] = None

header_date1 = ['Characteristics','Transplant date']
header_date2 = ['Characteristics','Date of COVID-19 Vaccine Dose 1 date']
df, count = days_between_dates(df, header_date1, header_date2, day_column_name)

day_min, day_max, day_mean, day_var, day_filled = data_stat(df['Characteristics','Tran_to_first_dose_days'], 90,'SOT day to first dose (90 days)')

axis_legends, months_year = days_to_month_year(df['Characteristics','Tran_to_first_dose_days'], 'SOD to first_dose(day)')

#Distribution of pre vaccine blood sample to First Covid dose
day_column_name = ['Characteristics','pre_sample_to_first_dose_days']
df[day_column_name[0],day_column_name[1]] = None

header_date1 = ['Characteristics','Pre-vaccine blood sample date ']
header_date2 = ['Characteristics','Date of COVID-19 Vaccine Dose 1 date']
df, count = days_between_dates(df, header_date1, header_date2, day_column_name)

day_min, day_max, day_mean, day_var, day_filled = data_stat(df['Characteristics','pre_sample_to_first_dose_days'], 1,'Pre-vaccine to first dose (days)')


#Distribution of Days from first dose to Vx or Vy sample collection
day_column_name = ['Characteristics','first_to_Vx_days']
df[day_column_name[0],day_column_name[1]] = None

header_date1 = ['Characteristics','Date of COVID-19 Vaccine Dose 1 date']
header_date2 = ['Characteristics','VX or VY Sample collection date']
df, count = days_between_dates(df, header_date1, header_date2, day_column_name)

day_min, day_max, day_mean, day_var, day_filled = data_stat(df['Characteristics','first_to_Vx_days'], 1,'first dose to first visit (days)')

#Distribution of Days from first dose to second dose
day_column_name = ['Characteristics','first_to_second_dose_days']
df[day_column_name[0],day_column_name[1]] = None

header_date1 = ['Characteristics','Date of COVID-19 Vaccine Dose 1 date']
header_date2 = ['Characteristics','Vaccine dose 2 date']
df, count = days_between_dates(df, header_date1, header_date2, day_column_name)

day_min, day_max, day_mean, day_var, day_filled = data_stat(df['Characteristics','first_to_second_dose_days'], 1,'first dose to second dose (days)')

day_min, day_max, day_mean, day_var, day_filled = data_stat(df.loc[df['Characteristics','first_to_second_dose_days'] > 10]['Characteristics','first_to_second_dose_days'], 1,'first dose to second dose (days)')


#Distribution of Days from second dose to second sample collection
day_column_name = ['Characteristics','second_dose_to_second_collection_days']
df[day_column_name[0],day_column_name[1]] = None

header_date1 = ['Characteristics','Vaccine dose 2 date']
header_date2 = ['Characteristics','Post-dose 2 sample collection date']
df, count = days_between_dates(df, header_date1, header_date2, day_column_name)

day_min, day_max, day_mean, day_var, day_filled = data_stat(df['Characteristics','second_dose_to_second_collection_days'], 1,'Second dose to second sample collection(days)')
day_min, day_max, day_mean, day_var, day_filled = data_stat(df.loc[df['Characteristics','first_to_second_dose_days'] > 10]['Characteristics','second_dose_to_second_collection_days'], 1,'Second dose to second sample collection(days)')


