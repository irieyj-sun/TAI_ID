# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 12:00:52 2023

@author: Ghazal Azarfar
"""

import pandas as pd
import matplotlib.pyplot as plt


# %matplotlib qt 

from utility import data_stat as data_stat
from utility import data_pichart as data_pichart
from utility import days_to_month_year as days_to_month_year
from utility import select_patients_SOD as select_patients_SOD
from utility import days_between_dates as days_between_dates


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

day_min, day_max, day_mean, day_var, day_filled = data_stat(df['Characteristics','Tran_to_first_dose_days'], 1,'SOT day to first dose (90 days)')

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


#Distribution of Days from fist dose to Covid diagnosis

day_column_name = ['COVID Information','days_from_first_dose_to_Covid']
df[day_column_name[0],day_column_name[1]] = None

header_date1 = ['Characteristics','Date of COVID-19 Vaccine Dose 1 date']
header_date2 = ['COVID Information','COVID-19 diagnosis date']

df, count = days_between_dates(df, header_date1, header_date2, day_column_name)

day_min, day_max, day_mean, day_var, day_filled = data_stat(df['COVID Information','days_from_first_dose_to_Covid'], 1,'days_from_first_dose_to_Covid')


#Distribution of Days from second dose to Covid diagnosis
day_column_name = ['COVID Information','days_from_second_dose_to_Covid']
df[day_column_name[0],day_column_name[1]] = None

header_date1 = ['Characteristics','Vaccine dose 2 date']
header_date2 = ['COVID Information','COVID-19 diagnosis date']

df, count = days_between_dates(df, header_date1, header_date2, day_column_name)

day_min, day_max, day_mean, day_var, day_filled = data_stat(df['COVID Information','days_from_second_dose_to_Covid'], 1,'days_from_second_dose_to_Covid')

#Distribution of Days from third dose to Covid diagnosis
day_column_name = ['COVID Information','days_from_third_dose_to_Covid']
df[day_column_name[0],day_column_name[1]] = None

header_date1 = ['Third Dose Information','Date of third dose ']
header_date2 = ['COVID Information','COVID-19 diagnosis date']

df, count = days_between_dates(df, header_date1, header_date2, day_column_name)

day_min, day_max, day_mean, day_var, day_filled = data_stat(df['COVID Information','days_from_third_dose_to_Covid'], 1,'days_from_third_dose_to_Covid')


#Distribution of Days from fourth dose to Covid diagnosis
day_column_name = ['COVID Information','days_from_fourth_dose_to_Covid']
df[day_column_name[0],day_column_name[1]] = None

header_date1 = ['Fourth Dose','Date of fourth dose ']
header_date2 = ['COVID Information','COVID-19 diagnosis date']

df, count = days_between_dates(df, header_date1, header_date2, day_column_name)

day_min, day_max, day_mean, day_var, day_filled = data_stat(df['COVID Information','days_from_fourth_dose_to_Covid'], 1,'days_from_fourth_dose_to_Covid')
