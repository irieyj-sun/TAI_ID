# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 08:54:05 2023

@author: Ghazal Azarfar
"""
# %matplotlib qt 


##############################################################################
#Initialization
##############################################################################
import pandas as pd
import numpy as np
from utility import normalize as normalize
from utility import normalize_categorized as normalize_categorized
from utility import days_between_dates as days_between_dates
from utility import data_stat as data_stat



##############################################################################
# Read Data
##############################################################################
path2data = "D:\\UHN\\Covid19 Vaccination\\"
fname = 'REDCap Extraction July17 2023.xlsx'
df = pd.read_excel(path2data + fname, header = [0,1])

##############################################################################
# Exclude Patients
##############################################################################
#Exclude_patients = [396,478,841]
#df.drop(labels = Exclude_patients, axis = 0, inplace = True)

data = np.zeros((len(df),84)) # define an input variable for the predictive model
##############################################################################
# Patients Characteristics
##############################################################################
data[:,0] = normalize_categorized(df['Characteristics','Data Access Group'])
data[:,1]  = normalize(df['Characteristics','Age'])
data[:,2] = normalize(df['Characteristics','Height (cm)'])
data[:,3]  = normalize(df['Characteristics','Weight (kg)'])
data[:,4]  = normalize(df['Characteristics','BMI'])
df.loc[df['Characteristics','Sex'].isna()]['Characteristics','Sex']= 'MyNaN'
data[:,5]= normalize_categorized(df['Characteristics','Sex'])


##############################################################################
# Organ Transplanted
##############################################################################
Organ_Transplanted = ['Lung','Heart','Liver','Kidney','Pancreas','Stem Cell','Islet cell']

indx = 6
for i in range(0,len(Organ_Transplanted)):
    column = 'Organ Transplanted (check all that apply) (choice=' + Organ_Transplanted[i] + ')'
    df.loc[df['Characteristics',column].isna(), [['Characteristics',column]]]= 'MyNaN'
    data[:,indx] = normalize_categorized(df['Characteristics',column])
    indx = indx + 1


##############################################################################
#Transplant Info
##############################################################################
column = ['Re-transplant?','Transplant Induction ','Drug of induction','Treatment for rejection (in the past 3 months)',
          'Does the patient have chronic kidney disease (defined as eGFR < 30)',
          'Does the patient have any other immunosuppressive conditions (i.e. HIV, concurrent chemotherapy, etc)',
          'Type of HSCT transplant  (choice=Allogeneic- matched sibling)',
          'Type of HSCT transplant  (choice=Allogeneic- matched unrelated)',
          'Type of HSCT transplant  (choice=Allogeneic- haploidentical)',
           'Type of HSCT transplant  (choice=Allogeneic- mismatched (non-haplo))',
           'Type of HSCT transplant  (choice=Autologous)']
for i in range(0,len(column)):
    df.loc[df['Characteristics',column[i]].isna(), [['Characteristics',column[i]]]]= 'MyNaN'
    data[:,indx] = normalize_categorized(df['Characteristics',column[i]])
    indx = indx+1


##############################################################################
#GVHD Info
##############################################################################
GVHD = ['Anti-thymocyte globulin','Post-transplant cyclophosphamide','Cyclosporine','Tacrolimus','MMF','Sirolimus','Other']
for i in range(0,len(GVHD)):
    column = 'GVHD Prophylaxis (choice=' + GVHD[i] + ')'
    df.loc[df['Characteristics',column].isna(),[['Characteristics',column]]]= 'MyNaN'
    data[:,indx] = normalize_categorized(df['Characteristics',column])
    indx = indx + 1


##############################################################################
#Vaccine Info
##############################################################################
vaccine_dose =  ['Vaccine administered first dose', 'Vaccine administered second dose', 'Vaccine administered ', 'Vaccine administered ' ]
vaccine_header = ['Characteristics','Characteristics', 'Third Dose Information','Fourth Dose' ]
for i in range(0,len(vaccine_dose)):
    df.loc[df[vaccine_header[i], vaccine_dose[i]].isna(),[[vaccine_header[i], vaccine_dose[i]]]]= 'MyNaN'
    tmp = df[vaccine_header[i], vaccine_dose[i]]
    tmp = tmp.replace('MyNaN', 0) 
    tmp = tmp.replace('UNK', 0)
    tmp = tmp.replace('Moderna', 1) 
    tmp = tmp.replace('Pfizer', 2) 
    tmp = tmp.replace('Astra Zeneca', 3) 
    data[:,indx] = tmp.values
    indx = indx + 1


##############################################################################
# Time Diagram
##############################################################################
#Days from Transplant to First Covid dose, and between the the doses

Events = ['Transplant date','Date of COVID-19 Vaccine Dose 1 date' ,'Vaccine dose 2 date', 'Date of third dose\xa0', 'Date of fourth dose\xa0']
Header_row = ['Characteristics','Characteristics','Characteristics', 'Third Dose Information','Fourth Dose' ]
Vaccination_Event_Time_Interval = ['Tran to first dose days','first to second dose days', 'second_to_third_dose_days' , 'third_to_fourth_dose_days']
 


for i in range(0, len(Vaccination_Event_Time_Interval)):
    day_column_name = ['Vaccination Event Time Interval',Vaccination_Event_Time_Interval[i]]
    df[day_column_name[0],day_column_name[1]] = None
    header_date1 = [Header_row[i],Events[i]]
    header_date2 = [Header_row[i+1],Events[i+1]]
    df, count = days_between_dates(df, header_date1, header_date2, day_column_name)
    day_min, day_max, day_mean, day_var, day_filled = data_stat(df['Vaccination Event Time Interval',Vaccination_Event_Time_Interval[i]], 1,Vaccination_Event_Time_Interval[i])
    df['Vaccination Event Time Interval',Vaccination_Event_Time_Interval[i]].fillna(value=np.nan, inplace=True)
    mydate = []
    for j in range(0,len(df)):
        mydate = np.append(mydate,df['Vaccination Event Time Interval',Vaccination_Event_Time_Interval[i]][j])
    mydate[np.isnan(mydate)] = 0
    mydate[mydate < 0] = 0
    minval = np.min(mydate[np.nonzero(mydate)])
    maxval = np.max(mydate)
    mydate = (mydate - minval)*0.8/(maxval-minval)+0.1
    data[:,indx]  = mydate
    indx = indx + 1

##############################################################################
# Visit first Info
##############################################################################
df.loc[df['Visit 1 First Dose','Hospitalizations'].isna(), [['Visit 1 First Dose','Hospitalizations']]]= 'MyNaN'
data[:,indx]  = normalize_categorized(df['Visit 1 First Dose','Hospitalizations'])
indx = indx + 1

df.loc[df['Visit 1 First Dose','Documented COVID-19 infection'].isna(), [['Visit 1 First Dose','Documented COVID-19 infection']]]= 'MyNaN'
data[:,indx] = normalize_categorized(df['Visit 1 First Dose','Documented COVID-19 infection'])
indx = indx + 1

df.loc[df['Visit 1 First Dose','Rejection since last visit\xa0'].isna(), [['Visit 1 First Dose','Rejection since last visit\xa0']]]= 'MyNaN'
data[:,indx]  = normalize_categorized(df['Visit 1 First Dose','Rejection since last visit\xa0'])
indx = indx + 1

headr = 'Visit 1 First Dose'
Medication = ['Prednisone','Cyclosporin','Tacrolimus','Sirolimus','Azathioprine','Mycophenolate mofetil or mycophenolate sodium']
for i in range(0,len(Medication)):
    df.loc[df[headr,Medication[i]].isna(), [[headr,Medication[i]]]]= 'MyNaN'
    data[:,indx]  = normalize_categorized(df[headr,Medication[i]])
    indx = indx + 1


##############################################################################
# VISIT VX/VY Info
##############################################################################
df.loc[df['VISIT VX/VY','Any changes in immunosuppression medication since last visit?'].isna(), [['VISIT VX/VY','Any changes in immunosuppression medication since last visit?']]]= 'MyNaN'
data[:,indx] = normalize_categorized(df['VISIT VX/VY','Any changes in immunosuppression medication since last visit?'])
indx = indx + 1

df.loc[df['VISIT VX/VY','Hospitalizations'].isna(), [['VISIT VX/VY','Hospitalizations']]]= 'MyNaN'
data[:,indx]= normalize_categorized(df['VISIT VX/VY','Hospitalizations'])
indx = indx + 1

headr = 'VISIT VX/VY'
Medication = ['Prednisone','Cyclosporin','Tacrolimus','Sirolimus\xa0','Azathioprine','Mycophenolate mofetil or mycophenolate sodium']
for i in range(0,len(Medication)):
    df.loc[df[headr,Medication[i]].isna(), [[headr,Medication[i]]]]= 'MyNaN'
    data[:,indx]  = normalize_categorized(df[headr,Medication[i]])
    indx = indx + 1

##############################################################################
# Visit 2 Second Dose
##############################################################################

df.loc[df['Visit 2 Second Dose', 'Any changes in immunosuppression medication since last visit?'].isna(), [['Visit 2 Second Dose', 'Any changes in immunosuppression medication since last visit?']]]= 'MyNaN'
data[:,indx] = normalize_categorized(df['Visit 2 Second Dose', 'Any changes in immunosuppression medication since last visit?'])
indx = indx + 1

df.loc[df['Visit 2 Second Dose', 'Hospitalizations'].isna(), [['Visit 2 Second Dose', 'Hospitalizations']]]= 'MyNaN'
data[:,indx] = normalize_categorized(df['Visit 2 Second Dose', 'Hospitalizations'])
indx = indx + 1

df.loc[df['Visit 2 Second Dose', 'Documented COVID-19 infection'].isna(), [['Visit 2 Second Dose', 'Documented COVID-19 infection']]]= 'MyNaN'
data[:,indx] = normalize_categorized(df['Visit 2 Second Dose', 'Documented COVID-19 infection'])
indx = indx + 1


headr = 'Visit 2 Second Dose'
Medication = ['Prednisone','Cyclosporin','Tacrolimus','Sirolimus\xa0','Azathioprine','Mycophenolate mofetil or mycophenolate sodium']
for i in range(0,len(Medication)):
    df.loc[df[headr,Medication[i]].isna(), [[headr,Medication[i]]]]= 'MyNaN'
    data[:,indx]  = normalize_categorized(df[headr,Medication[i]])
    indx = indx + 1


##############################################################################
# COVID Information
##############################################################################
df.loc[df['COVID Information', 'Did patient contract COVID-19 at any time during the study?'].isna(), [['COVID Information', 'Did patient contract COVID-19 at any time during the study?']]]= 'MyNaN'
data[:,indx] = normalize_categorized(df['COVID Information', 'Did patient contract COVID-19 at any time during the study?'])
indx = indx + 1



df.loc[df['COVID Information', 'post COVID Antibody results U/ml'].isna(), ('COVID Information', 'post COVID Antibody results U/ml')]= 0
data[:,indx] = df['COVID Information', 'post COVID Antibody results U/ml']
indx = indx + 1
##############################################################################
# Third Dose Information
##############################################################################

df.loc[df['Third Dose Information', 'Any changes in immunosuppression medication since last visit?'].isna(), [['Third Dose Information', 'Any changes in immunosuppression medication since last visit?']]]= 'MyNaN'
data[:,indx] = normalize_categorized(df['Third Dose Information', 'Any changes in immunosuppression medication since last visit?'])
indx = indx + 1



headr = 'Third Dose Information'
Medication = ['Prednisone','Cyclosporin','Tacrolimus','Sirolimus\xa0','Azathioprine','Mycophenolate mofetil or mycophenolate sodium']
for i in range(0,len(Medication)):
    df.loc[df[headr,Medication[i]].isna(), [[headr,Medication[i]]]]= 'MyNaN'
    data[:,indx]  = normalize_categorized(df[headr,Medication[i]])
    indx = indx + 1


##############################################################################
# 6 Month Visit
##############################################################################

df.loc[df['6M Visit', 'Any changes in immunosuppression medication since last visit?'].isna(), [['6M Visit', 'Any changes in immunosuppression medication since last visit?']]]= 'MyNaN'
data[:,indx] = normalize_categorized(df['6M Visit', 'Any changes in immunosuppression medication since last visit?'])
indx = indx + 1

df.loc[df['6M Visit','Hospitalizations'].isna(), [['6M Visit','Hospitalizations']]]= 'MyNaN'
data[:,indx]  = normalize_categorized(df['6M Visit','Hospitalizations'])
indx = indx + 1

df.loc[df['6M Visit','Rejection since last visit\xa0'].isna(), [['6M Visit','Rejection since last visit\xa0']]]= 'MyNaN'
data[:,indx]  = normalize_categorized(df['6M Visit','Rejection since last visit\xa0'])
indx = indx + 1

df.loc[df['6M Visit','Documented COVID-19 infection'].isna(), [['6M Visit','Documented COVID-19 infection']]]= 'MyNaN'
data[:,indx] = normalize_categorized(df['6M Visit','Documented COVID-19 infection'])
indx = indx + 1



headr = '6M Visit'
Medication = ['Prednisone','Cyclosporin','Tacrolimus','Sirolimus','Azathioprine','Mycophenolate mofetil or mycophenolate sodium']
for i in range(0,len(Medication)):
    df.loc[df[headr,Medication[i]].isna(), [[headr,Medication[i]]]]= 'MyNaN'
    data[:,indx]  = normalize_categorized(df[headr,Medication[i]])
    indx = indx + 1


##############################################################################
# Labels
##############################################################################
Header = 'Antibody Information'
columns = ['Post-third dose Antibody results (U/ml)','Post-fourth dose Antibody results (U/ml)']


df[Header,'Immune'] = None


df.loc[df[Header,columns[0]] == '<0.400', (Header,columns[0])] = 0.4
df.loc[df[Header,columns[0]] == '<0.4', (Header,columns[0])] = 0.4


df.loc[df[Header,columns[1]] == '<0.400', (Header,columns[1])] = 0.4
df.loc[df[Header,columns[1]] == '<0.4', (Header,columns[1])] = 0.4

df.loc[df[Header,columns[0]].isna(), (Header,columns[0])]= 0
df.loc[df[Header,columns[1]].isna(), (Header,columns[1])]= 0
df[Header,'Immune'] = df[Header, columns[0]]



df.loc[df[Header,'Immune'] < df[Header,columns[1]], (Header,'Immune')] = df.loc[df[Header,'Immune'] < df[Header,columns[1]], (Header,columns[1])]

labels = df[Header,'Immune'].values

labels[labels < 80 ] = 0
labels[labels >= 80 ] = 1


##############################################################################
# Saving data to a .npz file
##############################################################################
path2save = 'D:\\UHN\\Covid19 Vaccination\\'
fname = 'SOT_COVID_Data_3.npz'
X = [data[0,:]]
y = [labels[0]]
for i in range(1,len(labels)):
    X = np.append(X,[data[i,:]], axis=0)
    y = np.append(y,[labels[i]])
    

np.savez(path2save + fname, X = X, y = y)


