# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:18:01 2024

@author: azarf
"""
import pandas as pd
import random as rn
import numpy as np
from LSTM_wtih_DynamicRouting import LSTM_DyanRout_model
from My_LSTM import My_LSTM
from feature_names_longformat import feature_names_longformat

from matplotlib import pyplot as plt

import tensorflow as tf
import altair as alt
alt.renderers.enable('altair_viewer')
from timeshap.utils import calc_avg_sequence
from timeshap.explainer import local_report,global_report

from feature_names_longformat import plot_feats
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input,Rescaling



class evaluate_model():
    def __init__(self,model,X,y,feature_names):
        self.model = model
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.df = pd.DataFrame(np.reshape(X,(X.shape[0]*X.shape[1],X.shape[2])), columns = feature_names)

        self.df['id']=np.repeat(np.array(range(0,X.shape[0])),(X.shape[1]))
        self.df['label'] = np.repeat(y,(X.shape[1]))
        self.df['timestamp'] = np.tile(np.transpose(np.array(range(0,X.shape[1]))),(X.shape[0]))
        
class TimeShap(evaluate_model):
    def __init__(self,model,X,y,feature_names,random_number,tolerance,top_x_feats,top_x_events, plot_feats):
        super(TimeShap, self).__init__(model,X,y,feature_names)
        self.plot_feats = plot_feats
        self.tolerance = tolerance
        self.top_x_feats = top_x_feats 
        self.top_x_events = top_x_events
        self.random_number= random_number
      
        self.f = lambda x: self.model.predict(x)
        self.average_sequence = calc_avg_sequence(self.df, numerical_feats=self.feature_names, categorical_feats=[],model_features=self.feature_names, entity_col=self.df['id'])

    def Individual(self,row):
        pruning_dict = {'tol': self.tolerance}
        event_dict = {'rs': self.random_number, 'nsamples': self.X.shape[0]}
        feature_dict = {'rs':self.random_number, 'nsamples': self.X.shape[0], 'feature_names': self.feature_names, 'plot_features': self.plot_feats}
        cell_dict = {'rs': self.random_number, 'nsamples': self.X.shape[0], 'top_x_feats': self.top_x_feats, 'top_x_events': self.top_x_events}
        x_data = np.expand_dims(self.X[row,:,:], axis=0)
        cell_level = local_report(self.f, x_data, pruning_dict, event_dict, feature_dict, cell_dict=cell_dict, entity_uuid=row, entity_col='id', baseline=self.average_sequence)
        return cell_level
    
    def group(self,label,path):
        data = self.df.loc[self.df['label'] == label]
        pruning_dict = {'tol': [0.000], 'path': path+'run_all_tf.csv'}
        event_dict = {'path': path+'event_all_tf.csv', 'rs': 2023, 'nsamples': len(data)}
        feature_dict = {'path': path+'feature_all_tf.csv', 'rs': 2023, 'nsamples': len(data), 'feature_names': self.feature_names, 'plot_features': self.plot_feats, }
        prun_stats, global_plot = global_report(self.f, data, pruning_dict, event_dict, feature_dict,self.average_sequence, self.feature_names, list(data.columns) , 'id', 'timestamp', )
        return prun_stats,global_plot
    
    def plot_TimeShap(self,path):
        color = (0.7843137254901961, # red
         0.8784313725490196, # green
         0.7058823529411765, # blue
         1) # transparency  
        fname = 'feature_all_tf.csv'
        TimeShap_value = pd.read_csv(path + fname)
        n = int(len(TimeShap_value)/self.X.shape[2])
        Shap_values = np.reshape(TimeShap_value['Shapley Value'].values,(n,self.X.shape[2]))
        mean_Shap_values = np.mean(np.abs(Shap_values), axis = 0)
        importances = pd.Series(mean_Shap_values, index=feature_names_longformat)
        importances = pd.DataFrame(importances, columns = ['rank'])
        importances = importances.sort_values("rank", ascending=False)
        fig, ax = plt.subplots()

        # Horizontal Bar Plot
        ax.barh(importances.index[0:self.top_x_feats],importances['rank'][0:self.top_x_feats].values,color=color,edgecolor =[0,0,0],linewidth=1)
        # Remove axes splines
        for s in ['top','right']: 
            ax.spines[s].set_visible(False)
        # Remove y Ticks
        ax.yaxis.set_ticks_position('none')
        # Show top values
        ax.invert_yaxis()

        # Add Plot Title
        ax.set_title('Feature importance',loc ='left', )
        plt.xlabel('Shap Values')
        # Show Plot
        plt.show()
    
class Shap_Analysis(evaluate_model):
    def __init__(self,model,X,y,feature_names,random_number,tolerance,top_x_feats,top_x_events, plot_feats):
        super(Shap_Analysis, self).__init__(model,X,y,feature_names)

    def plot_TimeShap(self,path):
        

    

#path2data = 'D:\\UHN\\Covid19 Vaccination\\SOT_COVID_Data_Long_20240102.npz'    
#path2model = 'D:\\UHN\\Covid19 Vaccination\\LSTM_DynamicRouting\\12 Months\\regression\\20240108\\'
#model = tf.keras.models.load_model(path2model)
    

#data = np.load(path2data)   

#X_12M = data['X']
#y_12M_class = data['y']
path2data = 'D:\\UHN\\Covid19 Vaccination\\'
#fname = 'SOT_COVID_Data_Long_6M_padded_event_days.npz'
fname = 'SOT_COVID_Data_Long_20240102.npz'

data = np.load(path2data + fname)    



X = data['X']
Antibody = data['y_value']
y = data['y']

time_point = 5
feature_size = X.shape[1]
sample_count = int(X.shape[0]/time_point) 

epoch = 20
pc_num = 20


# myseed = 5000
# np.random.seed(myseed)
# tf.random.set_seed(myseed)
# rn.seed(myseed)
# X,pc_percentage,pca = data_reduc(X, pc_num)

feature = feature_size

X = np.reshape(X,[sample_count,time_point,feature])
Antibody = np.reshape(Antibody,[sample_count,time_point])
y = np.reshape(y,[sample_count,time_point])


p = np.random.default_rng(seed=66).permutation(sample_count)
X = X[p,:,:]
Antibody = Antibody[p,:]
y = y[p,:]


y_12M = np.zeros((X.shape[0],1))
X_12M = np.zeros(np.shape(X))
y_12M_class = np.zeros((X.shape[0],1))


#6M
n = 0
j = 0
for i in range(0,len(Antibody)):
    if np.isinf(Antibody[i,4]):
        j = j + 1
        # if np.isinf(Antibody[i,3]):
        #     j = j + 1
        # else: 
        #     y_12M[n,0] = Antibody[i,3]
        #     X_12M[n,0:4,:] = X[i,0:4,:]
        #     n = n + 1
    else:
        y_12M[n,0] = Antibody[i,4]
        y_12M_class[n,0] = y[i,4]
        X_12M[n,:,:] = X[i,:,:]
        n = n + 1
        

y_12M_class = np.ravel(y_12M_class[0:n,0])
y_12M_class[y_12M_class == 1] = 0
y_12M_class[y_12M_class == 2] = 1
y_12M = np.ravel(y_12M[0:n,0])
X_12M = X_12M[0:n,:,:]


path2save = 'D:\\UHN\\Covid19 Vaccination\\Github_submission\\LSTM_DynamicRounting_weights\\'
model = LSTM_DyanRout_model
model.load_weights(path2save)    
random_number = 42
tolerance = 0
top_x_feats = 20
top_x_events = 5
row = 10
group_label= 1
Explainer = TimeShap(model,X_12M,y_12M_class,feature_names_longformat,random_number,tolerance,top_x_feats,top_x_events, plot_feats)
local_report = Explainer.Individual(row)
path = 'outputs_seed_20240306/'
prun_stats, global_plot = Explainer.group(group_label,path)
alt.data_transformers.disable_max_rows()
global_plot.show()

Explainer.plot_TimeShap(path)
