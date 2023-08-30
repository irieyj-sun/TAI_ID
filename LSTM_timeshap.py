# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 09:29:30 2023

@author: azarf
"""
from IPython.display import display, HTML

import numpy as np
import pandas as pd
  
import tensorflow as tf  
from feature_names_longformat import feature_names_longformat
from feature_names_longformat import plot_feats

########################################################
#Read Data and model
########################################################


path2data = 'D:\\UHN\\Covid19 Vaccination\\'
fname = 'SOT_COVID_Data_Long_12M.npz'

data = np.load(path2data + fname)    


EPOCHS = 100
X = data['X']
y = data['y']


model_features = feature_names_longformat
time_point = 5
feature_size = X.shape[1]
sample_count = int(X.shape[0]/time_point) 

y = np.reshape(y,[sample_count,time_point])
y_12months = y[:,time_point-1]

df = pd.DataFrame(X, columns = feature_names_longformat)
df['label'] = np.repeat(y_12months,(5))
df['activity'] = None
df.loc[df['label'] == 1, ['activity']] = 'immune'
df.loc[df['label'] == 0, ['activity']] = 'not immune'

df['id'] = np.repeat(np.array(range(0,307)),(5))
df['timestamp'] = np.tile(np.transpose(np.array(range(0,5))),(307))
df['all_id'] =np.array(range(0,307*5))
time_feat = 'timestamp'
label_feat = 'label'
sequence_id_feat = 'id'

X = np.reshape(X,[sample_count,time_point,feature_size])





model = tf.keras.models.load_model('D:\\UHN\\Covid19 Vaccination\\LSTM without PCA\\12 Months\\')




########################################################
#TimeSHAP 
########################################################
import timeshap
#model entry point
f = lambda x: model.predict(x)


#baseline event
from timeshap.utils import calc_avg_event
average_event = calc_avg_event(df, numerical_feats=model_features, categorical_feats=[])

print('average event is:')
average_event

from timeshap.utils import calc_avg_sequence
average_sequence = calc_avg_sequence(df, numerical_feats=model_features, categorical_feats=[],model_features=model_features, entity_col=sequence_id_feat)
print('average sequence is:')
average_sequence

#Average score over baseline
from timeshap.utils import get_avg_score_with_avg_event
avg_score_over_len = get_avg_score_with_avg_event(f, average_sequence, top=20)

########################################################
#TimeSHAP - Local Explanations
########################################################
ids_for_test = np.random.choice(df['id'].unique(), size = 4, replace=False)
positive_sequence_id = np.random.choice(ids_for_test)
pos_x_pd = df.loc[df['id'] == positive_sequence_id]

# select model features only
pos_x_data = pos_x_pd[model_features]
# convert the instance to numpy so TimeSHAP receives it
pos_x_data = np.expand_dims(pos_x_data.to_numpy().copy(), axis=0)



#Local Report on positive instance
from timeshap.explainer import local_report

pruning_dict = {'tol': 0.025}
event_dict = {'rs': 42, 'nsamples': 300}
feature_dict = {'rs': 42, 'nsamples': 300, 'feature_names': model_features, 'plot_features': plot_feats}
cell_dict = {'rs': 42, 'nsamples': 300, 'top_x_feats': 2, 'top_x_events': 2}

local_report(f, pos_x_data, pruning_dict, event_dict, feature_dict, cell_dict=cell_dict, entity_uuid=positive_sequence_id, entity_col='id', baseline=average_sequence)
########################################################
#TimeSHAP - Global Explanations
########################################################
#Explain all
from timeshap.explainer import global_report

pos_dataset = df[df['label'] == 1]


schema = schema = list(pos_dataset.columns)

pruning_dict = {'tol': [0.05, 0.075], 'path': 'outputs/prun_all_tf.csv'}
event_dict = {'path': 'outputs/event_all_tf.csv', 'rs': 42, 'nsamples': 300}
feature_dict = {'path': 'outputs/feature_all_tf.csv', 'rs': 42, 'nsamples': 300, 'feature_names': model_features, 'plot_features': plot_feats,}
prun_stats, global_plot = global_report(f, pos_dataset, pruning_dict, event_dict, feature_dict, average_event, model_features, schema, 'id', time_feat, )
