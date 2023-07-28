# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:59:56 2023

@author: azarf
"""
import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split

# feature importance
import time
from feature_names import feature_names
import matplotlib.pyplot as plt

# # Tree Visualisation
# from sklearn.tree import export_graphviz
# from IPython.display import Image, display
# import graphviz



path2data = 'D:\\UHN\\Covid19 Vaccination\\'
fname = 'SOT_COVID_Data_3.npz'
data = np.load(path2data + fname)    

X = data['X']
y = data['y']

X = (X-np.min(X))/(np.max(X)-np.min(X))

# # Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)




param_dist = {'n_estimators': [2,4,5,8,50,100,150,200,250],#randint(50,500),
              'max_depth': [2,4,8,16,32]}#randint(1,20)}

# Create a random forest classifier
rf = RandomForestClassifier()

# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(rf, 
                                 param_distributions = param_dist, 
                                 n_iter=20, 
                                 cv=5)

# Fit the random search object to the data
rand_search.fit(X_train, y_train)


df = pd.concat([pd.DataFrame(rand_search.cv_results_["rank_test_score"], columns=["rank_test_score"]),
           pd.DataFrame(rand_search.cv_results_["params"]),
           pd.DataFrame(rand_search.cv_results_["mean_test_score"], columns=["Accuracy"]),
           pd.DataFrame(rand_search.cv_results_["mean_fit_time"], columns=["mean_fit_time"])],axis=1)

print(df.sort_values("rank_test_score"))

best_rf = rand_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)


# Generate predictions with the best model
y_pred = best_rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

precision = precision_score(y_test, y_pred)
print("Precision:", precision)

recall = recall_score(y_test, y_pred)
print("Recall:", recall)



# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(confusion_matrix=cm).plot();

#feature importance based on mean decrease in impurity



start_time = time.time()
importances = best_rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in best_rf.estimators_], axis=0)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")




forest_importances = pd.Series(importances, index=feature_names)
forest_importances = pd.DataFrame(forest_importances, columns = ['rank'])
forest_importances = forest_importances.sort_values("rank", ascending=False)


fig, ax = plt.subplots()
forest_importances[0:10].plot.bar(yerr=std[0:10], ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()



# # Export the first three decision trees from the forest

# for i in range(3):
#     tree = rf.estimators_[i]
#     dot_data = export_graphviz(tree,
#                                feature_names=X_train.columns,  
#                                filled=True,  
#                                max_depth=2, 
#                                impurity=False, 
#                                proportion=True)
#     graph = graphviz.Source(dot_data)
#     display(graph)