"""
Serena Zhang
Nov 5th
"""

"""
This script calls functions from modules and completes a whole run for the google amazon dataset
"""

## Preprocess any specific columns (e.g. price column for differnt currency)
## Then get the three types of matrices for both google and amazon data (6 matrices in total)
## Call similarity functions on 3 pairs of matrices
## Concatenate previous results to form the final dataset for modeling
## Call modeling functions (train test split etc)

import sys
sys.path.append('..')
sys.path.append('/anaconda/lib/python3.6/site-packages')
import pandas as pd
import numpy as np
from modules.preprocessing import Preprocessing
from modules.preprocessing.generate_labels import gen_labels
from modules.feature_generation.gen_similarities import similarities
from sklearn.model_selection import train_test_split

'''
read data
'''
df1 = pd.read_csv('data/companies_data_neoway_subsample/reference.csv')
df2 = pd.read_csv('data/companies_data_neoway_subsample/input.csv')
match_df = pd.read_csv('data/companies_data_neoway_subsample/match.csv')

'''
specify id names
'''
df1_id = 'serial'
df2_id = 'serial'
match_id1 = 'serial_reference' #corresponds to df1_id
match_id2 = 'serial_input' #corresponds to df2_id

'''
id column manipulation
'''
# save for later use to generate labels
df1_id_col = df1[df1_id]
df2_id_col = df2[df2_id]

# drop id columns because we don't need to compute id similarity
df1 = df1.drop(columns = [df1_id])
df2 = df2.drop(columns = [df2_id])

'''
preprocess both dataframes
'''
processed_data = Preprocessing().overall_preprocess(df1, df2) # may take a while bc loading pretrained word embedding model

'''
get numerical data
'''

num_matrix_1, num_matrix_2 = processed_data["numerical"][0],processed_data["numerical"][1]
embed_matrix_1, embed_matrix_2 = processed_data["word_embedding_fields"][0],processed_data["word_embedding_fields"][1]
spc_matrix_1, spc_matrix_2 = processed_data["special_fields"][0],processed_data["special_fields"][1]

'''
calculate similarities
'''
num_final_data = similarities().numerical_similarity_on_matrix(num_matrix_1,num_matrix_2)
embed_final_data = similarities().vector_similarity_on_matrix(embed_matrix_1,embed_matrix_2)
spc_final_data = similarities().text_similarity_on_matrix(spc_matrix_1,spc_matrix_2)


'''
concatenate all data
'''
# only concatenate non-empty similarity matrices
non_empty = []

for m in num_final_data, embed_final_data, spc_final_data:
    if m.size !=0:
        non_empty.append(m)

x = np.concatenate([i for i in non_empty], axis = 1)

print(x.shape)

'''
train test split
'''

# generate y labels
y = gen_labels(df1_id_col, df2_id_col, match_df, match_id1, match_id2)

# simple check to see if x and y match in size
print(y.shape[0] == x.shape[0])
print(y.sum() == match_df.shape[0])


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, stratify = y)

'''
modeling
'''
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve,  precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV

# #upsample
# x_maj = x_train[y_train==0]
# x_min = x_train[y_train==1]
# x_min_upsampled = resample(x_min,n_samples=x_maj.shape[0],random_state=42)
# x_train_new = np.vstack((x_maj, x_min_upsampled))
# y_train_new = np.hstack((np.zeros(x_maj.shape[0]), np.ones(x_maj.shape[0])))

# # CV
# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
# print(random_grid)
# # Use the random grid to search for best hyperparameters
# rf = RandomForestClassifier()
# # Random search of parameters and use all available cores
# rf_random = RandomizedSearchCV(estimator=rf,
#                                param_distributions=random_grid,
#                                n_iter=100,
#                                cv=3, verbose=2, random_state=42,
#                                n_jobs=-1, scoring='f1')
# rf_random.fit(x_train_new, y_train_new)
# print(rf_random.best_params_)

# fit
rf_random = RandomForestClassifier(random_state=42)
rf_random.fit(x_train, y_train)
# predict
y_pred_rf = rf_random.predict(x_test)
y_pred_prob_rf = rf_random.predict_proba(x_test)[:, 1]
# roc curve
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_prob_rf)
# precision, recall, f1
print("\tPrecision: %1.3f" % precision_score(y_test, y_pred_rf))
print("\tRecall: %1.3f" % recall_score(y_test, y_pred_rf))
print("\tF1: %1.3f" % f1_score(y_test, y_pred_rf))
print("\tAccuracy: {}".format(sum(y_pred_rf==y_test)/len(y_test)))

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
