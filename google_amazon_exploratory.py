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
df1 = pd.read_csv("data/amazon_google/sample/amazon_sample.csv")
df2 = pd.read_csv("data/amazon_google/sample/google_sample.csv")
match_df = pd.read_csv("data/amazon_google/sample/amazon_google_sample_match.csv")

'''
specify id names
'''
df1_id = 'id'
df2_id = 'id'
match_id1 = 'idAmazon' # corresponds to df1_id
match_id2 = 'idGoogleBase' # corresponds to df2_id

'''
custom data cleaning, currently this section is for google dataset only
we still need to convert currency. right now just ignoring currency effect
'''
df2["price"] = df2.price.str.replace(r"[a-zA-Z]",'').astype(float)


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
processed_data = Preprocessing().overall_preprocess(df1, df2,
                                                    special_columns=['title','manufacturer'],
                                                    embedding_weight='tfidf',
                                                    word_embedding_model="glove")
 # may take a while bc loading pretrained word embedding model

'''
get data
'''

num_matrix_1,num_matrix_2 = processed_data["numerical"][0],processed_data["numerical"][1]
embed_matrix_1,embed_matrix_2 = processed_data["word_embedding_fields"][0],processed_data["word_embedding_fields"][1]
spc_matrix_1,spc_matrix_2 = processed_data["special_fields"][0],processed_data["special_fields"][1]


'''
calculate similarities
'''

num_final_data = similarities().numerical_similarity_on_matrix(num_matrix_1,num_matrix_2)
embed_tfidf_data = similarities().vector_similarity_on_matrix(embed_matrix_1,embed_matrix_2)
#embed_mean_data = similarities().vector_similarity_on_matrix(embed_matrix_1,embed_matrix_2)
#embed_min_data = similarities().vector_similarity_on_matrix(embed_matrix_1,embed_matrix_2)
#embed_max_data = similarities().vector_similarity_on_matrix(embed_matrix_1,embed_matrix_2)
spc_final_data = similarities().text_similarity_on_matrix(spc_matrix_1,spc_matrix_2)


'''
concatenate all data
'''
# only concatenate non-empty similarity matrices
non_empty = []

for m in num_final_data, spc_final_data, embed_tfidf_data:#, embed_mean_data, embed_max_data, embed_min_data:
    if m.size !=0:
        non_empty.append(m)

x = np.concatenate([i for i in non_empty], axis = 1)

print(x.shape)

'''
train test split
'''
y = gen_labels(df1_id_col, df2_id_col, match_df, 'idAmazon', 'idGoogleBase')

print(y.shape[0] == x.shape[0])

# generate y labels
# y = gen_labels(df1_id_col, df2_id_col, match_df, match_id1, match_id2)

# simple check to see if x and y match in size
# print (y.shape[0] == x.shape[0])
# print(y.sum() == match_df.shape[0])


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, stratify = y, random_state=42)

'''
modeling
'''
col_means = np.nanmean(x_train, axis=0)
inds_train = np.where(np.isnan(x_train))
inds_test = np.where(np.isnan(x_test))
x_train[inds_train]=np.take(col_means, inds_train[1])
x_test[inds_test]=np.take(col_means, inds_test[1])

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

n_estimators = [300]
# Number of features to consider at every split
max_features = ['sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Class weights for class imbalance issue
class_weight = [None, "balanced", "balanced_subsample"]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'class_weight': class_weight}
print(random_grid)
# Use the random grid to search for best hyperparameters
rf = RandomForestClassifier()
# Random search of parameters and use all available cores
random_search = RandomizedSearchCV(estimator=rf,
                               param_distributions=random_grid,
                               n_iter=100,
                               cv=3, verbose=2, random_state=42,
                               n_jobs=-1, scoring='f1')
random_search.fit(x_train, y_train)
print(random_search.best_params_)
print("\tMean CV f1-score : %1.3f" % random_search.best_score_ )
# fit
rf_random = random_search.best_estimator_
#rf_random = RandomForestClassifier(n_estimators=300,
#                                   min_samples_split=5,
#                                   min_samples_leaf=1,
#                                   max_features='sqrt', max_depth=90,
#                                   bootstrap=True, random_state=42)
rf_random.fit(x_train, y_train)
# predict
y_pred_rf = rf_random.predict(x_test)
y_pred_prob_rf = rf_random.predict_proba(x_test)[:, 1]
# roc curve
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_prob_rf)
# precision, recall, f1
print('RF')
print("\tPrecision: %1.3f" % precision_score(y_test, y_pred_rf))
print("\tRecall: %1.3f" % recall_score(y_test, y_pred_rf))
print("\tF1: %1.3f" % f1_score(y_test, y_pred_rf))
print("\tAccuracy: {}".format(sum(y_pred_rf==y_test)/len(y_test)))

# print(cross_val_score(lasso, X, y, cv=3))
import sklearn
dt = sklearn.tree.DecisionTreeClassifier().fit(x_train,y_train)
# predict
y_pred_dt = dt.predict(x_test)
y_pred_prob_dt = dt.predict_proba(x_test)[:, 1]
# roc curve
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_prob_dt)
# precision, recall, f1
print('DT')
print("\tPrecision: %1.3f" % precision_score(y_test, y_pred_dt))
print("\tRecall: %1.3f" % recall_score(y_test, y_pred_dt))
print("\tF1: %1.3f" % f1_score(y_test, y_pred_dt))
print("\tAccuracy: {}".format(sum(y_pred_dt==y_test)/len(y_test)))
print('SVC')
svc =sklearn.svm.SVC().fit(x_train,y_train)
# predict
y_pred_svc = svc.predict(x_test)
y_pred_prob_svc = svc.predict_proba(x_test)[:, 1]
# roc curve
fpr_svc, tpr_svc, _ = roc_curve(y_test, y_pred_prob_svc)
# precision, recall, f1
print("\tPrecision: %1.3f" % precision_score(y_test, y_pred_svc))
print("\tRecall: %1.3f" % recall_score(y_test, y_pred_svc))
print("\tF1: %1.3f" % f1_score(y_test, y_pred_svc))
print("\tAccuracy: {}".format(sum(y_pred_svc==y_test)/len(y_test)))

import matplotlib.pyplot as plt
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, 'r',label='RF')
plt.plot(fpr_dt, tpr_dt, 'g',label='DT')
plt.plot(fpr_svc, tpr_svc, 'b',label='SVC')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve - Bibliographic')
plt.legend(loc='best')
plt.show()

