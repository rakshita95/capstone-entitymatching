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
from modules.preprocessing import Preprocessor

'''
read data
'''
#df1 = pd.read_csv("data/acm_dblp/full/ACM.csv", engine='python')
#df2 = pd.read_csv("data/acm_dblp/full/DBLP2.csv", engine='python')
#match_df = pd.read_csv("data/acm_dblp/full/DBLP-ACM_perfectMapping.csv")

df1 = pd.read_csv("data/acm_dblp/sample/acm_sample.csv")
df2 = pd.read_csv("data/acm_dblp/sample/dblp_sample.csv")
match_df = pd.read_csv("data/acm_dblp/sample/acm_dblp_sample_match.csv")

'''
specify id names
'''
df1_id = 'id'
df2_id = 'id'
match_id1 = 'idACM' # corresponds to df1_id
match_id2 = 'idDBLP' # corresponds to df2_id

'''
train/test split on input dataset
'''
#random split inputs into train/test using original dataset
df1_train, df1_test = train_test_split(df1, test_size=0.33, random_state=42)


#set index dic
df1_train_index = dict(zip(df1_train[df1_id], df1_train.reset_index().index))
df1_test_index = dict(zip(df1_test[df1_id], df1_test.reset_index().index))

'''
id column manipulation
'''
# save for later use to generate labels
df1_train_id_col = df1_train[df1_id]
df1_test_id_col = df1_test[df1_id]
df2_id_col = df2[df2_id]

match_train = match_df[match_df['idACM'].isin(df1_train_id_col)]
match_test = match_df[match_df['idACM'].isin(df1_test_id_col)]

#drop id columns because we don't need to compute id similarity
df1_train = df1_train.drop(columns = [df1_id])
df1_test = df1_test.drop(columns = [df1_id])
df2 = df2.drop(columns = [df2_id])

print('\tdf1 train shape: ', df1_train.shape, '\n',
      '\tmatch train shape: ', match_train.shape, '\n'
      '\tdf1 test shape: ', df1_test.shape, '\n'
      '\tmatch test shape: ', match_test.shape)

print("***preprocessing***")

processor = Preprocessor(special_columns=['title', 'authors', 'venue'])
processor.fit(df1_train,df2) #fitting on training dataset for input and on whole dataset for ref

processed_train = processor.transform(df1_train, df2)
processed_test = processor.transform(df1_test, df2)

def gen_x(processed_data):

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
    embed_tfidf_data = similarities().vector_similarity_on_matrix(embed_matrix_1,embed_matrix_2)
    #embed_mean_data = similarities().vector_similarity_on_matrix(embed_matrix_1,embed_matrix_2)
    #embed_min_data = similarities().vector_similarity_on_matrix(embed_matrix_1,embed_matrix_2)
    #embed_max_data = similarities().vector_similarity_on_matrix(embed_matrix_1,embed_matrix_2)
    spc_lav_data = similarities().text_similarity_on_matrix(spc_matrix_1,spc_matrix_2, method = "lavenshtein")
    spc_jw_data = similarities().text_similarity_on_matrix(spc_matrix_1, spc_matrix_2, method="jaro_winkler")
    spc_jc_data = similarities().text_similarity_on_matrix(spc_matrix_1, spc_matrix_2, method="jaccard")
    '''
    concatenate all data
    '''
    # only concatenate non-empty similarity matrices
    non_empty = []

    for m in num_final_data, embed_tfidf_data, spc_lav_data, spc_jw_data, spc_jc_data:
        if m.size !=0:
            non_empty.append(m)

    x = np.concatenate([i for i in non_empty], axis = 1)

    print(x.shape)

    return x

x_train = gen_x(processed_train)
x_test = gen_x(processed_test)

'''
train test split
'''
# generate y labels
y_train = gen_labels(df1_train_id_col, df2_id_col, match_train, match_id1, match_id2)

# simple check to see if x and y match in size
print (y_train.shape[0] == x_train.shape[0])
print(y_train.sum() == match_train.shape[0])

# generate y labels
y_test = gen_labels(df1_test_id_col, df2_id_col, match_test, match_id1, match_id2)

# simple check to see if x and y match in size
print (y_test.shape[0] == x_test.shape[0])
print(y_test.sum() == match_test.shape[0])

'''
save features
'''
np.save('acm_x_train',x_train)
#del x_train

print("***x_train saved***")

np.save('acm_x_test',x_test)
#del x_test

print("***x_test saved***")


# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, stratify = y,random_state=42)
#
# precision = []
# recall = []
# f1 = []
# accuracy = []
# test_size = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
# for split in test_size:
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split,
#                                                         stratify=y,
#                                                         random_state=42)
#     col_means = np.nanmean(x_train, axis=0)
#     inds_train = np.where(np.isnan(x_train))
#     inds_test = np.where(np.isnan(x_test))
#     x_train[inds_train] = np.take(col_means, inds_train[1])
#     x_test[inds_test] = np.take(col_means, inds_test[1])
#
#     rf_random = RandomForestClassifier(n_estimators=300,
#                                        min_samples_split=5,
#                                        min_samples_leaf=1,
#                                        max_features='sqrt', max_depth=100,
#                                        bootstrap=False, random_state=42)
#     rf_random.fit(x_train, y_train)
#     y_pred = rf_random.predict(x_test)
#     precision.append(precision_score(y_test, y_pred))
#     recall.append(recall_score(y_test, y_pred))
#     f1.append(f1_score(y_test, y_pred))
#     accuracy.append(sum(y_pred == y_test) / len(y_test))
#
# plt.figure()
# plt.plot([int((1-x)*len(y)) for x in test_size], recall, label='recall')
# plt.plot([int((1-x)*len(y)) for x in test_size], f1, label='f1_score')
# plt.xlabel('Number of training samples')
# plt.ylabel('Performance')
# plt.title('Total number of samples = '+str(len(y)))
# plt.suptitle('Bibliographic')
# plt.legend()

'''
modeling
'''
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve,  precision_score, recall_score, f1_score
#import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV

col_means = np.nanmean(x_train,axis=0)
inds_train  = np.where(np.isnan(x_train))
inds_test = np.where(np.isnan(x_test))
x_train[inds_train]=np.take(col_means, inds_train[1])
x_test[inds_test]=np.take(col_means, inds_test[1])

# #upsample
# x_maj = x_train[y_train==0]
# x_min = x_train[y_train==1]
# x_min_upsampled = resample(x_min,n_samples=x_maj.shape[0],random_state=42)
# x_train_new = np.vstack((x_maj, x_min_upsampled))
# y_train_new = np.hstack((np.zeros(x_maj.shape[0]), np.ones(x_maj.shape[0])))

# # CV
# # Number of trees in random forest
# # n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
# n_estimators=[300]
# # Number of features to consider at every split
# max_features = ['sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
# # Class weights for class imbalance issue
# class_weight = [None, "balanced", "balanced_subsample"]
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap,
#                'class_weight': class_weight}
# print(random_grid)
# # Use the random grid to search for best hyperparameters
# rf = RandomForestClassifier()
# # Random search of parameters and use all available cores
# random_search = RandomizedSearchCV(estimator=rf,
#                                param_distributions=random_grid,
#                                n_iter=100,
#                                cv=3, verbose=2, random_state=42,
#                                n_jobs=-1, scoring='f1')
# random_search.fit(x_train, y_train)
# print(random_search.best_params_)
# print("\tMean CV f1-score : %1.3f" % random_search.best_score_ )
# # fit
# rf_random = random_search.best_estimator_
# #rf_random = RandomForestClassifier(n_estimators=300,
# #                                   min_samples_split=5,
# #                                   min_samples_leaf=1,
# #                                   max_features='sqrt', max_depth=90,
# #                                   bootstrap=True, random_state=42)
# random_search.fit(x_train, y_train)
# print(random_search.best_params_)
# print("\tMean CV f1-score : %1.3f" % random_search.best_score_ )
# # fit

# rf_random = random_search.best_estimator_
rf_random = RandomForestClassifier(n_estimators=300,
                                   min_samples_split=5,
                                   min_samples_leaf=1,
                                   max_features='sqrt', max_depth=100,
                                   bootstrap=False, random_state=42,n_jobs=-1)
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



# save the classifier
import pickle
with open('acm_rf.pkl', 'wb') as fid:
   pickle.dump(rf_random, fid, protocol=4)

print("***model saved***")

# dt = sklearn.tree.DecisionTreeClassifier().fit(x_train,y_train)
# # predict
# y_pred_dt = dt.predict(x_test)
# y_pred_prob_dt = dt.predict_proba(x_test)[:, 1]
# # roc curve
# fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_prob_dt)
# # precision, recall, f1
# print('DT')
# print("\tPrecision: %1.3f" % precision_score(y_test, y_pred_dt))
# print("\tRecall: %1.3f" % recall_score(y_test, y_pred_dt))
# print("\tF1: %1.3f" % f1_score(y_test, y_pred_dt))
# print("\tAccuracy: {}".format(sum(y_pred_dt==y_test)/len(y_test)))
# print('SVC')
# svc =sklearn.svm.SVC(probability=True).fit(x_train,y_train)
# # predict
# y_pred_svc = svc.predict(x_test)
# y_pred_prob_svc = svc.predict_proba(x_test)[:, 1]
# # roc curve
# fpr_svc, tpr_svc, _ = roc_curve(y_test, y_pred_prob_svc)
# # precision, recall, f1
# print("\tPrecision: %1.3f" % precision_score(y_test, y_pred_svc))
# print("\tRecall: %1.3f" % recall_score(y_test, y_pred_svc))
# print("\tF1: %1.3f" % f1_score(y_test, y_pred_svc))
# print("\tAccuracy: {}".format(sum(y_pred_svc==y_test)/len(y_test)))
#
# import matplotlib.pyplot as plt
# plt.figure(1)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_rf, tpr_rf, 'r',label='RF')
# plt.plot(fpr_dt, tpr_dt, 'g',label='DT')
# plt.plot(fpr_svc, tpr_svc, 'b',label='SVC')
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve - Bibliographic')
# plt.legend(loc='best')
# plt.show()

