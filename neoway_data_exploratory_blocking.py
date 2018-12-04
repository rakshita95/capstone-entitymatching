import sys
sys.path.append('..')
sys.path.append('/anaconda/lib/python3.6/site-packages')
import pandas as pd
import numpy as np
from modules.preprocessing import Preprocessing
from modules.preprocessing.generate_labels import gen_labels
from modules.feature_generation.gen_similarities_blocking import similarities
from sklearn.model_selection import train_test_split
from modules.preprocessing import Preprocessor

'''
read data
'''
df1 = pd.read_csv('data/companies_data_neoway/input.csv')
df2 = pd.read_csv('data/companies_data_neoway/reference.csv')
block = pd.read_csv('company_zipcode_blocked_test_test.csv')
block = block.drop_duplicates() #in case there are duplicates in blocked.csv

'''
specify id names
'''
df1_id = 'serial'
df2_id = 'serial'
match_id1 = 'serial_input' #corresponds to df1_id
match_id2 = 'serial_reference' #corresponds to df2_id

'''
train/test split on input dataset
'''
#random split inputs into train/test using original dataset
df1_train, df1_test = train_test_split(df1, test_size=0.33, random_state=42)

#set index dic
df1_train_index = dict(zip(df1_train[df1_id], df1_train.reset_index().index))
df1_test_index = dict(zip(df1_test[df1_id], df1_test.reset_index().index))
df2_index = dict(zip(df2[df2_id], df2.reset_index().index))

'''
id column manipulation
'''
# save for later use to generate labels
df1_train_id_col = df1_train[df1_id]
df1_test_id_col = df1_test[df1_id]
df2_id_col = df2[df2_id]

#drop id columns because we don't need to compute id similarity
df1_train = df1_train.drop(columns = [df1_id])
df1_test = df1_test.drop(columns = [df1_id])
df2 = df2.drop(columns = [df2_id])

#also split block into train/test according to df1_train and df1_test
block_train = block[block['input_serial'].isin(df1_train_id_col)]
block_test = block[block['input_serial'].isin(df1_test_id_col)]

print('preprocessing')

processor = Preprocessor(special_columns=['name','addressStreet'],zip_code='addressZip')
processor.fit(df1_train,df2) #TODO: add fit_tansform function so no need to transform after fitting on training data


'''
get numerical data
'''
print('generate feature matrix')
def get_feature_matrix(df1,df2,df1_index,df2_index,block):
    processed_data = processor.transform(df1,df2)
    num_matrix_1,num_matrix_2 = processed_data["numerical"][0],processed_data["numerical"][1]
    embed_matrix_1,embed_matrix_2 = processed_data["word_embedding_fields"][0],processed_data["word_embedding_fields"][1]
    spc_matrix_1,spc_matrix_2 = processed_data["special_fields"][0],processed_data["special_fields"][1]
    X = []
    for i, r in block.iterrows():
        row=[]
        df1_i = df1_index[r['input_serial']]
        df2_i = df2_index[r['refer_serial']]
        row+=[similarities().numerical_similarity_on_matrix(num_matrix_1[[df1_i]],num_matrix_2[[df2_i]])]
        row+=[similarities().vector_similarity_on_matrix(embed_matrix_1[[df1_i]],embed_matrix_2[[df2_i]])]
        row+=[similarities().text_similarity_on_matrix(spc_matrix_1[[df1_i]],spc_matrix_2[[df2_i]],method = "lavenshtein")]
        row+=[similarities().text_similarity_on_matrix(spc_matrix_1[[df1_i]],spc_matrix_2[[df2_i]],method = "jaro_winkler")]
        row+=[similarities().text_similarity_on_matrix(spc_matrix_1[[df1_i]],spc_matrix_2[[df2_i]],method = "jaccard")]
        X+=[np.hstack(row)]
    X = np.vstack(X)
    return X

x_train = get_feature_matrix(df1_train,df2,df1_train_index,df2_index,block_train)
x_test = get_feature_matrix(df1_test,df2,df1_test_index,df2_index,block_test)

'''
generate labels
'''
print('generate labels')
mapping=pd.read_csv('data/companies_data_neoway/match.csv')
mapping['label']=1

y_train = pd.merge(block_train,mapping,left_on=['input_serial','refer_serial'],right_on=['serial_input','serial_reference'],how='left')
#y_train = y_train.drop_duplicates() #bc some depulicates in abt_buy_perfectMapping.csv?
y_train = y_train["label"].fillna(0).astype(int)

y_test = pd.merge(block_test,mapping,left_on=['input_serial','refer_serial'],right_on=['serial_input','serial_reference'],how='left')
#y_test = y_test.drop_duplicates() #bc some depulicates in abt_buy_perfectMapping.csv?
y_test = y_test["label"].fillna(0).astype(int)

print(y_train.shape[0] == x_train.shape[0])
print(y_test.shape[0] == x_test.shape[0])

print("***start modeling***")
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
# random_search = RandomizedSearchCV(estimator=rf,
#                                param_distributions=random_grid,
#                                n_iter=100,
#                                cv=3, verbose=2, random_state=42,
#                                n_jobs=-1, scoring='f1')
# random_search.fit(x_train, y_train)
# print(random_search.best_params_)
# print("\tMean CV f1-score : %1.3f" % random_search.best_score_ )
# fit

# rf_random = random_search.best_estimator_
rf_random = RandomForestClassifier(n_estimators=300,
                                   min_samples_split=5,
                                   min_samples_leaf=1,
                                   max_features='sqrt', max_depth=90,
                                   bootstrap=True, random_state=42, n_jobs=-1)
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


import pickle
# save the classifier
with open('neoway_rf.pkl', 'wb') as fid:
   pickle.dump(rf_random, fid)

