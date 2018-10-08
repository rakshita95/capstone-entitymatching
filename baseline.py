
# coding: utf-8

# In[1]:

import pandas as pd

input_df = pd.read_csv('companies_data_neoway/input.csv')
ref_df = pd.read_csv('companies_data_neoway/reference.csv')
match_df = pd.read_csv('companies_data_neoway/match.csv')


# In[2]:

import py_entitymatching as em
em.set_key(input_df, 'serial')
em.set_key(ref_df, 'serial')

# Downsample the datasets 
sample_input, sample_ref = em.down_sample(input_df, ref_df, size=1000, y_param=1, show_progress=False)
print(len(sample_input), len(sample_ref))


# In[2]:

# sample_input.head()


# In[4]:

## blocking: blocking on addressCity since addressState format is different for input and reference datasets
ob = em.OverlapBlocker()

# Specify the tokenization to be 'word' level and set overlap_size to be 3.
C = ob.block_tables(sample_input, sample_ref, 'addressCity', 'addressCity', word_level=True, overlap_size=2, 
                    l_output_attrs=['name', 'addressStreet', 'addressZip','addressState'], 
                    r_output_attrs=['name', 'addressStreet', 'addressZip','addressState'],
                    show_progress=False)


# In[5]:

## labelling
C['gold']=0
with_labels = (C['ltable_serial'].map(str)+'_'+C['rtable_serial'].map(float).map(str))                             .isin(match_df['serial_input'].map(str)+'_'+match_df['serial_reference'].map(str))
C.loc[with_labels,'gold'] = 1


# In[6]:

# C.head()


# In[7]:

# Generate features automatically 
feature_table = em.get_features_for_matching(sample_input, sample_ref, validate_inferred_attr_types=False)
feature_table = feature_table[(feature_table.left_attribute!='addressCity') & (feature_table.left_attribute!='serial')]
feature_table

# name	addressStreet	addressCity	addressZip	addressState
# Select the attrs. to be included in the feature vector table
attrs_from_table = ['ltable_name', 'ltable_addressStreet', 'ltable_addressZip', 'ltable_addressState',
                    'rtable_name', 'rtable_addressStreet', 'rtable_addressZip', 'rtable_addressState']
# Convert the labeled data to feature vectors using the feature table
H = em.extract_feature_vecs(C, 
                            feature_table=feature_table, 
                            attrs_before = attrs_from_table,
                            attrs_after='gold',
                            show_progress=False)


# In[1]:

# H.head()


# In[9]:

## Impute features

print(any(pd.notnull(H)))

attrs_to_be_excluded = []
attrs_to_be_excluded.extend(['_id', 'ltable_serial', 'rtable_serial', 'gold'])
attrs_to_be_excluded.extend(attrs_from_table)

# Impute feature vectors with the mean of the column values.
H = em.impute_table(H, 
                exclude_attrs=attrs_to_be_excluded,
                strategy='mean')

# print(any(pd.notnull(H)))


# In[10]:

## RF Matcher
rf = em.RFMatcher()
rf.fit(table=H, exclude_attrs=attrs_to_be_excluded, target_attr='gold')


# In[11]:

# Select the best ML matcher using 5 fold CV
result = em.select_matcher([rf], table=H, 
        exclude_attrs=attrs_to_be_excluded,
        k=5,
        target_attr='gold', metric_to_select_matcher='f1', random_state=42)
print(result['cv_stats'])


# 

# In[ ]:



