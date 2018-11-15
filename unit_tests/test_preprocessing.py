import sys
import pytest
import numpy as np
import gensim
import pandas as pd

sys.path.append('..')
from modules.preprocessing import Preprocessing
from modules.preprocessing.process_text import Process_text
from modules.preprocessing.word_embedding import Word_embedding


#test functions from Process_text
def test_remove_non_ascii():
    words = ["6Â$%GSq","918Â","417fdfÂ","712dfaåbäcödf"]
    assert(Process_text().remove_non_ascii(words)==["6A$%GSq","918A","417fdfA","712dfaabacodf"])


def test_to_lowercase():
    words = ["6Â$%GSq","918Â","417fdfÂ","712dfaÅbäcödf"]
    assert(Process_text().to_lowercase(words)==["6â$%gsq","918â","417fdfâ","712dfaåbäcödf"])

def test_remove_punctuation():
    words = ["6Â$%GSq","918Â","417fdfÂ","712dfaÅbäcödf()*&^%$#&@!~-`+[]:\"\'/?><.,\|"]
    assert(Process_text().remove_punctuation(words)==["6ÂGSq","918Â","417fdfÂ","712dfaÅbäcödf"])

#test functions from Word_embeddings
#passed below tests but commented out for now bc you may not have the bin file and it takes a while to load the bin file)
'''
def test_sentence_to_embedding():
    sentence = "6Â$%GSq 918Â 417fdfÂ 712dfaÅbäcödf() *&^%$#&@!~-`+[]:\"\'/?><.,\|abc"
    assert(Word_embedding('/Users/shihhuayu/capstone/capstone-entitymatching/data/embeddings/GoogleNews-vectors-negative300.bin').sentence_to_embedding(sentence).shape == (300,)) #does not work in new version (word embedding with fit/transform) bc sentence_to_embedding was removed

def test_dataframe_to_embedding():
    df = pd.read_csv('/Users/shihhuayu/capstone/companies_data_neoway_subsample/input.csv')
    with pytest.raises(ValueError):
        Word_embedding('/Users/shihhuayu/capstone/capstone-entitymatching/data/embeddings/GoogleNews-vectors-negative300.bin').dataframe_to_embedding(df,["name","addressStreet","addressCity","wrong_attribute"])
    assert(Word_embedding('/Users/shihhuayu/capstone/capstone-entitymatching/data/embeddings/GoogleNews-vectors-negative300.bin').dataframe_to_embedding(df,["name","addressStreet","addressCity","addressState"]).shape == (189, 4, 300)) #works in new version (word embedding with fit/transform)
'''


    
#test functions from Preprocessing
#passed below tests but commented out for now bc you may not have the bin file and it takes a while to load the bin file)
'''
df1 = pd.read_csv('/Users/shihhuayu/capstone/companies_data_neoway_subsample/reference.csv')
df2 = pd.read_csv('/Users/shihhuayu/capstone/companies_data_neoway_subsample/input.csv')
processed_data = Preprocessing().overall_preprocess(df1,df2)

def test_overall_preprocess_num_col():
    assert(processed_data["numerical"][0].shape==(367,2) and processed_data["numerical"][1].shape==(189,2)) #right now also detecting "serial" as num col, TODO: fix "overall_preprocess" and change test to check if shape equals (367,1)

def test_overall_preprocess_embed_col():
    assert(processed_data["word_embedding_fields"][0].shape==(367,4,300) and processed_data["word_embedding_fields"][1].shape==(189,4,300))
    #assert(processed_data["word_embedding_fields"][0].size == 0 and processed_data["word_embedding_fields"][1].size == 0) #empty case
'''
