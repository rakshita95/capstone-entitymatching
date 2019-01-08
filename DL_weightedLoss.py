import sys
sys.path.append('..')
sys.path.append('/anaconda/lib/python3.6/site-packages')
from modules.preprocessing import Preprocessing
from modules.preprocessing.generate_labels import gen_labels
from modules.feature_generation.gen_similarities import similarities

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch import LongTensor, FloatTensor
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn.functional as func
import nltk
from modules.preprocessing.process_text import Process_text
from sklearn.metrics import precision_recall_fscore_support, f1_score, recall_score


def tokenize_normalize_sentence(sentence):
    """
    tokenize and normalize a sentence
    :arg:
        sentence: sentence to convert to list of normalized words; type: string
    :return:
        list of strings
    """
    text_processor = Process_text()
    processed_sentence = nltk.word_tokenize(sentence)
    processed_sentence = text_processor.remove_non_ascii(processed_sentence)
    processed_sentence = text_processor.to_lowercase(processed_sentence)
    processed_sentence = text_processor.remove_punctuation(processed_sentence)
    processed_sentence = text_processor.remove_nan(processed_sentence)
    processed_sentence = text_processor.remove_stopwords(processed_sentence)

    return processed_sentence

batch_size = 100

df1 = pd.read_csv("data/amazon_google/sample/amazon_sample_v1.csv")
df2 = pd.read_csv("data/amazon_google/sample/google_sample_v1.csv")
match_df = pd.read_csv("data/amazon_google/sample/amazon_google_sample_match_v1.csv")

df1['description'] = df1['description'].apply(str).apply(tokenize_normalize_sentence)
df2['description'] = df2['description'].apply(str).apply(tokenize_normalize_sentence)

all_words = set(x for lst in df1['description'] for x in lst)
all_words = all_words.union(set(x for lst in df2['description'] for x in lst))

word2idx = {}
for i, word in enumerate(all_words):
    if word not in word2idx.keys():
        word2idx[word] = i

print('#words', len(word2idx))

df1_id = 'id'
df2_id = 'id'
match_id1 = 'idAmazon'  #corresponds to df1_id
match_id2 = 'idGoogleBase'  #corresponds to df2_id
df2["price"] = df2.price.str.replace(r"[a-zA-Z]",'').astype(float)

# save for later use to generate labels
df1_id_col = df1[df1_id]
df2_id_col = df2[df2_id]

# drop id columns because we don't need to compute id similarity
df1 = df1.drop(columns=[df1_id])
df2 = df2.drop(columns=[df2_id])

processed_data = Preprocessing().overall_preprocess(
    df1.drop(columns=['description']), df2.drop(columns=['description']),
    special_columns=['title', 'manufacturer'],
    word_embedding_model='none') # may take a while bc loading pretrained word embedding model

num_matrix_1, num_matrix_2 = processed_data["numerical"][0], processed_data["numerical"][1]
spc_matrix_1, spc_matrix_2 = processed_data["special_fields"][0], processed_data["special_fields"][1]
num_final_data = similarities().numerical_similarity_on_matrix(num_matrix_1,num_matrix_2)
spc_final_data = similarities().text_similarity_on_matrix(spc_matrix_1,spc_matrix_2,method='jaccard')

df1['key'] = 0
df2['key'] = 0
merged = pd.merge(df1, df2, on='key')[['description_x', 'description_y']]

'''
train-test split
'''
non_empty = []

for m in num_final_data, spc_final_data:#, embed_mean_data, embed_max_data, embed_min_data:
    if m.size !=0:
        non_empty.append(m)

sim = np.concatenate([i for i in non_empty], axis = 1)
y = gen_labels(df1_id_col, df2_id_col, match_df, 'idAmazon', 'idGoogleBase')
# sim_train, sim_test, y_train, y_test = train_test_split(sim, y, test_size = 0.33, stratify = y, random_state=42)
sim_train, sim_test, y_train, y_test, desc_train, desc_test = train_test_split(
                     sim,
                     y, merged,
                     test_size=0.33,
                     stratify=y,
                     random_state=42)
print("train_true:", sum(y_train), "test_true: ",sum(y_test))
print("train_size: ", len(y_train), "test_size: ", len(y_test))

sim_dev, sim_finaltest, y_dev, y_finaltest, desc_dev, desc_finaltest = train_test_split(
                     sim_test,
                     y_test, desc_test,
                     test_size=3000,
                     stratify=y_test,
                     random_state=42)
print(sum(y_finaltest), sum(y_dev))
print(len(y_finaltest), len(y_dev))
max_desc1_len = max([len(x) for x in merged['description_x']])
max_desc2_len = max([len(x) for x in merged['description_y']])
# max474len = max([len(x.strip().split()) for x in train_df['Q6.33']])

'''
Make Tensors
'''
# collate_fn=pad_batch,

class Transform(Dataset):
    def __init__(self, sim, description, label, word2idx):
        self.sim = sim
        self.desc = description
        self.label = label
        self.word2idx = word2idx
        self.max_len_desc1 = 114#219-1997
        self.max_len_desc2 = 52#42-42

    def __len__(self):
        return len(self.label)#+len(self.desc)+len(self.sim)

    def __getitem__(self, index):
        sim = self.sim[index]
        desc1 = self.desc[index]['description_x']
        desc2 = self.desc[index]['description_y']
        label = self.label[index]

        desc1_idx1 = [self.word2idx.get(word, 1) for word in desc1]
        desc2_idx1 = [self.word2idx.get(word, 1) for word in desc2]

        if len(desc1_idx1) > self.max_len_desc1:
            desc1_idx1 = desc1_idx1[:self.max_len_desc1]
        if len(desc2_idx1) > self.max_len_desc2:
            desc2_idx1 = desc2_idx1[:self.max_len_desc2]

        # Zero pad
        desc1_idx = LongTensor(1, self.max_len_desc1).zero_()  # N X max_len
        desc2_idx = LongTensor(1, self.max_len_desc2).zero_()
        # print(desc1_idx1.shape, len(desc1_idx))
        # print(desc2_idx1.shape, len(desc2_idx))
        if len(desc1_idx1)!=0:
            desc1_idx[0, 0:len(desc1_idx1)].copy_(LongTensor(desc1_idx1))
        if len(desc2_idx1) != 0:
            desc2_idx[0, 0:len(desc2_idx1)].copy_(LongTensor(desc2_idx1))

        return desc1_idx, desc2_idx, sim, label


train_dataset = Transform(sim_train, desc_train.to_dict('records'), y_train,
                          word2idx)
train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=5,
    pin_memory=True)

# train_dataset = Transform(sim_train, desc_train.to_dict('records'), y_train,
#                           word2idx)
# class_sample_count = [len(np.where(y_train == 0)[0]),
#                       len(np.where(y_train == 1)[0])]
# print(class_sample_count)
# weights = 1 / torch.Tensor(class_sample_count)
# samples_weight = np.array([weights[t] for t in y_train])
# samples_weight = torch.from_numpy(samples_weight)
# train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
#     samples_weight.double(),
#     len(samples_weight)) # len(y_train)
# train_loader = torch.utils.data.DataLoader(
#     train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=5,
#     pin_memory=True)


dev_dataset = Transform(sim_dev, desc_dev.to_dict('records'), y_dev,
                        word2idx)  # feature2idx,
dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
dev_loader = torch.utils.data.DataLoader(
    dev_dataset, batch_size=batch_size, sampler=dev_sampler, num_workers=5,
    pin_memory=True)

'''
Embeddings
'''

def load_embeddings(word2idx, glove_file):
    corpus_words = set()
    for key in word2idx.keys():
        # if key not in {'<NULL>','<UNK>'}:
        corpus_words.add(key)

    glove_big = {}
    with open(glove_file, "rb") as infile:
        for line in infile:
            parts = line.split()
            word = parts[0].decode()  # .lower()
            nums = map(float, parts[1:])
            if (word in corpus_words):
                glove_big[word] = list(nums)

    idx2word = {v: k for k, v in word2idx.items()}

    weights_matrix = np.random.normal(scale=0.6, size=(
    len(idx2word), 300))  # np.zeros((len(idx2word), 300))
    words_found = 0

    for word in corpus_words:
        if word in glove_big.keys():
            weights_matrix[word2idx[word]] = glove_big[word]
            words_found += 1
    print("%d words found out of %d" % (words_found, len(idx2word)))

    return weights_matrix


weights_matrix = load_embeddings(word2idx, '/Users/serenazhang/Desktop/capstone/capstone-entitymatching/glove/glove.840B.300d.txt')
emb_layer = nn.Embedding(num_embeddings=len(word2idx), embedding_dim=300,
                         padding_idx=0)
emb_layer.weight.data.copy_(torch.from_numpy(weights_matrix))
emb_layer.weight.requires_grad = False


'''
Model Definition
'''

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.rnn1 = nn.LSTM(input_size=300, hidden_size=50, bidirectional=True,
                            num_layers=1) # for entity1
        self.rnn2 = nn.LSTM(input_size=300, hidden_size=50, bidirectional=True,
                            num_layers=1)  # for entity1
        # self.rnn2 = nn.LSTM(input_size=300, hidden_size=50) # for entity2
        # self.rnnbi2 = nn.LSTM(input_size=300, hidden_size=50, bidirectional=True)

        self.fcn1 = nn.Sequential(
            nn.Linear(203, 50),  # 53
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )  # nn.Dropout(0.2) nn.Linear(100, 1),
        # self.merge_fcn1.weight.data.normal_()

        self.final_fcn = nn.Sequential(
            nn.Linear(50, 2))  # nn.Dropout(0.2) nn.Linear(100, 1), nn.ReLU(inplace=True),
        # self.final_fcn.weight.data.normal_() nn.ReLU(inplace=True),



    def forward(self, desc1_embed, desc2_embed, sim):

        batch_size = desc1_embed.size()[0]

        desc1_encoded = self.rnn1(desc1_embed.transpose(1, 0))[1][0]
        forward_state1, backward_state1 = desc1_encoded[0, :, :].squeeze(), \
                                          desc1_encoded[1, :, :].squeeze()
        merged_state1 = torch.cat((forward_state1, backward_state1), dim=1)

        desc2_encoded = self.rnn2(desc2_embed.transpose(1, 0))[1][0]
        forward_state2, backward_state2 = desc2_encoded[0, :, :].squeeze(), \
                                          desc2_encoded[1, :, :].squeeze()
        merged_state2 = torch.cat((forward_state2, backward_state2), dim=1)

        # print("A1: ", merged_state1.size())
        # print("A2: ", merged_state2.size())
        # print("sim: ", sim.size())
        # print(type(sim))
        # print(type(desc1_encoded - desc2_encoded))
        sim_added = torch.cat([merged_state1 - merged_state2,
                               merged_state1 * merged_state2,
                               sim.float()], 1)
        # print("final: ", sim_added.size())

        output = self.fcn1(sim_added)

        final = self.final_fcn(output)
        #        print(final.size())
        return final


'''
Initialize model
'''
# Initialize the model
matcher = Model()#.cuda()
optimizer = optim.Adam(matcher.parameters())
# optimizer = optim.Adadelta(reader.parameters(), lr=0.001)
# optimizer = optim.SGD(reader.parameters(), lr = 0.05)
# torch.cuda.set_device(-1)
# model_name = 'with_topics_best'
class_sample_count = [len(np.where(y_train==0)[0]),len(np.where(y_train==1)[0])]
weights = 1 / torch.Tensor(class_sample_count)
# weights = torch.FloatTensor([0.1,1])
loss_func = nn.CrossEntropyLoss(weight=weights)
# loss_func = nn.CrossEntropyLoss()

for m in matcher.modules():
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.02)
        # torch.nn.init.xavier_normal_(m.weight)


def validate(data_loader, network):
    correct = 0
    examples = 0
    truth = []
    prediction = []

    for ex in data_loader:
        batch_size = ex[0].size(0)

        # Predicting....
        network.eval()

        desc1_idx = Variable(ex[0])
        desc2_idx = Variable(ex[1])
        sim = Variable(ex[2])
        actual = Variable(ex[3])

        desc1_embed = emb_layer(desc1_idx.squeeze())
        desc2_embed = emb_layer(desc2_idx.squeeze())
        # ques622_embed = emb_layer(ques622.repeat(ans622.size()[0], 1))
        #        print(ques622_embed.size())

        output = network(desc1_embed, desc2_embed, sim)

        _, pred = torch.max(output, 1)
        truth.append(list(actual.data))
        prediction.append(list(pred.data))
        correct += (pred.data == actual.data).sum()

        examples += batch_size

    truth_flat = [item for sublist in truth for item in sublist]
    pred_flat = [item for sublist in prediction for item in sublist]

    f1 = f1_score(truth_flat, pred_flat)
    recall = recall_score(truth_flat, pred_flat)

    return correct / examples, f1, recall

'''
Training
'''

num_epochs = 10

loss_list = []
train_list = []
test_list = []

for epoch in range(0, num_epochs):
#for epoch in range(10, 30):
    train_loss = 0
    for idx, sample in enumerate(train_loader):

        matcher.train()
        desc1_idx = Variable(sample[0])
        desc2_idx = Variable(sample[1])
        sim = Variable(sample[2])
        actual = Variable(sample[3])

        desc1_embed = emb_layer(desc1_idx.squeeze())
        desc2_embed = emb_layer(desc2_idx.squeeze())

        pred = matcher(desc1_embed, desc2_embed, sim)


        loss = loss_func(pred, actual)
        optimizer.zero_grad()  # set gradients to zero for each iteration
        loss.backward()  # backpropagate
        optimizer.step()  # update parameters

        # print(loss.data)
        loss_list.append(loss.data[0])

        train_loss += loss.data[0]
        # print(idx)
        if (idx % 50 == 0):
            print("Epoch", epoch + 1, "Batch-step", idx, "\t/",
                  len(train_loader), "\tloss", loss.data[0])
        torch.save({
            'epoch': epoch,
            'model_state_dict': matcher.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, str(epoch) + "model")

    val_acc, val_f1, val_recall = validate(dev_loader, matcher)
    train_acc, train_f1, train_recall = validate(train_loader, matcher)
    print(val_f1)
    print(train_f1)
    test_list.append(val_f1)
    train_list.append(train_f1)

epoch_list = list(range(0, len(train_list)))

test_acc, test_f1, test_recall, truth, pred = validate(dev_loader, matcher_loaded)