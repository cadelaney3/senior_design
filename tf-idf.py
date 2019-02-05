import pandas as pd 
import numpy as np 
import nltk
nltk.download('punkt')
import re
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize 
import math

# headers are: id, title, publication, author, 
# date, year, month, url, content
df = pd.read_csv('../data/articles1.csv', sep=',')

#print(df.head)
#print(df['id'].dtypes)
#print(df.describe())

content1 = df.iloc[1]['content']
content2 = df.iloc[2]['content']

#print(content1)

# pre-processing function
# takes in a string
# returns string with special chars removed
# source: https://towardsdatascience.com/tfidf-for-piece-of-text-in-python-43feccaa74f8
def remove_string_special_chars(s):
    # replace special char with ' '
    stripped = re.sub('[^\w\s]', '', s)
    stripped = re.sub('_', '', stripped)

    # change any whitespace into one space
    stripped = re.sub('\s+', ' ', stripped)

    # remove start and end white spaces
    stripped = stripped.strip()

    return stripped

# function for splitting text into sentences and considering
# each sentence as a document, calculates the total word count
# of each
# source: https://towardsdatascience.com/tfidf-for-piece-of-text-in-python-43feccaa74f8
def get_doc(sent):
    doc_info = []
    i = 0
    for sent in text_sents_clean:
        i += 1
        count = count_words(sent)
        temp = {'doc_id' : i, 'doc_length' : count}
        doc_info.append(temp)
    return doc_info

# returns the total num of words in input text
# source: https://towardsdatascience.com/tfidf-for-piece-of-text-in-python-43feccaa74f8
def count_words(sent):
    count = 0
    words = word_tokenize(sent)
    for word in words:
        count += 1
    return count

# creates a frequency dictionary for each word in doc
# source: https://towardsdatascience.com/tfidf-for-piece-of-text-in-python-43feccaa74f8
def create_freq_dict(sents):
    i = 0
    freqDict_list = []
    for sent in sents:
        i += 1
        freq_dict = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            if word in freq_dict:
                freq_dict[word] += 1
            else:
                freq_dict[word] = 1
            temp = {'doc_id' : i, 'freq_dict' : freq_dict}
        freqDict_list.append(temp)
    return freqDict_list

# gets term-frequency (tf)
# source: https://towardsdatascience.com/tfidf-for-piece-of-text-in-python-43feccaa74f8
def computeTF(doc_info, freqDict_list):
    TF_scores = []
    for tempDict in freqDict_list:
        id = tempDict['doc_id']
        for k in tempDict['freq_dict']:
            temp = {'doc_id' : id,
                    'TF_score' : tempDict['freq_dict'][k]/doc_info[id-1]['doc_length'],
                    'key' : k}
            TF_scores.append(temp)
    return TF_scores

# computes the inverse-document-frequency (IDF)
# source https://towardsdatascience.com/tfidf-for-piece-of-text-in-python-43feccaa74f8
def computeIDF(doc_info, freqDict_list):
    IDF_scores = []
    counter = 0
    for dict in freqDict_list:
        counter += 1
        for k in dict['freq_dict'].keys():
            count = sum([k in tempDict['freq_dict'] for tempDict in freqDict_list])
            temp = {'doc_id' : counter, 'IDF_score' : math.log(len(doc_info)/count), 'key' : k}
            IDF_scores.append(temp)

    return IDF_scores

# computes TF-IDF
# source: https://towardsdatascience.com/tfidf-for-piece-of-text-in-python-43feccaa74f8
def computeTFIDF(TF_scores, IDF_scores):
    TFIDF_scores = []
    for j in IDF_scores:
        for i in TF_scores:
            if j['key'] == i['key'] and j['doc_id'] == i['doc_id']:
                temp = {'doc_id' : j['doc_id'],
                        'TFIDF_score' : j['IDF_score'] * i['TF_score'],
                        'key' : i['key']}
        TFIDF_scores.append(temp)
    return TFIDF_scores

text_sents = sent_tokenize(content1)
text_sents_clean = [remove_string_special_chars(s) for s in text_sents]
doc_info = get_doc(text_sents_clean)
freqDict_list = create_freq_dict(text_sents_clean)
TF_scores = computeTF(doc_info, freqDict_list)
IDF_scores = computeIDF(doc_info, freqDict_list)
TFIDF_scores = computeTFIDF(TF_scores, IDF_scores)

#print(doc_info, '\n')
#print(freqDict_list)
#print(TF_scores, '\n', IDF_scores)
#print(TFIDF_scores)

#print(df.shape[0])

############################################################
# https://medium.com/deep-math-machine-learning-ai/chapter-9-1-nlp-word-vectors-d51bff9628c1
##############################################################
from sklearn.feature_extraction.text import TfidfVectorizer

docs = []
for i in range(1, 10):#df.shape[0]):
    text_sents = sent_tokenize(df.iloc[i]['content'])
    text_sents_clean = [remove_string_special_chars(s) for s in text_sents]
    text_sents_clean = ' '.join(word for word in text_sents_clean)
    docs.append(text_sents_clean)

vectorizer1 = TfidfVectorizer(sublinear_tf=True, max_df=1.0)
bow1 = vectorizer1.fit_transform(docs)

feature_names = vectorizer1.get_feature_names()
corpus_index = [n for n in docs]

df2 = pd.DataFrame(bow1.todense(), index=corpus_index, columns=feature_names)

############################################################
# https://medium.com/deep-math-machine-learning-ai/chapter-9-2-nlp-code-for-word2vec-neural-network-tensorflow-544db99f5334
############################################################

sentences = df.iloc[1:11]['content'].tolist()
normalized_sentences=[]
for sentence in sentences:
    text_sents = sent_tokenize(sentence)
    text_sents_clean = [remove_string_special_chars(s) for s in text_sents]
    text_sents_clean = ' '.join(word for word in text_sents_clean)
    normalized_sentences.append(text_sents_clean)

#print(normalized_sentences[:5])

import collections
words = " ".join(normalized_sentences).split()
count = collections.Counter(words).most_common()
print("Word count", count[:5])

unique_words = [i[0] for i in count]
dic = {w: i for i,w in enumerate(unique_words)}
voc_size = len(dic)
#print(dic)

data = [dic[word] for word in words]
#print('Sample data', data[:10], words[:10])

cbow_pairs = []
for i in range(1, len(data)-1):
    cbow_pairs.append([[data[i-1], data[i+1]], data[i]])
print('Context pairs rank ids', cbow_pairs[:5])
print()

cbow_pairs_words = []
for i in range(1, len(words)-1):
    cbow_pairs_words.append([[words[i-1], words[i+1]], words[i]])
print('Context pairs words', cbow_pairs_words[:5])

skip_gram_pairs = []
for c in cbow_pairs:
    skip_gram_pairs.append([c[1], c[0][0]])
    skip_gram_pairs.append([c[1], c[0][1]])
print('skip_gram pairs', skip_gram_pairs[:5])
print()

skip_gram_pairs_words = []
for c in cbow_pairs_words:
    skip_gram_pairs_words.append([c[1], c[0][0]])
    skip_gram_pairs_words.append([c[1], c[0][1]])
print('skip-gram pair words', skip_gram_pairs_words[:5])

def get_batch(size):
    assert size<len(skip_gram_pairs)
    X = []
    Y = []
    rdm = np.random.choice(range(len(skip_gram_pairs)), size, replace=False)

    for r in rdm:
        X.append(skip_gram_pairs[r][0])
        Y.append([skip_gram_pairs[r][1]])
    return X,Y

print('Batches (x,y)', get_batch(3))

import tensorflow as tf

batch_size = 20
embedding_size = 2
num_sampled = 15

X = tf.placeholder(tf.int32,shape=[batch_size])
Y = tf.placeholder(tf.int32,shape=[batch_size,1])

with tf.device("/cpu:0"):
    embeddings = tf.Variable(tf.random_uniform([voc_size,embedding_size],-1.0,-1.0))
    embed = tf.nn.embedding_lookup(embeddings, X)

nce_weights = tf.Variable(tf.random_uniform([voc_size,embedding_size],-1.0,-1.0))
nce_biases = tf.Variable(tf.zeros([voc_size]))

loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, Y, embed, num_sampled, voc_size))
optimizer = tf.train.AdamOptimizer(1e-1).minimize(loss)

epochs=10000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        batch_inputs, batch_labels = get_batch(batch_size)
        _, loss_val = sess.run([optimizer,loss],feed_dict = {X : batch_inputs, Y : batch_labels })

        if epoch % 1000 == 0:
            print("Loss at ", epoch, loss_val)

        trained_embeddings = embeddings.eval()

import matplotlib.pyplot as plt 

if trained_embeddings.shape[1] == 2:
    labels = unique_words[:100]
    for i, label in enumerate(labels):
        x, y = trained_embeddings[i,:]
        plt.scatter(x,y)
        plt.annotate(label, xy=(x,y), xytext=(5,2),
            textcoords='offset points', ha='right', va='bottom')
    plt.savefig("word2vec.png")
    plt.show()