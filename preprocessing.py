import pandas as pd 
import numpy as np 
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import re
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize 
from nltk.stem import WordNetLemmatizer
import math

lemmatizer = WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')

# headers are: id, title, publication, author, 
# date, year, month, url, content
df = pd.read_csv('../data/articles1.csv', sep=',')

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

# document vectorization
documents = []
for i in range(1,100):
    documents.append(df.iloc[i]['content'])
    #print(documents[i-1])

docs = []
for document in documents:
    text = remove_string_special_chars(document)
    nl_text=''
    for word in word_tokenize(text):
        if word not in stopwords:
            nl_text += (lemmatizer.lemmatize(word)) + ' '
        docs.append(nl_text)
        #print(nl_text, '\n')

import collections
words = " ".join(docs).split()
count = collections.Counter(words).most_common()
print(count)