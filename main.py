import nltk
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from nltk.tokenize import word_tokenize
#nltk.download("stopwords")
#nltk.download('punkt')
#nltk.download('wordnet')
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import os
from textblob import TextBlob
from nltk.stem import PorterStemmer
from textblob import Word
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import sklearn.feature_extraction.text as text
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import CountVectorizer
from scipy import spatial
from sent2vec.vectorizer import Vectorizer
import gensim
from gensim.models import Word2Vec
from pyemd import emd

stemmer = SnowballStemmer("english")
import pandas as pd
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
nltk.download("stopwords")
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import sent_tokenize, word_tokenize

df = pd.read_csv('train.csv')
df = df.iloc[0:4000, :]
df.head(4000)

# making all words lowercase
df['question1'] = df['question1'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df['question2'] = df['question2'].apply(lambda x: " ".join(x.lower() for x in x.split()))


# data cleaning
def removingSpecialChars(word):
    word = word.replace("Â£", ' ')
    word = word.replace("@", 'at')
    word = word.replace("kÃ¶ln", ' ')
    word = word.replace("aren't", 'are not')
    word = word.replace("isn't", 'is not')
    word = word.replace("i've", 'i have')
    word = word.replace("can't", 'can not')
    word = word.replace("masterâ€™s", 'masters')
    word = word.replace("â€™s", ' ')
    word = word.replace('vs.', 'versus')
    word = word.replace("i'm", 'i am')
    word = word.replace('1-2', '1 to 2')
    word = word.replace('&', 'and')
    word = word.replace("4D", '4 dimentional')
    word = word.replace("7th", 'seventh')
    word = word.replace("10th", 'tenth')
    word = word.replace("_", ' ')
    word = word.replace("â€™", ' ')
    word = word.replace("â€˜", ' ')
    word = word.replace("=", ' ')
    word = word.replace(">", ' ')
    word = word.replace("?", ' ')
    word = word.replace("[", ' ')
    word = word.replace("]", ' ')
    word = word.replace("\\", ' ')
    word = word.replace("^", ' ')
    word = word.replace("`", ' ')
    word = word.replace("{", ' ')
    word = word.replace("}", ' ')
    word = word.replace("|", ' ')
    word = word.replace("~", ' ')
    word = word.replace("-", ' ')
    word = word.replace("#", ' ')
    word = word.replace("'", ' ')
    word = word.replace("(", ' ')
    word = word.replace(")", ' ')
    word = word.replace("*", ' ')
    word = word.replace("+", ' ')
    word = word.replace(",", ' ')
    word = word.replace("-", ' ')
    word = word.replace(".", ' ')
    word = word.replace("/", ' ')
    word = word.replace(":", ' ')
    word = word.replace(";", ' ')
    word = word.replace("<", ' ')
    word = word.replace('/', ' ')
    word = word.replace("-", ' ')
    word = word.replace("#", ' ')
    word = word.replace("$", 'dollar')
    word = word.replace("it's", 'it is')
    word = word.replace("3G", '3 generation')
    word = word.replace("â€™", ' ')
    word = word.replace("â‚¹", ' ')
    word = word.replace("?", ' ')
    word = word.replace("5th", 'fifth')
    word = word.replace("6th", 'sixth')
    word = word.replace("8th", 'eighth')
    word = word.replace("9th", 'ninth')
    word = word.replace("ã‚·", 'a')
    word = word.replace("ã—", 'a')
    word = word.replace("!", ' ')
    word = word.replace('"', ' ')
    word = word.replace('%', 'percent')
    word = word.replace("doesn't", 'does not')
    word = word.replace('18k', '18 kilo')
    word = word.replace('10k', '10 kilo')
    word = word.replace("don't", 'do not')
    word = word.replace("clockâ€™s", 'clock')
    word = word.replace("4G", '4 generation')
    word = word.replace("hadn't", 'had not')
    word = word.replace("3D", '3 dimensional')
    word = word.replace("5D", '5 dimensional')
    word = word.replace('í', 'i')
    word = word.replace('á', 'a')
    word = word.replace("couldn't", 'could not')
    word = word.replace("What's", 'What is')
    word = word.replace("US", 'united states')
    word = word.replace("â€", ' ')
    word = word.replace('á', 'a')
    word = word.replace("1st", 'first')
    word = word.replace("2nd", 'second')
    word = word.replace("3rd", 'third')
    word = word.replace("4th", 'fourth')
    word = word.replace("hadn't", 'had not')
    word = word.replace("hasn't", 'has not')
    word = word.replace("haven't", 'have not')
    word = word.replace("wouldn't", 'would not')
    word = word.replace("shouldn't", 'should not')
    word = word.replace("ã‚·", ' ')
    word = word.replace("ã—", ' ')
    word = word.replace("â‚¹", ' ')
    word = word.replace("â€“", ' ')

    return word


df['question1_cleaned'] = df['question1'].apply(lambda x: " ".join(removingSpecialChars(x) for x in x.split()))
df['question2_cleaned'] = df['question2'].apply(lambda x: " ".join(removingSpecialChars(x) for x in x.split()))

# removing stop words
stop_words = set(stopwords.words("english"))
pat = r'\b(?:{})\b'.format('|'.join(stop_words))
df['question1_cleaned'] = df['question1_cleaned'].astype(str).str.replace(pat, '')
df['question1_cleaned'] = df['question1_cleaned'].str.replace(r'\s+', ' ')

df['question2_cleaned'] = df['question2_cleaned'].astype(str).str.replace(pat, '')
df['question2_cleaned'] = df['question2_cleaned'].str.replace(r'\s+', ' ')

# removing spaces
df['question1_cleaned'] = [word for word in df['question1_cleaned'] if word]

df['question2_cleaned'] = [word for word in df['question2_cleaned'] if word]


# tokenization
def tokenization(text):
    tokens = word_tokenize(text)
    return tokens


df['q1_tokens'] = df['question1_cleaned'].apply(lambda x: tokenization(x))
df['q2_tokens'] = df['question2_cleaned'].apply(lambda x: tokenization(x))

# stemmeing
snowballStemmer = SnowballStemmer("english")


def SnowballStemmer(text):
    stemmed = ' '.join([snowballStemmer.stem(word) for word in text])
    return stemmed


df['question1_stemmed'] = df['q1_tokens'].apply(lambda x: SnowballStemmer(x))
df['question2_stemmed'] = df['q2_tokens'].apply(lambda x: SnowballStemmer(x))

print(df)

df.to_csv('nlp_data1.csv', index=False)

data = pd.DataFrame(df)
data.drop('id', inplace=True, axis=1)
data.drop('qid1', inplace=True, axis=1)
data.drop('qid2', inplace=True, axis=1)
data.drop('question1', inplace=True, axis=1)
data.drop('question2', inplace=True, axis=1)
data.drop('question1_cleaned', inplace=True, axis=1)
data.drop('question2_cleaned', inplace=True, axis=1)
data.drop('q1_tokens', inplace=True, axis=1)
data.drop('q2_tokens', inplace=True, axis=1)
data = data[['question1_stemmed', 'question2_stemmed', 'is_duplicate']]
data['question1_stemmed'] = data['question1_stemmed'].apply(
    lambda x: " ".join(removingSpecialChars(x) for x in x.split()))
data['question2_stemmed'] = data['question2_stemmed'].apply(
    lambda x: " ".join(removingSpecialChars(x) for x in x.split()))

data.to_csv('nlp_data2.csv', index=False)
print(data)

vectorizer = TfidfVectorizer()


def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0, 1]


Tfidf_scores = []

for i in range(len(data)):
    score = cosine_sim(data['question1_stemmed'][i], data['question2_stemmed'][i])
    Tfidf_scores.append(score)

print(Tfidf_scores)
data['scores'] = Tfidf_scores
column_to_move = data.pop("is_duplicate")
data.insert(3, "is_duplicate", column_to_move)
data.to_csv('nlp_data2.csv', index=False)

# model training
x = data[['scores']]
y = data['is_duplicate']

train_x, test_x, train_y, test_y = model_selection.train_test_split(x, y)

model = DecisionTreeClassifier()
model.fit(train_x, train_y)

predictions = model.predict(test_x)
accuracy = metrics.accuracy_score(predictions, test_y)
print("Accuracy: ", accuracy)

