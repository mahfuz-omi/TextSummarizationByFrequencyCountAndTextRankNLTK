# https://www.analyticsvidhya.com/blog/2018/11/introduction-text-summarization-textrank-python/

import nltk
import numpy as np
import gensim
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
myStopword = stopwords.words('english')

def noisyTextToCleanText(sentence):
    words = nltk.word_tokenize(sentence)

    #remove punctuation from each word
    import string
    table = str.maketrans('', '', string.punctuation)
    stripped_words = [w.translate(table) for w in words]
    words_without_stopwords = []

    # removing stopwords
    for word in stripped_words:
        if word not in myStopword:
            words_without_stopwords.append(word)

    # convert word list to sentence
    sentence = ""
    for word in words_without_stopwords:
        sentence += " " + word
    return sentence

text = 'So, keep working. ' \
       'Keep striving. Never give up. ' \
       'Fall down seven times, get up eight. ' \
       'Ease is a greater threat to progress than hardship. ' \
       'Ease is a greater threat to progress than hardship. ' \
       'So, keep moving, keep growing, keep learning. ' \
       'See you at work.'



# convert text to sentences
sentences = text.split('.')
sentences.remove('')

print(sentences)
# ['So, keep working',
# ' Keep striving',
# ' Never give up',
# ' Fall down seven times, get up eight',
# ' Ease is a greater threat to progress than hardship',
# ' Ease is a greater threat to progress than hardship',
# ' So, keep moving, keep growing, keep learning', ' See you at work', '']

# Text Preprocessing
new_sentences = []
for sentence in sentences:
    new_sentences.append(noisyTextToCleanText(sentence))

print(new_sentences)

# download pre-trained word2vec
# import gensim.downloader as api
# wv = api.load('word2vec-google-news-300')

# senyences needed to be converted ibto vectors.
# I could use count or tf0idf. rather, i used word2vec.
# load downloaded pre-trained model
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors

file = get_tmpfile(r"F:\python codes\interview_codes\text_summarization\word2vec.txt")
wv = KeyedVectors.load_word2vec_format(file)

sentence_vectors = []

for sentence in new_sentences:
    sentence_vector = np.zeros(300)
    for word in nltk.word_tokenize(sentence):
        word_vector = wv[word]
        print(len(word_vector))
        sentence_vector = sentence_vector + word_vector

    sentence_vector = sentence_vector / len(nltk.word_tokenize(sentence))
    sentence_vectors.append(sentence_vector)

print('sentence vectors: ',sentence_vectors)

sentenceCount = len(new_sentences)

similarity_matrix = cosine_similarity(sentence_vectors, sentence_vectors)
print(similarity_matrix)
print(similarity_matrix.shape)

print(similarity_matrix[0])

scores = []

for i in range(0,sentenceCount):
    scores.append(np.linalg.norm(similarity_matrix[i]))

print(scores)

# ascending sort of the index(indirect sort)
index = np.argsort(scores)
print(index)

# [3 2 7 1 4 5 6 0]
# [0 6 5 4 1 7 2 3]


# reverse to get descending
index = index[::-1]

print(index)
# [6 5 4 0 3 7 2 1]

def printSummary(sentenceCount = 1):
    for i in range(0,sentenceCount):
        print(new_sentences[index[i]])


printSummary(3)

# So  keep working
#  So  keep moving  keep growing  keep learning
#  Ease greater threat progress hardship

