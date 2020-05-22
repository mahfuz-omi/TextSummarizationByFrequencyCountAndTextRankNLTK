# https://stackabuse.com/text-summarization-with-nltk-in-python/

import nltk
import numpy as np
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

# all words
all_tokens = []
for sentence in new_sentences:
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        all_tokens.append(token)


print(all_tokens)
print(len(all_tokens))

freqDist = nltk.FreqDist(all_tokens)

print(freqDist.items())

# [('Ease', 2), ('Fall', 1), ('Keep', 1),
# ('Never', 1), ('See', 1), ('So', 2),
# ('eight', 1), ('get', 1), ('give', 1),
# ('greater', 2), ('growing', 1), ('hardship', 2),
# ('keep', 4), ('learning', 1), ('moving', 1),
# ('progress', 2), ('seven', 1), ('striving', 1),
# ('threat', 2), ('times', 1), ('work', 1), ('working', 1)]

print(freqDist.most_common(1))
# [('keep', 4)]

print(freqDist.get('working'))
# 1

print(freqDist.get('pp'))
# 1

print(freqDist.__contains__('pp'))
# False

print(freqDist.__contains__('working'))
# True

# scores of each sentence
scores = []
for i in range(0,len(new_sentences)):
    score = 0
    words = nltk.word_tokenize(new_sentences[i])
    for word in words:
        if freqDist.__contains__(word):
            score = score + freqDist.get(word)

    scores.append(score)

print(scores)
# [7, 2, 2, 5, 10, 10, 17, 2]

# ascending sort of the index(indirect sort)
index = np.argsort(scores)
print(index)

# reverse to get descending
index = index[::-1]

print(index)
# [6 5 4 0 3 7 2 1]

def printSummary(sentenceCount = 1):
    for i in range(0,sentenceCount):
        print(new_sentences[index[i]])


printSummary(3)

# So  keep moving  keep growing  keep learning
#  Ease greater threat progress hardship
#  Ease greater threat progress hardship