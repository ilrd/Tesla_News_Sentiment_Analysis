from datetime import datetime
import pickle
import re
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords
from nltk.stem.wordnet import wordnet
from string import punctuation
from collections import Counter


def load_news_data(path):
    text_dates = []

    with open(path, 'rb') as f:
        data = pickle.load(f)

    if 'phys' in path:
        for text in data:
            date = datetime.strptime(re.sub(r'(20\d\d).*$', r'\1', text), '%B %d, %Y')
            text_dates.append((text, date))

    elif 'nytimes' in path:
        short_months = {
            'April': 'Apr',
            'Aug.': 'Aug',
            'Dec.': 'Dec',
            'Feb.': 'Feb',
            'Jan.': 'Jan',
            'July': 'Jul',
            'June': 'Jun',
            'March': 'Mar',
            'May': 'May',
            'Nov.': 'Nov',
            'Oct.': 'Oct',
            'Sept.': 'Sep'
        }

        for text, date in data:
            month = short_months.get(date.split(' ')[0])

            if date.split(' ')[1][-1] == ',':
                day = date.split(' ')[1][:-1]
            else:
                day = date.split(' ')[1]

            if len(date.split(' ')) == 3:
                year = int(date.split(' ')[2])
            else:
                year = 2020

            date = datetime.strptime(f'{month} {day}, {year}', '%b %d, %Y')

            text_dates.append((text, date))

    return text_dates


# nlp preprocessing
class NLP_Preprocessor:

    def __init__(self):
        pass

    # Splitting sentences
    @staticmethod
    def split_sent(text_data):
        # Sentence splitting
        sent_tok_text_data = [' [SENT] '.join(sent_tokenize(text)) for text in text_data]

        return sent_tok_text_data

    # Tokenization
    @staticmethod
    def tokenize(text_data, num_words=None):
        tokens = [word_tokenize(text) for text in text_data]

        if num_words:
            counter = Counter(word_tokenize(' '.join(text_data)))
            common_words = [word for word, _ in counter.most_common(num_words)]

            tokens = [[token for token in text if token in common_words] for text in tokens]

        return tokens

    # Lemmatization
    @staticmethod
    def lemmatize_tokens(tokens, pos=True):
        word_lem = WordNetLemmatizer()

        if pos:
            return [[word_lem.lemmatize(word, pos) for word, pos in
                     zip(tokens_i, map(NLP_Preprocessor.wordnet_pos, tokens_i))] for tokens_i in tokens]
        else:
            return [[word_lem.lemmatize(word) for word in tokens_i] for tokens_i in tokens]

    # POS tagging
    @staticmethod
    def wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
        }

        return tag_dict.get(tag, wordnet.NOUN)

    # Stemming
    @staticmethod
    def stem_lokens(tokens):
        sst = EnglishStemmer()
        return [[sst.stem(word) for word in tokens_i] for tokens_i in tokens]

    # Stopwords removal
    @staticmethod
    def remove_stopwords(tokens):
        stops = stopwords.words('english')

        return [[word for word in tokens_i if word not in stops] for tokens_i in tokens]

    # Punctuation removal
    @staticmethod
    def remove_punctuation(tokens):
        return [[word for word in tokens_i if word not in punctuation] for tokens_i in tokens]

    # Duplicates removal
    @staticmethod
    def remove_duplicates(data):
        non_dups = [['dummy', 'dummy']]
        for text, date in data:
            if text not in np.array(non_dups)[:, 0]:
                non_dups.append([text, date])

        return non_dups[1:]

    # Tokens to text
    @staticmethod
    def tokens_to_text(tokens):
        return [' '.join(tokens_i) for tokens_i in tokens]


