from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
from datetime import datetime
from src.preprocessing.preprocessing import NLP_Preprocessor, load_news_data
import pickle

data = load_news_data('../../data/raw/data_nytimes.pickle')
preprocessor = NLP_Preprocessor()
data = preprocessor.remove_duplicates(data)
data = np.array(data)
text_data = data[:, 0]
dates = data[:, 1]

# text_data = preprocessor.split_sent(text_data)
# tokens = preprocessor.tokenize(text_data, num_words=8000)
# tokens = preprocessor.lemmatize_tokens(tokens)
# tokens = preprocessor.remove_stopwords(tokens)
# tokens = preprocessor.remove_punctuation(tokens)
# text_data = preprocessor.tokens_to_text(tokens)

vader = SentimentIntensityAnalyzer()

scores_dates = [(vader.polarity_scores(text_data[i])['compound'], dates[i]) for i in range(len(data))]

scores_dates.sort(key=lambda item: item[1])
scores_dates_df = pd.DataFrame(data=scores_dates, index=None, columns=['Score', 'Date'])

# --------------------------------------------#
# Loading stock data
df = pd.read_csv('../../data/TSLA_daily.csv')

# String date to datetime
df['Date'] = df['Date'].apply(lambda str_date: datetime.strptime(str_date, '%Y-%m-%d'))

# Filtering dataframes so they only have records on the same dates
intrs_dates = np.intersect1d(df['Date'], scores_dates_df['Date'])

df = df[df.apply(lambda row: row[0] in intrs_dates, axis=1)]
df.reset_index(drop=True, inplace=True)
scores_dates_df = scores_dates_df[scores_dates_df.apply(lambda row: row[1] in intrs_dates, axis=1)]

# Dropping duplicates
no_dup_df = pd.DataFrame(columns=['Score', 'Date'])
no_dup_dict = {}
for row in zip(list(list(scores_dates_df.items())[0][1].to_numpy()), list(scores_dates_df.items())[1][1].to_numpy()):
    current_date = row[1]
    current_score = row[0]
    if current_date in no_dup_dict:
        prev_avg = no_dup_dict[current_date][0]
        prev_count = no_dup_dict[current_date][1]
        no_dup_dict[current_date][0] = (prev_avg / prev_count + current_score) / (prev_count + 1)
        no_dup_dict[current_date][1] += 1
    else:
        no_dup_dict[current_date] = [current_score, 1]

no_dup_items = np.array(list(no_dup_dict.items()))
no_dup_data = zip(np.array(list(map(np.array, no_dup_items[:, 1])))[:, 0],
                  np.array(list(map(np.array, no_dup_items[:, 0]))))
no_dup_df = pd.DataFrame(data=no_dup_data, columns=['Score', 'Date'])

# Merging datasets
df['Sentiment'] = no_dup_df['Score']


# --------------------------------------------#

def EMA(points, alpha=0.1):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * (1 - alpha) + point * alpha)
        else:
            smoothed_points.append(point)
    return smoothed_points


def SMA(points, window=5):
    sma_points = points.rolling(window, min_periods=1).mean()
    return sma_points


# Plotting stock prices
import matplotlib.pyplot as plt

plt.figure()
plt.plot(df['Close'])
plt.title('Tesla stock closing prices')
plt.show()

plt.figure()
plt.plot(EMA(df['Sentiment'], alpha=0.1))
plt.title('EMA of news sentiment about Tesla')
plt.show()

# SMA of news sentiment
# plt.plot(SMA(df['Sentiment'], window=10))
# plt.title('SMA of news sentiment about Tesla')

# --------------------------------------------#
# Correlations

from scipy.stats import pearsonr


def corr_table(df, measure='corr'):
    corr_stars = lambda corr: '' if abs(corr) < 0.05 else '*' if 0.05 <= abs(corr) < 0.2 else '**' \
        if 0.2 <= abs(corr) < 0.5 else '***'

    p_stars = lambda p: '***' if p < 0.01 else '**' if 0.01 <= p < 0.05 else \
        '*' if 0.05 <= p < 0.15 else ''

    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            corr = round(pearsonr(df[r], df[c])[0], 4)
            p = round(pearsonr(df[r], df[c])[1], 4)

            if measure == 'corr':
                pvalues[r][c] = str(corr) + corr_stars(corr)
            elif measure == 'p':
                pvalues[r][c] = str(p) + p_stars(p)

    return pvalues


print('*** - p value < 0.01, ** - p value < 0.05, * - p value < 0.15, no star otherwise\n')

# Correlation between the features
print('Correlation table between close price, volume and news sentiment:\n',
      corr_table(df[['Close', 'Volume', 'Sentiment']], measure='corr'), sep='', end='\n\n')


with open('../../data/processed/stock_sentiment.pickle', 'wb') as f:
    pickle.dump(df, f)
