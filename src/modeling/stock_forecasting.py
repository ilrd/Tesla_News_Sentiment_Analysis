from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.model_selection import train_test_split

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

with open('../../data/processed/stock_sentiment.pickle', 'rb') as f:
    df = pickle.load(f)

diff = df['Open'] - df['Close']
diff.reset_index(drop=True, inplace=True)
y_cont = diff.to_numpy()
scaler = StandardScaler()
y_cont = scaler.fit_transform(y_cont.reshape((-1, 1))).flatten()

X_df = df[['Open', 'Sentiment']].copy()
X_df['Open'] = scaler.fit_transform(X_df['Open'].values.reshape((-1, 1))).flatten()
X_df.reset_index(drop=True, inplace=True)
X = X_df.values
# --------------------------------------------#
# Prediction of Close-Open value using prices and sentiments of previous days and opening price and sentiment of current day
T = 4

series_range = range((len(X) - 1) // T)
X_series = np.array([X[T * i:T * (i + 1)] for i in series_range])
y_series = np.array([y_cont[T * (i + 1)] for i in series_range])

X_train, X_test, y_train, y_test = train_test_split(X_series, y_series, test_size=0.2, shuffle=False)

inputs = Input((T, 2))
hid = Flatten()(inputs)
hid = Dense(16, activation='relu')(hid)
outputs = Dense(1)(hid)

model = Model(inputs, outputs)

model.compile('adam', 'mse', 'mae')

history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test))

plt.figure()
plt.title('Training and validation losses')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.figure()
plt.title('y_pred and y_true of training data')
plt.plot(y_train, label='y_true')
plt.plot(model.predict(X_train), label='y_pred')
plt.legend()
plt.show()

plt.figure()
plt.title('y_pred and y_true of testing data')
plt.plot(y_test, label='y_true')
plt.plot(model.predict(X_test), label='y_pred')
plt.legend()
plt.show()

# As we see, model is unable to make decent stock price predictions, so let's try a simpler task: predict which way the price will go
# --------------------------------------------#
# Boolean y values that represent a growth or a
y = np.array([1 if y_abs > 0 else 0 for y_abs in y_cont])


# --------------------------------------------#
# 10-Fold cross validation to decide how many data points to use to predict the next y value

def cross_val():
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=10)

    accs = []
    for T in range(1, 10):
        total_acc = 0

        for train_index, test_index in kf.split(X):
            X_train = X[train_index]
            y_train = y[train_index]

            train_range = range((len(train_index) - 1) // T)
            X_train = np.array([X_train[T * i:T * (i + 1)] for i in train_range])
            y_train = np.array([y_train[T * (i + 1)] for i in train_range])

            X_test = X[test_index]
            y_test = y[test_index]

            test_range = range((len(test_index) - 1) // T)
            X_test = np.array([X_test[T * i:T * (i + 1)] for i in test_range])
            y_test = np.array([y_test[T * (i + 1)] for i in test_range])

            inputs = Input((T, 2))
            hid = Flatten()(inputs)
            hid = Dense(64, activation='relu')(hid)
            outputs = Dense(1, activation='sigmoid')(hid)

            model = Model(inputs, outputs)

            model.compile('adam', 'binary_crossentropy', 'acc')

            history = model.fit(X_train, y_train, batch_size=16, epochs=30, verbose=0)

            loss, acc = model.evaluate(X_test, y_test)
            total_acc += acc

        print(f'div {T} done')
        accs.append(total_acc / 10)

    accs = np.array(accs)
    best_T = np.argmax(accs) + 1  # 5
    return best_T


# --------------------------------------------#
best_T = 5

eval_range = range((len(X) - 1) // best_T)
X_eval = np.array([X[best_T * i:best_T * (i + 1)] for i in eval_range])
y_eval = np.array([y[best_T * (i + 1)] for i in eval_range])

X_train, X_test, y_train, y_test = train_test_split(X_eval, y_eval, test_size=0.2, shuffle=True)

inputs = Input((best_T, 2))
hid = Flatten()(inputs)
hid = Dense(64, activation='relu')(hid)
outputs = Dense(1, activation='sigmoid')(hid)

model = Model(inputs, outputs)

model.compile('adam', 'binary_crossentropy', 'acc')

history = model.fit(X_train, y_train, batch_size=16, epochs=30, verbose=0)

plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['acc'], label='acc')
plt.legend()
plt.show()

loss, acc = model.evaluate(X_test, y_test)
print(acc)  # 68.3333

# The model can make predictions about the way the price goes throughout the day with ~68.333% accuracy
