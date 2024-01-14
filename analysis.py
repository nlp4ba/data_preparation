import pandas as pd
import numpy as np
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
df = pd.read_csv(dir_path + '/cleaned_recipes.csv', delimiter=',',  on_bad_lines='skip')

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
print(len(df['description']))
vectorizer = CountVectorizer()
#data = vectorizer.fit_transform(df['description'])

tdata = vectorizer.fit_transform(df['description'])
ft = vectorizer.get_feature_names_out()  
newDF  = pd.DataFrame.sparse.from_spmatrix(tdata, columns=ft)

newDF['duration'] = df['full_duration_numeric']

from sklearn.model_selection import train_test_split
train, test = train_test_split(newDF, test_size=0.3)

#from sklearn.linear_model import LogisticRegression
#m = LogisticRegression(penalty=None)
#m.fit(data, df['full_duration_numeric'])

from keras.models import Model, Sequential
from keras.layers import LSTM, Dense
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus: 
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=5292)]
    )


import sklearn as sk
import keras
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

check_gpu = len(tf.config.list_physical_devices('GPU'))>0

print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
print("GPU is", "available" if check_gpu \
      else "NOT AVAILABLE")

model = Sequential()

print(newDF.shape)

textData = train.loc[:, train.columns!='duration']

model.add(LSTM(4, input_dim = textData.shape[1], input_length = textData.shape[0]))
model.add(Dense(1, activation='relu'))

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.allow_growth = True

logical_gpus = tf.config.list_logical_devices('GPU')
print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")

history = model.fit(textData, train['duration'],
              batch_size=1, epochs=3,
              verbose = 1)