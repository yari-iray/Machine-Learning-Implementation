import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

df_all = pd.read_csv('Machine-Learning-Implementation/Machine Learning Project/milknew.csv')

#normalize the data
df_all.iloc[:,:-1] = (df_all.iloc[:,:-1] - df_all.iloc[:,:-1].mean()) / df_all.iloc[:,:-1].std()

#move the Grade column to the end
cols = list(df_all)
cols.insert(len(cols), cols.pop(cols.index('Grade')))
df_all = df_all.loc[:, cols]

#change grades low medium and high to 0 1 and 2
df_all['Grade'] = df_all['Grade'].map({'low':0, 'medium':1, 'high':2})

print(df_all.head())

model = tf.keras.Sequential([
            tf.keras.layers.Dense(64,input_shape=(7,), activation='softmax'),
            tf.keras.layers.Dense(30, activation='sigmoid'),
            tf.keras.layers.Dense(3, activation='softmax')
            ])

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)

model.compile(optimizer=optimizer,
                loss=keras.losses.MeanSquaredError(),
                metrics=['accuracy'])

model.fit(
        df_all.iloc[:-10,0:-1].values, 
        df_all.iloc[:-10,-1].values, 
        batch_size=1,
        epochs=10)
