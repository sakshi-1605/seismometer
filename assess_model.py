import sys
import os

sys.path.append(os.path.abspath("src"))

from seismometer.seismometer import Seismogram

import pandas as pd
import numpy as np
import zipfile
import os
import glob
import pickle
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import LSTM, Dense, Bidirectional, BatchNormalization, Input, Add, Masking
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, auc

# Define Inclusion/Exclusion (I/E) criteria
ie_criteria = {
    "Age": {"exclude": list(range(0, 18))},  # Exclude patients younger than 18
    "SepsisLabel": {"include": [0, 1]},  # Include only valid labels
    "O2Sat": {"exclude": [None, np.nan]},  # Exclude missing oxygen saturation data
}

def apply_inclusion_exclusion(df, criteria):
    """Applies Inclusion/Exclusion criteria to filter the dataset."""
    for column, conditions in criteria.items():
        if column in df.columns:
            include_values = conditions.get("include", None)
            exclude_values = conditions.get("exclude", None)
            if include_values is not None:
                df = df[df[column].isin(include_values)]
            if exclude_values is not None:
                df = df[~df[column].isin(exclude_values)]
    return df

with zipfile.ZipFile('training_setB.zip', 'r') as zip_ref:
    zip_ref.extractall('training_setB')

files = glob.glob('training_setB/training_setB/*.psv')
df_list = [pd.read_csv(f, sep='|').assign(patient=f.split('/')[-1].split('.')[0]) for f in files]
df = pd.concat(df_list).reset_index(drop=True)

df = apply_inclusion_exclusion(df, ie_criteria)

df.to_pickle('filtered_combined.pkl')

patients_training_data = df['patient'].unique()
np.random.shuffle(patients_training_data)
patients_training_data = patients_training_data[:-6000]

df_mean_std = df[df['patient'].isin(patients_training_data)].describe().loc[['mean', 'std']]
df_mean_std.to_pickle('mean_std_scaling.pkl')

y_train = np.asarray(df['SepsisLabel'])
y_train = to_categorical(y_train)
X_train_cont = df[['HR', 'MAP', 'O2Sat', 'SBP', 'Resp']].fillna(method='bfill').fillna(method='ffill').values

input1 = Input(shape=(10, 5))
model1 = Bidirectional(LSTM(100, kernel_regularizer=l2(0.001), return_sequences=True))(input1)
model1 = Bidirectional(LSTM(75, kernel_regularizer=l2(0.001)))(model1)
model1 = Dense(35, activation='relu', kernel_regularizer=l2(0.001))(model1)
model1 = BatchNormalization()(model1)
model1 = Dense(15, activation='relu', kernel_regularizer=l2(0.001))(model1)
model1 = BatchNormalization()(model1)

output = Dense(2, activation='softmax', kernel_regularizer=l2(0.001))(model1)
model = Model(inputs=input1, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam')

checkpoint = ModelCheckpoint('model.keras', monitor='val_loss', save_best_only=True)
earlystop = EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(X_train_cont, y_train, batch_size=64, epochs=50, validation_split=0.2, callbacks=[checkpoint, earlystop])

y_pred = model.predict(X_train_cont)
print("Model Performance After I/E Filtering:")
print(classification_report(y_train.argmax(axis=1), y_pred.argmax(axis=1)))
print("AUC:", roc_auc_score(y_train[:, 1], y_pred[:, 1]))