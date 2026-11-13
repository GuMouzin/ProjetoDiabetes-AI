import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import matplotlib.pyplot as plt

import pickle

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(BASE_DIR, '2025.1.dataset.diabetes.br.categorico.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'modelo.ia.diabetes.h5')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.modelo.ia.diabetes.pkl')

df = pd.read_csv(CSV_PATH)

X = df.drop('diabetes', axis=1)
y = df['diabetes']

scaler = StandardScaler()
X_normalizado = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_normalizado, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()


history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

loss, acc = model.evaluate(X_test, y_test)
print(f"Acurácia no teste: {acc:.4f}")

plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia durante o treino')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.grid(True)
plt.show()

model.save(MODEL_PATH)

### verificacao pickle ###

with open(SCALER_PATH, 'wb') as fScaler:
    pickle.dump(scaler, fScaler)

print(type(scaler))

