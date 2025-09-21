import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load and prepare data
df = pd.read_csv('cardio_train.csv', delimiter=';')
df = df.drop(['id'], axis=1)
X = df.drop('cardio', axis=1)
y = df['cardio']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build and compile model
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
single_patient = X_test[0].reshape(1, -1)
probability = model.predict(single_patient)[0][0]
prediction = "Yes" if probability >= 0.5 else "No"
print(f"Predicted probability: {probability:.2f}, Prediction: {prediction}")


# Train model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
