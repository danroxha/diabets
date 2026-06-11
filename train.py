import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import os

# 1. Load Data
print("Loading data...")
df = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')

# 2. Feature Engineering
print("Feature engineering...")
# Creating interaction terms
df['BMI_Age'] = df['BMI'] * df['Age']
df['HighBP_HighChol'] = df['HighBP'] * df['HighChol']
df['GenHlth_PhysHlth'] = df['GenHlth'] * df['PhysHlth']

# 3. Preprocessing
X = df.drop('Diabetes_012', axis=1)
y = df['Diabetes_012']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert labels to categorical (one-hot)
y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)

# 4. Build MLP Model
model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(64, activation='relu'),
    BatchNormalization(),
    
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5. Train Model
print("Starting training...")
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

history = model.fit(X_train_scaled, y_train_cat,
                    epochs=100,
                    batch_size=256,
                    validation_split=0.1,
                    callbacks=[early_stopping, reduce_lr],
                    verbose=1)

# 6. Evaluate
print("\nEvaluating model...")
loss, accuracy = model.evaluate(X_test_scaled, y_test_cat)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# 7. Save Model and Scaler
print("Saving model and scaler...")
model.save('diabetes_mlp_model.keras')
joblib.dump(scaler, 'scaler.joblib')

# Save history plot
def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.savefig('training_history.png')

plot_history(history)
