import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib
import os

# 1. Load Data
print("Loading data...")
df = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')

# 2. Convert to Binary Classification (0: No Diabetes, 1: Prediabetes/Diabetes)
print("Converting to binary classification...")
df['Diabetes_Binary'] = df['Diabetes_012'].replace({2: 1})
y = df['Diabetes_Binary']
X = df.drop(['Diabetes_012', 'Diabetes_Binary'], axis=1)

# 3. Feature Engineering
print("Feature engineering...")
X['BMI_Age'] = X['BMI'] * X['Age']
X['HighBP_HighChol'] = X['HighBP'] * X['HighChol']
X['GenHlth_PhysHlth'] = X['GenHlth'] * X['PhysHlth']

# 4. Preprocessing
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Build MLP Model
model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(64, activation='relu'),
    BatchNormalization(),
    
    Dense(1, activation='sigmoid') # Binary output
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

# 6. Train Model
print("Starting training...")
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

history = model.fit(X_train_scaled, y_train,
                    epochs=100,
                    batch_size=256,
                    validation_split=0.1,
                    callbacks=[early_stopping, reduce_lr],
                    verbose=1)

# 7. Evaluate
print("\nEvaluating model...")
loss, accuracy, auc = model.evaluate(X_test_scaled, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")
print(f"Test AUC: {auc:.4f}")

# 8. Save Model and Scaler
print("Saving model and scaler...")
model.save('diabetes_binary_mlp_model.keras')
joblib.dump(scaler, 'scaler.joblib')

# Predictions and Report
y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

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
    plt.savefig('training_history_binary.png')

plot_history(history)
