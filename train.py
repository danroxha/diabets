import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import joblib
import os

"""
TREINAMENTO DO MODELO MULTICLASSE (0, 1, 2) - DIABETES
Este script utiliza a mesma arquitetura profunda validada no artigo para a classificação original de 3 classes.
"""

# 1. CARREGAMENTO DOS DADOS
print("Carregando dataset...")
df = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')

# 2. ENGENHARIA DE ATRIBUTOS
# Mantemos as mesmas interações para consistência com o modelo binário
print("Executando engenharia de atributos...")
df['BMI_Age'] = df['BMI'] * df['Age']
df['HighBP_HighChol'] = df['HighBP'] * df['HighChol']
df['GenHlth_PhysHlth'] = df['GenHlth'] * df['PhysHlth']

# 3. PRÉ-PROCESSAMENTO
X = df.drop('Diabetes_012', axis=1)
y = df['Diabetes_012']

# Divisão Estratificada
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Conversão para One-Hot Encoding (Necessário para a função de perda categorical_crossentropy)
y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)

# 4. CONSTRUÇÃO DA REDE NEURAL PROFUNDA (MLP)
# Arquitetura espelhada do modelo binário (Seção 5 do artigo) para manter o poder de extração
model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    
    # Camadas de Alta Capacidade
    Dense(1024, activation='relu'),
    BatchNormalization(), # Estabiliza o treinamento
    Dropout(0.4),        # Regularização para evitar Overfitting (quando o modelo decora o ruído do treino e falha em dados novos)
    
    Dense(1024, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    
    # Camadas de Compressão Progressiva
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
    Dropout(0.2),
    
    # Camada de Saída Multiclasse: 3 Neurônios (Saudável, Pré-diabetes, Diabetes)
    # Importância: A função 'Softmax' converte os sinais da rede em probabilidades que somam 100%.
    # Isso permite que vejamos qual classe o modelo considera mais provável para aquele indivíduo.
    Dense(3, activation='softmax')
])

# Loss (Função de Perda): 'categorical_crossentropy' calcula o erro penalizando predições confiantes que estão erradas.
# É o "custo" que o modelo tenta minimizar durante o aprendizado.
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5. TREINAMENTO
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

print("Iniciando treinamento multiclasse...")
history = model.fit(X_train_scaled, y_train_cat,
                    epochs=100,
                    batch_size=256,
                    validation_split=0.1,
                    callbacks=[early_stopping, reduce_lr],
                    verbose=1)

# 6. AVALIAÇÃO
print("\nAvaliação no conjunto de teste:")
loss, accuracy = model.evaluate(X_test_scaled, y_test_cat)
print(f"Acurácia Global: {accuracy*100:.2f}%")

# 7. SALVAMENTO
print("Salvando modelo e scaler...")
model.save('diabetes_mlp_model.keras')
joblib.dump(scaler, 'scaler.joblib')

# Relatório de Classificação
y_pred = np.argmax(model.predict(X_test_scaled), axis=1)
print("\nRelatório de Classificação (0: Saudável, 1: Pré-diabetes, 2: Diabetes):")
print(classification_report(y_test, y_pred))

# Plotagem
def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.title('Acurácia')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Perda')
    plt.legend()
    plt.savefig('training_history.png')

plot_history(history)
print("\nTreinamento concluído. Gráfico salvo como 'training_history.png'.")
