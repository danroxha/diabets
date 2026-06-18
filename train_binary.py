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

"""
TREINAMENTO DO MODELO BINÁRIO - DIABETES TIPO II
Este script implementa a arquitetura de rede neural profunda (MLP) descrita no artigo:
'Predição de Risco de Diabetes tipo II via Redes Neurais Profundas: Um Estudo com o Dataset BRFSS 2015'
"""

# 1. CARREGAMENTO DOS DADOS
# Utilizamos o dataset BRFSS 2015 (Behavioral Risk Factor Surveillance System)
print("Carregando dataset...")
df = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')

# 2. CONVERSÃO PARA CLASSIFICAÇÃO BINÁRIA
# O dataset original possui 3 classes (0: saudável, 1: pré-diabetes, 2: diabetes)
# Para o estudo binário, agrupamos 1 e 2 como 'Risco/Diabetes' (1)
print("Convertendo para classificação binária (0: Saudável, 1: Risco/Diabetes)...")
df['Diabetes_Binary'] = df['Diabetes_012'].replace({2: 1})
y = df['Diabetes_Binary']
X = df.drop(['Diabetes_012', 'Diabetes_Binary'], axis=1)

# 3. ENGENHARIA DE ATRIBUTOS (Feature Engineering)
# Conforme seção 4.3 do artigo, criamos variáveis sintéticas para capturar interações entre fatores de risco
print("Executando engenharia de atributos...")
X['BMI_Age'] = X['BMI'] * X['Age']               # Interação entre IMC e Faixa Etária
X['HighBP_HighChol'] = X['HighBP'] * X['HighChol'] # Risco cardiovascular acumulado (Pressão Alta + Colesterol)
X['GenHlth_PhysHlth'] = X['GenHlth'] * X['PhysHlth'] # Relação entre percepção de saúde e incapacidade física

# 4. PRÉ-PROCESSAMENTO
# Divisão Estratificada: Mantém a proporção das classes nos conjuntos de treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalização: Essencial para Redes Neurais. Garante que todos os atributos estejam na mesma escala (Média 0, Desvio 1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. CONSTRUÇÃO DA ARQUITETURA DA REDE NEURAL (MLP)
# Conforme seção 5 do artigo, a rede possui 6 camadas ocultas para extração progressiva de características
model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    
    # Camadas de Alta Capacidade: Capturam relações complexas iniciais
    # Definição de Neurônios: 1024 em cada uma das duas primeiras camadas
    # Função de Ativação 'ReLU': Introduz não-linearidade e evita o problema do desaparecimento do gradiente
    Dense(1024, activation='relu'),
    BatchNormalization(), # Estabiliza o treinamento e acelera a convergência
    Dropout(0.4),        # Regularização: Desliga 40% dos neurônios aleatoriamente para evitar Overfitting (quando o modelo "decora" os dados de treino e perde a capacidade de generalizar para novos dados)
    
    Dense(1024, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    
    # Camadas de Compressão Progressiva: Reduzem a dimensionalidade para extrair fatores latentes
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
    
    # Camada de Saída: 1 Neurônio com ativação 'Sigmoid'
    # Importância: Transforma a saída em uma probabilidade entre 0 e 1 para classificação binária
    Dense(1, activation='sigmoid')
])

# Otimizador 'Adam': Um algoritmo inteligente que ajusta a velocidade do aprendizado para cada peso da rede.
# Loss 'binary_crossentropy': Mede o erro para decisões binárias (Sim/Não).
# AUC-ROC: Mede a capacidade do modelo de separar as classes. Diferente da acurácia, o AUC é robusto contra bases desequilibradas.
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

# Profundidade da Rede: Total de 6 camadas ocultas + 1 de saída (Rede Profunda/Deep Learning)
print(f"Modelo construído com {model.count_params()} parâmetros treináveis.")

# 6. TREINAMENTO
# Callbacks: Funções que monitoram e ajustam o treinamento automaticamente
# EarlyStopping (Parada Antecipada): Interrompe o treino se o erro de validação parar de cair, evitando desperdício de tempo e overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# ReduceLROnPlateau (Redução de Taxa de Aprendizado em Planaltos): Diminui o "tamanho do passo" (learning rate) se o modelo parar de evoluir
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

print("Iniciando treinamento (estimado em ~130 segundos)...")
history = model.fit(X_train_scaled, y_train,
                    epochs=100,      # Épocas: Número total de vezes que o modelo verá todo o conjunto de dados
                    batch_size=256,  # Batch Size: Número de amostras processadas antes de atualizar os pesos da rede
                    validation_split=0.1, # Parte dos dados (10%) usada para testar o modelo DURANTE o treino
                    callbacks=[early_stopping, reduce_lr],
                    verbose=1)

# 7. AVALIAÇÃO E RESULTADOS
print("\nAvaliação final no conjunto de teste:")
loss, accuracy, auc = model.evaluate(X_test_scaled, y_test)
print(f"Acurácia Global: {accuracy*100:.2f}% (Meta do Artigo: ~85.19%)")
print(f"AUC-ROC: {auc:.4f} (Meta do Artigo: 0.82)")

# 8. SALVAMENTO DOS ARTEFATOS
print("\nSalvando modelo e scaler para uso na aplicação web...")
model.save('diabetes_binary_mlp_model.keras')
joblib.dump(scaler, 'scaler.joblib')

# Relatório de Classificação Detalhado
y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Plotagem do Histórico de Treinamento
def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Treino (Acc)')
    plt.plot(history.history['val_accuracy'], label='Validação (Acc)')
    plt.title('Evolução da Acurácia')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Treino (Loss)')
    plt.plot(history.history['val_loss'], label='Validação (Loss)')
    plt.title('Evolução da Perda')
    plt.legend()
    plt.savefig('history_plot.png') # Nome do arquivo conforme citado no artigo

plot_history(history)
print("\nTreinamento concluído. Gráfico salvo como 'history_plot.png'.")
