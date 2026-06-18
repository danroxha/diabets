import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

"""
SCRIPT DE INFERÊNCIA - PREDIÇÃO DE RISCO DE DIABETES
Inferência: É a fase em que o modelo já treinado é usado para fazer previsões sobre novos dados reais.
"""

# 1. CARREGAMENTO DOS ARTEFATOS
print("Carregando modelo (MLP Profunda) e normalizador (Scaler)...")
# Scaler (Normalizador): Ajusta os dados de entrada para a mesma escala usada no treino (essencial para redes neurais)
model = tf.keras.models.load_model('diabetes_binary_mlp_model.keras')
scaler = joblib.load('scaler.joblib')

def predict_diabetes_risk(data):
    """
    Realiza a predição de risco com base nos dados do usuário.
    
    Importância: Os dados de entrada devem seguir exatamente a mesma ordem e 
    transformações utilizadas durante o treinamento (Seção 4.3 e 5 do artigo).
    """
    # Converter para DataFrame para facilitar a manipulação e engenharia de atributos
    features = pd.DataFrame([data])
    
    # Ordem rigorosa das colunas originais do BRFSS 2015
    cols = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 
            'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
            'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 
            'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']
    
    features = features[cols]
    
    # 2. ENGENHARIA DE ATRIBUTOS (Sync com train_binary.py)
    # Criamos as mesmas variáveis sintéticas que o modelo 'aprendeu' a interpretar
    features['BMI_Age'] = features['BMI'] * features['Age']
    features['HighBP_HighChol'] = features['HighBP'] * features['HighChol']
    features['GenHlth_PhysHlth'] = features['GenHlth'] * features['PhysHlth']
    
    # 3. NORMALIZAÇÃO
    # Utilizamos o transform() do scaler já ajustado aos dados de treino
    features_scaled = scaler.transform(features)
    
    # 4. INFERÊNCIA
    # O modelo retorna uma probabilidade (Ativação Sigmoid)
    prediction = model.predict(features_scaled, verbose=0)
    probability = float(prediction[0][0])
    
    return probability

# Exemplo de uso para teste local
if __name__ == "__main__":
    # Dados de exemplo representativos
    sample_data = {
        'HighBP': 1.0, 'HighChol': 1.0, 'CholCheck': 1.0, 'BMI': 30.0, 'Smoker': 0.0,
        'Stroke': 0.0, 'HeartDiseaseorAttack': 0.0, 'PhysActivity': 1.0, 'Fruits': 1.0,
        'Veggies': 1.0, 'HvyAlcoholConsump': 0.0, 'AnyHealthcare': 1.0, 'NoDocbcCost': 0.0,
        'GenHlth': 3.0, 'MentHlth': 0.0, 'PhysHlth': 0.0, 'DiffWalk': 0.0, 'Sex': 1.0,
        'Age': 9.0, 'Education': 5.0, 'Income': 6.0
    }

    risk_prob = predict_diabetes_risk(sample_data)
    print(f"\nProbabilidade de Risco de Diabetes: {risk_prob*100:.2f}%")
    
    # Threshold (Limiar de Decisão): Definimos 0.5 (50%) como o ponto de corte.
    # Em ferramentas de triagem, esse valor pode ser diminuído para aumentar a sensibilidade (pegar mais casos),
    # ou aumentado para ser mais conservador e evitar alarmes falsos.
    if risk_prob > 0.5:
        print("Resultado: ALTO RISCO DETECTADO - Recomenda-se triagem clínica.")
    else:
        print("Resultado: BAIXO RISCO DETECTADO.")
