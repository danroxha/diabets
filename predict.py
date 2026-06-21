import time

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

"""
SCRIPT DE INFERÊNCIA - PREDIÇÃO DE RISCO DE DIABETES
Inferência: É a fase em que o modelo já treinado é usado para fazer previsões sobre novos dados reais.
"""

# 1. CARREGAMENTO DOS ARTEFATOS
print("Carregando modelo (MLP Profunda) e normalizador (Scaler)...")
model = tf.keras.models.load_model("diabetes_binary_mlp_model.keras")
scaler = joblib.load("scaler.joblib")


def predict_diabetes_risk(data):
    """
    Realiza a predição de risco com base nos dados do usuário.

    Importância: Os dados de entrada devem seguir exatamente a mesma ordem e
    transformações utilizadas durante o treinamento (Seção 4.3 e 5 do artigo).
    """
    # Converter para DataFrame para facilitar a manipulação e engenharia de atributos
    features = pd.DataFrame([data])

    # Ordem rigorosa das colunas originais do BRFSS 2015
    cols = [
        "HighBP",
        "HighChol",
        "CholCheck",
        "BMI",
        "Smoker",
        "Stroke",
        "HeartDiseaseorAttack",
        "PhysActivity",
        "Fruits",
        "Veggies",
        "HvyAlcoholConsump",
        "AnyHealthcare",
        "NoDocbcCost",
        "GenHlth",
        "MentHlth",
        "PhysHlth",
        "DiffWalk",
        "Sex",
        "Age",
        "Education",
        "Income",
    ]

    features = features[cols]

    # 2. ENGENHARIA DE ATRIBUTOS (Sync com train_binary.py)
    features["BMI_Age"] = features["BMI"] * features["Age"]
    features["HighBP_HighChol"] = features["HighBP"] * features["HighChol"]
    features["GenHlth_PhysHlth"] = features["GenHlth"] * features["PhysHlth"]

    # 3. NORMALIZAÇÃO
    features_scaled = scaler.transform(features)

    # 4. INFERÊNCIA (retorna a probabilidade)
    prediction = model.predict(features_scaled, verbose=0)
    probability = float(prediction[0][0])

    return probability


# Exemplo de uso para teste local com benchmark
if __name__ == "__main__":
    # Dados de exemplo representativos
    sample_data = {
        "HighBP": 1.0,
        "HighChol": 1.0,
        "CholCheck": 1.0,
        "BMI": 30.0,
        "Smoker": 0.0,
        "Stroke": 0.0,
        "HeartDiseaseorAttack": 0.0,
        "PhysActivity": 1.0,
        "Fruits": 1.0,
        "Veggies": 1.0,
        "HvyAlcoholConsump": 0.0,
        "AnyHealthcare": 1.0,
        "NoDocbcCost": 0.0,
        "GenHlth": 3.0,
        "MentHlth": 0.0,
        "PhysHlth": 0.0,
        "DiffWalk": 0.0,
        "Sex": 1.0,
        "Age": 9.0,
        "Education": 5.0,
        "Income": 6.0,
    }

    # ---- Primeira chamada (cold start) ----
    with tf.device("/GPU:0"):
        start = time.perf_counter_ns()
        risk_prob = predict_diabetes_risk(sample_data)
        end = time.perf_counter_ns()
        cold_time_ms = (end - start) / 1_000_000
        print(f"Tempo da primeira inferência (cold start): {cold_time_ms:.3f} ms")

        # ---- Aquecimento (warm-up) ----
        print("Aquecendo o modelo (10 chamadas)...")
        for _ in range(10):
            _ = predict_diabetes_risk(sample_data)

        # ---- Medição real (média de 1000 chamadas) ----
        print("Medindo desempenho com 1000 chamadas...")
        start = time.perf_counter_ns()
        iterations = 1000
        for _ in range(iterations):
            _ = predict_diabetes_risk(sample_data)
        end = time.perf_counter_ns()

        avg_time_ms = (end - start) / iterations / 1_000_000
        print(f"Tempo médio por inferência (após aquecimento): {avg_time_ms:.3f} ms")

        # ---- Resultado final ----
        print(f"\nProbabilidade de Risco de Diabetes: {risk_prob * 100:.2f}%")

        # Threshold (Limiar de Decisão): 0.5 (50%)
        if risk_prob > 0.5:
            print("Resultado: ALTO RISCO DETECTADO - Recomenda-se triagem clínica.")
        else:
            print("Resultado: BAIXO RISCO DETECTADO.")
