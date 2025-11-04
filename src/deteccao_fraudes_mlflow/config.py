# config.py

# ----------------------------------------------------
# 1. Configurações do Dataset (Hugging Face)
# ----------------------------------------------------
DATASET_NAME = "vitaliy-sharandin/synthetic-fraud-detection"
TARGET_COLUMN = "isFraud"
# Usamos 500k linhas para agilizar o treino, pois o dataset completo tem 6.3M
SAMPLES_TO_USE = 500000


# ----------------------------------------------------
# 2. Configurações do MLflow
# ----------------------------------------------------
EXPERIMENT_NAME = "Detecção Fraudes - Logistic Regression Baseline"


# ----------------------------------------------------
# 3. Parâmetros do Modelo (RandomForest - Semana 3)
# ----------------------------------------------------
MODEL_PARAMS = {
    # Parâmetros padrão para o Run 1 (Baseline)
    "n_estimators": 100,
    "max_depth": 10,  # Profundidade moderada para evitar overfitting imediato
    "random_state": 42,
    # Crucial para problemas de fraude: dá mais peso à classe minoritária (fraude)
    "class_weight": "balanced"
}