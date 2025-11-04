# train.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve
import mlflow
import warnings

# Importa as constantes do módulo de configuração
try:
    from .config import TARGET_COLUMN
except ImportError:
    # Apenas para garantir que o arquivo não quebre se for rodado isoladamente
    pass

def train_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, params: dict,
                run_id: str, threshold: float = 0.5): # Adicionado o parâmetro 'threshold'
    """
    Treina o modelo de Random Forest, avalia e registra com o MLflow, usando um limiar ajustável.
    """

    # Faz uma cópia dos parâmetros para não alterar o dicionário original
    current_params = params.copy()

    # Remove parâmetro do LogisticRegression que não é usado pelo RF
    if 'max_iter' in current_params:
        del current_params['max_iter']

    print("Iniciando treinamento do modelo (RandomForestClassifier)...")

    # 1. INICIALIZAÇÃO DO MODELO E TREINAMENTO
    # O `LogisticRegression` é ótimo como baseline para problemas binários, mas não é adequado para detecção de outliers.
    # Não usaremos mais 'max_iter=1000' para garantir a convergência devido ao grande volume de dados.
    model = RandomForestClassifier(**current_params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    # 2. PREDIÇÃO DE PROBABILIDADES
    # Probabilidades de pertencer à classe 1 (Fraude)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 3. AJUSTE DO LIMIAR
    # Prediz a classe baseado no limiar (ex: 0.05)
    y_pred = (y_proba >= threshold).astype(int)

    # 4. AVALIAÇÃO DO MODELO
    # Métricas de classificação (agora baseadas no novo limiar)
    metrics = {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred)  # Registramos a acc, mas não é a métrica principal.
    }

    # Métrica de área (independente do limiar)
    try:
        auc_roc = roc_auc_score(y_test, y_proba)
        metrics["auc_roc"] = auc_roc
    except ValueError:
        # Pode ocorrer se a classe positiva não estiver presente
        metrics["auc_roc"] = 0.0

    # 5. REGISTRO NO MLFLOW (Obrigatório, pois estamos na função de treino)
    for key, value in metrics.items():
        mlflow.log_metric(key, value)

    # Registra o limiar usado para a avaliação da Run
    mlflow.log_param("evaluation_threshold", threshold)

    mlflow.sklearn.log_model(model, "random_forest_model")

    print(f"Treinamento e logging do modelo finalizados. Run ID: {run_id}")
    return model, metrics

# --- FIM DO ARQUIVO train.py ---