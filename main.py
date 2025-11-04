# main.py

import os
import sys

# Garante que o diretório raiz do projeto esteja no path para as importações
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from deteccao_fraudes_mlflow.config import TARGET_COLUMN, MODEL_PARAMS
from deteccao_fraudes_mlflow.dataset import load_fraud_data, preprocess_and_split
from deteccao_fraudes_mlflow.modeling.train import train_model
from deteccao_fraudes_mlflow.modeling.tracking import log_run


def main():
    print("Iniciando pipeline de detecção de fraudes...")

    # 1. CARREGAMENTO E PRÉ-PROCESSAMENTO (Hugging Face)
    try:
        # Carrega os dados (o download é feito aqui)
        df = load_fraud_data()

        # Verifica se o DataFrame foi carregado (pode falhar se não houver internet)
        if df.empty:
            print("Execução abortada devido a erro no carregamento dos dados.")
            return

        # Pré-processa e divide
        X_train, X_test, y_train, y_test = preprocess_and_split(df, TARGET_COLUMN)
        print("Dados carregados e pré-processados com sucesso!")

    except Exception as e:
        print(f"Erro na fase de carregamento/pré-processamento: {e}")
        return

    # 2. TREINAMENTO, AVALIAÇÃO E MLFLOW (Run 1 - RF Baseline: Depth 10, Peso Balanceado)
    with log_run(MODEL_PARAMS, "Run_1_RF_Baseline") as run_id_1:
        # O modelo será Random Forest no novo train.py
        model_1, metrics_1 = train_model(X_train, X_test, y_train, y_test, params=MODEL_PARAMS, run_id=run_id_1)

        print(f"Run 1 - Métricas: Precision: {metrics_1['precision']:.4f}, Recall: {metrics_1['recall']:.4f}, F1: {metrics_1['f1_score']:.4f}")

    # 3. TREINAMENTO, AVALIAÇÃO E MLFLOW (Run 2 - Otimizado: Mais Profundidade)
    params_2 = MODEL_PARAMS.copy()
    params_2["max_depth"] = 15  # Tentativa de aumentar o Recall para capturar mais padrões de fraude
    params_2["n_estimators"] = 150  # Pequeno aumento nas árvores

    # Definição da variável de Limiar (Threshold)
    THRESHOLD_RUN_2 = 0.10

    with log_run(params_2, "Run_2_RF_Deeper_Trees_T05") as run_id_2:
        model_2, metrics_2 = train_model(X_train, X_test, y_train, y_test, params=params_2, run_id=run_id_2, threshold=THRESHOLD_RUN_2)

        print(f"Run 2 - Métricas (Threshold {THRESHOLD_RUN_2}): Precision: {metrics_2['precision']:.4f}, Recall: {metrics_2['recall']:.4f}, F1: {metrics_2['f1_score']:.4f}, AUC: {metrics_2['auc_roc']:.4f}")

    # 4. TREINAMENTO, AVALIAÇÃO E MLFLOW (Run 3 - Teste de Velocidade/Simplicidade)
    # Testamos um modelo mais rápido/simples para comparar custo/benefício
    params_3 = MODEL_PARAMS.copy()
    params_3["max_depth"] = 8  # Menos profundidade
    params_3["n_estimators"] = 50  # Menos árvores para um treinamento mais rápido

    with log_run(params_3, "Run_3_RF_Fast_Simple") as run_id_3:
        model_3, metrics_3 = train_model(X_train, X_test, y_train, y_test, params=params_3, run_id=run_id_3)

        print(f"Run 3 - Métricas (Fast/Simple): Precision: {metrics_3['precision']:.4f}, Recall: {metrics_3['recall']:.4f}, F1: {metrics_3['f1_score']:.4f}")

    print("Pipeline de detecção de fraudes executado com sucesso!")

if __name__ == "__main__":
    main()