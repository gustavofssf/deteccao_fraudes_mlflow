# dataset.py

import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import warnings
import sys

# Importa as constantes do seu módulo de configuração
try:
    from .config import DATASET_NAME, TARGET_COLUMN, SAMPLES_TO_USE
except ImportError:
    # Caso o módulo seja executado de forma isolada (apenas para testes)
    print("Erro: Não foi possível importar constantes de config.py.")
    sys.exit(1)


def load_fraud_data():
    """
    Carrega o dataset de detecção de fraudes do Hugging Face.
    Devido ao tamanho, apenas uma amostra é carregada para agilizar o treinamento.
    """
    print(f"Carregando dataset: {DATASET_NAME}...")

    try:
        # Carrega o dataset (o download é gerenciado pela biblioteca datasets)
        dataset = load_dataset(DATASET_NAME, split='train', streaming=False)

        # Converte para Pandas e limita o número de amostras
        # O .take(N) é um método eficiente para amostragem inicial em datasets grandes
        df = pd.DataFrame(dataset.take(SAMPLES_TO_USE))

        print(f"Amostra carregada com sucesso: {len(df)} linhas.")
        return df

    except Exception as e:
        print(f"Erro ao carregar dados do Hugging Face: {e}")
        return pd.DataFrame()  # Retorna DataFrame vazio em caso de falha


def preprocess_and_split(df: pd.DataFrame, target_col: str):
    """
    Realiza o pré-processamento (Engenharia de Features e Codificação Categórica)
    e divide o dataset em conjuntos de treino e teste.
    """
    if df.empty:
        raise ValueError("DataFrame vazio, não é possível pré-processar.")

    print("Iniciando pré-processamento de dados...")

    # 1. DROP DE COLUNAS (nameOrig, nameDest, isFlaggedFraud)
    # nameOrig/nameDest são identificadores únicos, sem valor preditivo.
    # isFlaggedFraud é quase idêntica a isFraud para este dataset e pode causar vazamento de dados.
    cols_to_drop = ['nameOrig', 'nameDest', 'isFlaggedFraud']
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # 2. CODIFICAÇÃO ONE-HOT (Variável 'type')
    # Codifica as 5 categorias de transação (e.g., CASH_OUT, PAYMENT).
    df = pd.get_dummies(df, columns=['type'], drop_first=True)

    # 3. DEFINIÇÃO DE X (Features) e y (Target)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Verifica o desbalanceamento
    fraud_rate = y.mean() * 100
    print(f"Taxa de Fraude (Target): {fraud_rate:.4f}%")

    # 4. SPLIT E ESTRATIFICAÇÃO
    # Usamos stratify=y devido ao desbalanceamento severo (apenas 0.1% de fraude)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Dados prontos - Treino: {len(X_train)} amostras, Teste: {len(X_test)} amostras.")
    return X_train, X_test, y_train, y_test

# --- FIM DO ARQUIVO dataset.py ---