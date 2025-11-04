# tracking.py

import mlflow
from contextlib import contextmanager

# Importa as constantes do seu módulo de configuração
try:
    from ..config import EXPERIMENT_NAME
except ImportError:
    print("Erro: Não foi possível importar EXPERIMENT_NAME de config.py.")
    EXPERIMENT_NAME = "Default Experiment"


@contextmanager
def log_run(params: dict, run_name: str):
    """
    Gerenciador de contexto para inicializar e finalizar uma run do MLflow.

    Args:
        params (dict): Dicionário de parâmetros a serem registrados na run.
        run_name (str): Nome descritivo da run.

    Yields:
        str: O ID da run que foi iniciada.
    """
    # Define o nome do experimento (se ele não existir, o MLflow o cria)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Inicia a run do MLflow
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        print(f"\n--- MLflow Run Iniciada ---")
        print(f"Experimento: {EXPERIMENT_NAME}")
        print(f"Run ID: {run_id}")

        # Log de Parâmetros
        mlflow.log_params(params)

        # Devolve o controle para o bloco 'with' do main.py
        try:
            yield run_id
        finally:
            # Esta seção é executada quando o bloco 'with' é finalizado
            mlflow.end_run()
            print("--- MLflow Run Finalizada ---\n")

# --- FIM DO ARQUIVO tracking.py ---