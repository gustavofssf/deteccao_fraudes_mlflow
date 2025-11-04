# graph_visualizer.py

import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os

# Importa o nome do experimento
try:
    from ..config import EXPERIMENT_NAME
except ImportError:
    print("Erro: Não foi possível importar EXPERIMENT_NAME de config.py.")
    EXPERIMENT_NAME = "Detecção Fraudes - Logistic Regression Baseline"  # Fallback


def create_and_log_metrics_graph():
    """
    Busca todas as runs do MLflow, plota o gráfico Precision vs. Recall
    e o salva como um artefato.
    """

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        print(f"Erro: Experimento '{EXPERIMENT_NAME}' não encontrado.")
        return

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    # 🚨 FILTRO PARA GARANTIR QUE APENAS AS RUNS COM AMBAS AS MÉTRICAS SEJAM USADAS
    # Filtramos as runs que não têm Precision/Recall (ex: runs de teste que falharam antes)
    runs = runs.dropna(subset=['metrics.precision', 'metrics.recall'])

    print(f"--- Gerando gráfico para {len(runs)} runs no experimento: {EXPERIMENT_NAME} ---")

    # --- PLOTAGEM DO GRÁFICO PRECISION VS. RECALL ---

    plt.figure(figsize=(10, 8))

    # Eixo Z: Usaremos o F1-Score para colorir os pontos (melhor F1 = cor mais quente)
    # Usamos o AUC-ROC (se existir) como tamanho do ponto
    f1_scores = runs["metrics.f1_score"]

    # O tamanho do ponto será mapeado ao AUC-ROC (quanto maior o AUC, maior o ponto)
    # Se auc_roc não for uniforme (o que ocorre), é melhor usar F1-Score
    sizes = (f1_scores * 200) + 50  # Escala para visualização

    scatter = plt.scatter(
        runs["metrics.precision"],
        runs["metrics.recall"],
        c=f1_scores,
        s=sizes,
        cmap='viridis',  # Mapa de cores
        alpha=0.7
    )

    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.title("Trade-off Precision vs. Recall (Tamanho = F1-Score)")
    plt.grid(True)

    # Adicionar anotações dos nomes das runs
    for index, row in runs.iterrows():
        plt.text(
            row["metrics.precision"],
            row["metrics.recall"],
            f' {row["tags.mlflow.runName"]}',
            fontsize=9,
            ha='left',
            va='center'
        )

    # Adicionar barra de cores para o F1-Score
    cbar = plt.colorbar(scatter)
    cbar.set_label('F1-Score')

    # Ajustar limites do eixo para clareza (0 a 1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # --- SALVAR E LOGAR COMO ARTEFATO ---

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "precision_vs_recall_tradeoff.png")
        plt.savefig(path)

        # Logamos o gráfico dentro de uma run (p.ex., a Run 2, que é a melhor)
        # Para logar artefatos, é necessário uma run ativa.
        # Salva o gráfico na pasta 'artifacts/' para o usuário ver.

        # Para fins de demonstração local, salvamos na raiz:
        final_path = os.path.join(os.getcwd(), "precision_vs_recall_tradeoff.png")
        plt.savefig(final_path)
        print(f"Gráfico salvo localmente em: {final_path}")

    plt.show()  # Opcional: mostrar na tela


if __name__ == "__main__":
    # Esta função não loga o artefato DENTRO de uma run existente,
    # mas a executa de forma autônoma para análise.
    create_and_log_metrics_graph()