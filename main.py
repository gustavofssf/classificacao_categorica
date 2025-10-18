import sys
import os

# Adiciona o src (código-fonte) ao caminho para importações absolutas
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from classificacao_categorica.dataset import load_data, process_and_split
from classificacao_categorica.modeling.train import train_model
from classificacao_categorica.modeling.tracking import log_run
from classificacao_categorica.config import MODEL_PARAMS


def main():
    print("Iniciando pipeline de classificação...")

    try:
        # Carrega e pré-processa dados
        # template
        df = load_data('dataset_exemplo.csv')
        X_train, X_test, y_train, y_test = process_and_split(df, 'target')

        print("Dados carregados e pré-processados com sucesso!")

        # Treina modelo com dados já divididos
        model, accuracy, f1, X_test, y_test = train_model(X_train, X_test, y_train, y_test, params=MODEL_PARAMS)
        print(f"Modelo treinado - Acurácia: {accuracy:.4f}, F1: {f1:.4f}")

        # Log primeira rodada
        metrics_1 = {"accuracy": accuracy, "f1_score": f1}
        log_run(model, MODEL_PARAMS, metrics_1, "Run_1_Baseline")
        print("Primeiro run registrado no MLflow")

        # Segunda rodada com parâmetros diferentes
        params_2 = MODEL_PARAMS.copy()
        params_2["n_estimators"] = 200
        params_2["max_depth"] = 10

        # Treina o modelo 2 passando os novos parâmetros
        model_2, accuracy_2, f1_2, X_test_2, y_test_2 = train_model(X_train, X_test, y_train, y_test, params=params_2)
        print(f"Modelo 2 treinado - Acurácia: {accuracy_2:.4f}, F1: {f1_2:.4f}")
        metrics_2 = {"accuracy": accuracy_2, "f1_score": f1_2}
        log_run(model_2, params_2, metrics_2, "Run_2_Deeper_Trees")
        print("Segundo run registrado no MLflow")

        print("Pipeline executado com sucesso!")

    except Exception as e:
        print(f"Erro durante a execução: {e}")
        print("Verifique se:")
        print("1. Seu dataset está em data/raw/")
        print("2. O nome do arquivo e coluna target estão corretos")
        print("3. As dependências estão instaladas: pip install -r requirements.txt")


if __name__ == "__main__":
    main()