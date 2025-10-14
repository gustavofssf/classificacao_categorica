import pandas as pd
import numpy as np
from pathlib import Path


def criar_dataset_exemplo():
    """Cria um dataset de exemplo para teste do pipeline"""
    np.random.seed(42)
    n_samples = 200

    data = {
        'id': range(1, n_samples + 1),
        'idade': np.random.randint(18, 70, n_samples),
        'categoria': np.random.choice(['A', 'B', 'C'], n_samples),
        'regiao': np.random.choice(['Norte', 'Sul', 'Leste', 'Oeste'], n_samples),
        'valor_gasto': np.round(np.random.normal(100, 30, n_samples), 2),
        'target': np.random.choice([0, 1], n_samples)  # Coluna target
    }

    df = pd.DataFrame(data)

    # Garantir que a pasta existe
    Path("data/raw").mkdir(parents=True, exist_ok=True)

    # Salvar em data/raw/
    df.to_csv("data/raw/dataset_exemplo.csv", index=False)

    print("✅ Dataset de exemplo criado com sucesso!")
    print("📁 Local: data/raw/dataset_exemplo.csv")
    print("🎯 Coluna target: 'target'")
    print("📊 Formato:", df.shape)
    print("\nPrimeiras linhas:")
    print(df.head())


if __name__ == "__main__":
    criar_dataset_exemplo()