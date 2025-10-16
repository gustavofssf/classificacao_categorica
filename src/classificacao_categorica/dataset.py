import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from .config import DATA_RAW_PATH, DATA_PROCESSED_PATH, DATA_SPLIT_PARAMS


def load_data(file_name):
    """Carrega dados do arquivo CSV"""
    return pd.read_csv(DATA_RAW_PATH / file_name)


def process_and_split(df, target_column):
    """Pré-processa variáveis categóricas, divide em treino/teste e salva."""
    data = df.copy()
    if 'id' in data.columns:
        data.drop('id', axis=1, inplace=True)

    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Codificação de variáveis categóricas (Aplicada ao X completo
    categorical_columns = X.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Codifica target (se for categórico)
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    # Divisão dos Dados (Usa parâmetros do config.py)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=DATA_SPLIT_PARAMS.get("test_size"),
        random_state=DATA_SPLIT_PARAMS.get("random_state")
    )
    # Salva dados processados e divididos
    X_train.to_csv(DATA_PROCESSED_PATH / 'X_train.csv', index=False)
    X_test.to_csv(DATA_PROCESSED_PATH / 'X_test.csv', index=False)
    # Salvando Series como CSV. Ajuste se necessário
    y_train.to_csv(DATA_PROCESSED_PATH / 'y_train.csv', index=False, header=True)
    y_test.to_csv(DATA_PROCESSED_PATH / 'y_test.csv', index=False, header=True)

    return X_train, X_test, y_train, y_test