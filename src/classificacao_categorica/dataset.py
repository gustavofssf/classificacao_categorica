import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .config import DATA_RAW_PATH, DATA_PROCESSED_PATH


def load_data(file_name):
    """Carrega dados do arquivo CSV"""
    return pd.read_csv(DATA_RAW_PATH / file_name)


def preprocess_categorical_features(df, target_column):
    """Pré-processa variáveis categóricas"""
    data = df.copy()
    if 'id' in data.columns:
        data.drop('id', axis=1, inplace=True)

    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Codifica variáveis categóricas
    categorical_columns = X.select_dtypes(include=['object']).columns
    label_encoders = {}

    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # Codifica target se for categórico
    if y.dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
        label_encoders[target_column] = le_target

    # Salva dados processados
    processed_data = X.copy()
    processed_data[target_column] = y
    processed_data.to_csv(DATA_PROCESSED_PATH / 'processed_data.csv', index=False)

    return X, y, label_encoders