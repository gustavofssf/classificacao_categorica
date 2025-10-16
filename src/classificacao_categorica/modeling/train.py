from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def train_model(X_train, X_test, y_train, y_test, params):
    """Treina modelo de classificação usando dados já divididos.

    Args:
        X_train (pd.DataFrame): Features de treino
        X_test (pd.DataFrame): Features de teste
        y_train (pd.Series): Target de treino
        y_test (pd.Series): Target de teste
        params (dict): Dicionário de hiperparâmetros do modelo.

    Returns:
        tuple: (model, accuracy, f1, X_test, y_test)
"""
    # Filtra apenas parâmetros válidos para RandomForestClassifier  #
    valid_params = {
        'n_estimators': params.get('n_estimators', 100),
        'max_depth': params.get('max_depth', None),
        'random_state': params.get('random_state', None)
    }

    # Remove parâmetros None (usar defaults do RandomForest)
    valid_params = {k: v for k, v in valid_params.items() if v is not None}

    model = RandomForestClassifier(**valid_params)
    model.fit(X_train, y_train)

    # Predições e métricas
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    return model, accuracy, f1, X_test, y_test