# 🚀 Pipeline de Classificação Categórica com MLflow

**Atividade desenvolvida na disciplina de Inteligência Artificial - Ano 2025**  
**Universidade Federal Fluminense**  
**Mestrado em Engenharia de Produção e Sistemas Computacionais**  
**Sob orientação do Prof. Leonard Barreto**

### 👥 Autores
- **Everton Ferreira de Lima**
- **Gustavo Francisco Sant'Anna dos Santos de França**

---

![Python](https://img.shields.io/badge/Python-3.13%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2%2B-orange)
![MLflow](https://img.shields.io/badge/MLflow-2.0%2B-lightgrey)
![Git](https://img.shields.io/badge/Git-Professional%20Workflow-green)
![Cookiecutter](https://img.shields.io/badge/Cookiecutter-Data%20Science-brightgreen)

Projeto de Data Science focado em Classificação Categórica, implementado através de um Pipeline Modular seguindo a arquitetura Cookiecutter. Demonstra competência em MLOps Básico e Rastreamento de Experimentos com MLflow.

## ✨ Recursos Chave e MLOps

- ✅ **Estrutura Cookiecutter Data Science** - Arquitetura profissional e modular
- ✅ **Pipeline completo** - Pré-processamento, treinamento e tracking
- ✅ **MLflow Integration** - Rastreamento de experimentos com métricas e parâmetros
- ✅ **Código modular** - Separação clara de responsabilidades
- ✅ **Reprodutibilidade** - Versionamento de dados e modelos

## 📁 Estrutura do Projeto

Para seguir o padrão Cookiecutter Data Science, o projeto possui a seguinte estrutura modular:

```tree
.
├── data/
│ ├── raw / # Dados brutos (dataset_exemplo.csv)
│ └── processed / # Dados processados (X_train, y_test, etc.)
├── src/
│   └── classificacao_categorica/
├── config.py         # Configurações centralizadas
│       ├── dataset.py        # Carregamento e Pré-processamento Categórico
│       └── modeling/
│           ├── train.py      # Lógica de Treinamento
│           └── tracking.py   # Rastreamento MLflow
├── main.py                   # Orquestrador do Pipeline (Entry Point)
└── requirements.txt          # Dependências do Ambiente
```

## ⚙️ Instalação e Configuração

### 1. Clone o repositório

```bash
git clone https://github.com/gustavofssf/classificacao_categorica
cd classificacao_categorica
```

### 2. Configure o ambiente virtual

# Crie um ambiente virtual
```bash
python -m venv .venv
```
# Ative o ambiente virtual
```bash
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

### 3. Instale as dependências

```bash
py -m pip install -r requirements.txt
```

## Executar o Pipeline

```bash
py main.py
```

## Visualizar os Resultados no MLflow

```bash
mlflow ui  
```
Acesse: **http://localhost:5000**

Para usar seu próprio dataset:
1. Coloque o arquivo CSV em `data/raw/`
2. Atualize a chamada em `main.py`:

```Python
df = load_data('seu_dataset.csv')
X_train, X_test, y_train, y_test = preprocess__and_split(df, 'coluna_target')
```

## 📈 Resultados e Rastreamento de Experimentos

O pipeline treinou e registrou **dois runs distintos** para fins de comparação, variando os hiperparâmetros do `RandomForestClassifier`.

**Modelo Treinado:** Random Forest

### Comparativo de Métricas

| Métrica | Run 1 (Baseline) | Run 2 (Otimizado) |
| :--- | :---: | :---: |
| **Acurácia** | 0.5250 | 0.5000 |
| **F1-Score (Weighted)** | 0.5357 | 0.5067 |

*Nota:* O Run 1 (Baseline) obteve métricas ligeiramente superiores neste conjunto de dados.

## Experimentos registrados no MLflow

| Run | Parâmetros Chave | Objetivo |
| :--- | :--- | :--- |
| **Run 1 (Baseline)** | `n_estimators=100`, `max_depth=5` | Ponto de partida para a otimização. |
| **Run 2 (Otimizado)** | `n_estimators=200, max_depth=10` | Teste de um modelo mais complexo. |

## 🏗️ Arquitetura Modular e Separação de Responsabilidades

O pipeline adere ao princípio de modularização, onde cada componente possui uma responsabilidade única:

- **`config.py`**: Centraliza **Hiperparâmetros**, caminhos de dados e configurações de **MLflow**.
- **`dataset.py`**: Responsável pelo I/O (carregamento/salvamento) e pelo **Pré-processamento de Variáveis Categóricas**.
- **`modeling/train.py`**: Contém a lógica de **Treinamento e Avaliação** do modelo (`RandomForestClassifier`).
- **`modeling/tracking.py`**: Dedicado à **Integração MLflow**, registrando métricas, parâmetros e o objeto do modelo.

## Bibliotecas utilizadas

Scikit-learn - Modelos de ML  
MLflow - Tracking de experimentos  
Pandas - Manipulação de dados  
Cookiecutter - Estrutura do projeto  


