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

- classificacao_categorica /
├── data /
│ ├── raw / # Dados brutos
│ └── processed / # Dados processados
├── src/classificacao_categorica /
│ ├── config.py / # Configurações centralizadas
│ ├── dataset.py / # Carregamento e pré-processamento
│ └── modeling /
│ ├── train.py / # Lógica de treinamento
│ └── tracking.py / # Rastreamento com MLflow
├── main.py / # Pipeline principal
├── requirements.txt / # Dependências
└── README.md

## ⚙️ Instalação e Configuração

### 1. Clone o repositório
```bash
git clone https://github.com/gustavofssf/classificacao_categorica
cd classificacao_categorica

### 2. Configure o ambiente virtual
# Crie um ambiente virtual 
python -m venv .venv

# Ative o ambiente virtual
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

###3. Instale as dependências
1. `py -m pip install -r requirements.txt`
## Executar o Pipeline
2. `py main.py`
## Visualizar os Resultados no MLFlow
3. `mlflow ui` para visualizar resultados

Acesse: http://localhost:5000

Para usar seu próprio dataset:
1. Coloque o arquivo CSV em data/raw/
2. Atualize no main.py:

df = load_data('seu_dataset.csv')
X_train, X_test, y_train, y_test = preprocess__and_split(df, 'coluna_target')

📈 Resultados e Rastreamento de Experimentos

O pipeline treinou e registrou dois runs distintos para fins de comparação.

**Modelo Treinado:** Random Forest

###Métricas

### Comparativo de Métricas

| Métrica | Run 1 (Baseline) | Run 2 (Otimizado) |
| :--- | :---: | :---: |
| **Acurácia** | 0.5250 | 0.5000 |
| **F1-Score (Weighted)** | 0.5357 | 0.5067 |

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


