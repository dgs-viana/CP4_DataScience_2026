# InflationScope Brasil

Projeto desenvolvido para o Checkpoint 4 da disciplina de
Data Science & Statistical Computing - FIAP.

## Aplicação online

A aplicação desenvolvida em Streamlit pode ser acessada pelo link:

🔗 https://inflationscope-brasil.streamlit.app/

## Repositório

🔗 https://github.com/dgs-viana/CP4_DataScience_2026

## Objetivo

O **InflationScope Brasil** tem como objetivo analisar o comportamento do **IPCA entre 2015 e 2025** e investigar sua relação com diferentes indicadores macroeconômicos.

Foram utilizadas variáveis como:

* variação do dólar;
* taxa Selic;
* IBC-Br;
* inflação dos Estados Unidos;
* IPCA do mês anterior;
* Selic do mês anterior.

Os dados foram obtidos de fontes oficiais, como **IBGE, Banco Central do Brasil e Bureau of Labor Statistics (BLS)**.

## Tratamento dos dados

Durante o projeto foram realizadas etapas de:

* coleta de dados por APIs;
* padronização de datas;
* tratamento de valores ausentes;
* cálculo de variações mensais;
* integração das diferentes bases;
* criação de variáveis defasadas;
* análise de correlação e multicolinearidade.

## Modelagem

Foram testados quatro modelos:

* Baseline;
* Regressão Linear Simples;
* Regressão Linear Múltipla;
* Regressão Polinomial.

Os modelos foram avaliados utilizando as métricas **MAE, RMSE e R²**.

### Resultados

| Modelo                    |       MAE |      RMSE |        R² |
| ------------------------- | --------: | --------: | --------: |
| Baseline                  |     0.251 |     0.333 |    -0.191 |
| Regressão Linear Simples  |     0.217 |     0.296 |     0.063 |
| Regressão Linear Múltipla |     0.219 | **0.289** | **0.106** |
| Regressão Polinomial      | **0.216** |     0.300 |     0.037 |

A **Regressão Linear Múltipla** apresentou o melhor desempenho geral considerando principalmente RMSE e R².

Os resultados também mostraram que o **IPCA do mês anterior** possui uma relação relevante com o comportamento da inflação atual.

## Aplicação Streamlit

O projeto possui uma aplicação desenvolvida em **Streamlit**, criada para apresentar de forma visual os principais dados, indicadores e resultados dos modelos.

## Tecnologias utilizadas

Python, Pandas, NumPy, Matplotlib, Scikit-learn, Statsmodels, Streamlit e Jupyter Notebook.

## Como executar

Clone o repositório:

```bash
git clone https://github.com/dgs-viana/CP4_DataScience_2026.git
```

Entre na pasta do projeto:

```bash
cd CP4_DataScience_2026
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute a aplicação:

```bash
streamlit run app.py
```

## Conclusão

Os modelos apresentaram desempenho superior ao baseline, porém os valores de R² indicam que a inflação depende de diversos fatores além das variáveis utilizadas.

O projeto possui finalidade acadêmica e demonstra a aplicação de técnicas de **análise de dados, estatística e Machine Learning** em um problema econômico real.

---

## Alunos

Felipe Viana - RM 565341
Felipe Bonilha - RM 562356
Joan Ferreira - RM 562913
Levi de Jesus - RM 563279
Luigi Borghi - RM 563096


**FIAP – Engenharia de Software**
**CP4 – Data Science & Statistical Computing**
**2026**
