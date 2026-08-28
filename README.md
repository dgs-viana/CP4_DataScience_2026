# InflationScope Brasil

Projeto acadêmico de **Data Science & Statistical Computing — FIAP 2026**.

O InflationScope Brasil analisa o comportamento mensal do IPCA entre 2015 e 2025 e investiga sua associação com:

- IPCA do mês anterior;
- inflação mensal dos Estados Unidos;
- variação mensal do dólar;
- Selic do mês anterior;
- variação mensal do IBC-Br.

O modelo final é uma **Regressão Linear Múltipla**.

## Fontes dos dados

- Banco Central do Brasil (SGS): IPCA, dólar, Selic e IBC-Br  
  https://api.bcb.gov.br/dados/serie/
- Bureau of Labor Statistics (BLS): CPI dos Estados Unidos  
  https://api.bls.gov/publicAPI/v1/timeseries/data/

Período principal analisado: **2015 a 2025**.

## Estrutura do projeto

```text
InflationScope_Streamlit/
├── app.py
├── prepare_assets.py
├── CP4_DataScience_.ipynb
├── requirements.txt
├── README.md
├── dados/
│   └── base_tratada.csv        # gerado automaticamente
└── modelo/
    ├── modelo_multiplo.pkl     # gerado automaticamente
    ├── variaveis.pkl           # gerado automaticamente
    └── metricas.json           # gerado automaticamente
```

## Instalação

Abra o terminal na pasta do projeto e execute:

```bash
pip install -r requirements.txt
```

## Forma mais simples de executar

A aplicação consegue buscar os dados oficiais e treinar o mesmo modelo utilizado no notebook caso os arquivos de `dados/` e `modelo/` ainda não existam.

Execute:

```bash
streamlit run app.py
```

O navegador deverá abrir automaticamente.

## Preparar os arquivos antes de abrir o Streamlit

Se preferir gerar a base e o modelo primeiro:

```bash
python prepare_assets.py
```

Depois:

```bash
streamlit run app.py
```

## Usando os arquivos exportados pelo notebook

A última célula do notebook `CP4_DataScience_REVISADO.ipynb` exporta:

```text
dados/base_tratada.csv
modelo/modelo_multiplo.pkl
modelo/variaveis.pkl
modelo/metricas.json
```

Se esses arquivos estiverem nas respectivas pastas, o Streamlit os carregará diretamente.

## O que a aplicação apresenta

A aplicação contém:

- título, problema e fontes;
- identificação da variável resposta e dos preditores;
- amostra da base;
- estatísticas descritivas;
- gráficos exploratórios;
- MAE, RMSE e R²;
- gráfico de valores reais x previstos;
- gráfico de resíduos;
- formulário para informar novas entradas;
- previsão do IPCA em %;
- aviso automático de extrapolação.

## Modelo

A separação é cronológica:

- 70% dos meses mais antigos para treino;
- 30% dos meses mais recentes para teste.

Variáveis do modelo:

```text
ipca_anterior
inflacao_eua
var_dolar
selic_anterior
var_ibc_br
```

O aplicativo utiliza a mesma organização das variáveis utilizada no notebook.

Resultados esperados, aproximadamente:

- MAE: 0,219
- RMSE: 0,289
- R²: 0,106

Pequenas diferenças só devem ocorrer caso as fontes oficiais sejam revisadas.

## Limitações

O modelo explica apenas parte das variações mensais do IPCA e não deve ser interpretado como uma previsão econômica definitiva ou como evidência de causalidade.

Outras variáveis, como combustíveis, alimentos, energia, commodities e expectativas de inflação, poderiam melhorar análises futuras.

Entradas fora das faixas históricas são consideradas extrapolação. A aplicação mostra um aviso nesses casos.

## Publicação no Streamlit Community Cloud

1. Coloque esta pasta em um repositório do GitHub.
2. Acesse o Streamlit Community Cloud.
3. Crie uma nova aplicação apontando para o repositório.
4. Informe `app.py` como arquivo principal.
5. Faça o deploy.

O arquivo `requirements.txt` já contém as dependências necessárias.

## Apresentação rápida

Durante a apresentação:

1. Mostre a aba **Visão geral**.
2. Explique rapidamente os dados e o gráfico do IPCA.
3. Vá para **Modelo** e mostre MAE, RMSE, R², real x previsto e resíduos.
4. Vá para **Nova previsão**.
5. Digite valores e clique em **Prever IPCA**.
6. Mostre o alerta de extrapolação alterando um valor para fora da faixa histórica.
