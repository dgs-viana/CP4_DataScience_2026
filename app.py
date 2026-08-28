from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


# Configuração


st.set_page_config(
    page_title="InflationScope Brasil",
    page_icon="📈",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dados" / "base_tratada.csv"
MODEL_PATH = BASE_DIR / "modelo" / "modelo_multiplo.pkl"
VARIABLES_PATH = BASE_DIR / "modelo" / "variaveis.pkl"
METRICS_PATH = BASE_DIR / "modelo" / "metricas.json"

VARIAVEIS = [
    "ipca_anterior",
    "inflacao_eua",
    "var_dolar",
    "selic_anterior",
    "var_ibc_br",
]

ROTULOS = {
    "ipca_anterior": "IPCA do mês anterior (%)",
    "inflacao_eua": "Inflação dos EUA (%)",
    "var_dolar": "Variação mensal do dólar (%)",
    "selic_anterior": "Selic do mês anterior (%)",
    "var_ibc_br": "Variação mensal do IBC-Br (%)",
}


# Coleta dos dados


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    reraise=True,
)
def buscar_bcb(codigo, data_inicial, data_final):
    url = (
        f"https://api.bcb.gov.br/dados/serie/"
        f"bcdata.sgs.{codigo}/dados"
    )

    parametros = {
        "formato": "json",
        "dataInicial": data_inicial,
        "dataFinal": data_final,
    }

    resposta = requests.get(url, params=parametros, timeout=60)
    resposta.raise_for_status()

    dados = resposta.json()
    df = pd.DataFrame(dados)

    if df.empty:
        raise ValueError(f"A série {codigo} não retornou dados.")

    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")

    return df


def buscar_bcb_completo(codigo, inicio="01/01/2014"):
    parte1 = buscar_bcb(codigo, inicio, "31/12/2020")
    parte2 = buscar_bcb(codigo, "01/01/2021", "31/12/2025")

    return (
        pd.concat([parte1, parte2], ignore_index=True)
        .drop_duplicates(subset="data")
        .sort_values("data")
        .reset_index(drop=True)
    )


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    reraise=True,
)
def buscar_cpi_eua(ano_inicial, ano_final):
    url = "https://api.bls.gov/publicAPI/v1/timeseries/data/"

    dados = {
        "seriesid": ["CUSR0000SA0"],
        "startyear": str(ano_inicial),
        "endyear": str(ano_final),
    }

    resposta = requests.post(url, json=dados, timeout=60)
    resposta.raise_for_status()

    conteudo = resposta.json()

    try:
        serie = conteudo["Results"]["series"][0]["data"]
    except (KeyError, IndexError, TypeError) as erro:
        raise ValueError("Não foi possível interpretar a resposta da API do BLS.") from erro

    df = pd.DataFrame(serie)
    df = df[df["period"].str.match(r"M(0[1-9]|1[0-2])")].copy()

    df["mes"] = pd.to_datetime(
        df["year"] + "-" + df["period"].str.replace("M", "", regex=False) + "-01"
    )

    df["cpi_eua"] = pd.to_numeric(df["value"], errors="coerce")

    return df[["mes", "cpi_eua"]]


def construir_base():
    # IPCA - Brasil
    ipca = buscar_bcb_completo("433", inicio="01/01/2015")
    ipca = ipca.rename(columns={"data": "mes", "valor": "ipca"})

    # Dólar - média mensal e variação mensal
    dolar_diario = buscar_bcb_completo("1", inicio="01/01/2014")
    dolar_diario["mes"] = dolar_diario["data"].dt.to_period("M").dt.to_timestamp()

    dolar = (
        dolar_diario.groupby("mes", as_index=False)["valor"]
        .mean()
        .rename(columns={"valor": "dolar"})
    )
    dolar["var_dolar"] = dolar["dolar"].pct_change(fill_method=None) * 100

    # Selic acumulada no mês
    selic = buscar_bcb_completo("4390", inicio="01/01/2015")
    selic = selic.rename(columns={"data": "mes", "valor": "selic"})

    # IBC-Br com ajuste sazonal
    ibc = buscar_bcb_completo("24364", inicio="01/01/2014")
    ibc = ibc.rename(columns={"data": "mes", "valor": "ibc_br"})
    ibc["var_ibc_br"] = ibc["ibc_br"].pct_change(fill_method=None) * 100

    # CPI dos EUA
    cpi_eua = pd.concat(
        [
            buscar_cpi_eua(2014, 2020),
            buscar_cpi_eua(2021, 2025),
        ],
        ignore_index=True,
    )

    cpi_eua = (
        cpi_eua.drop_duplicates(subset="mes")
        .sort_values("mes")
        .reset_index(drop=True)
    )

    cpi_eua["inflacao_eua"] = (
        cpi_eua["cpi_eua"].pct_change(fill_method=None) * 100
    )

    # Junção
    df = ipca.merge(dolar[["mes", "var_dolar"]], on="mes")
    df = df.merge(selic[["mes", "selic"]], on="mes")
    df = df.merge(ibc[["mes", "var_ibc_br"]], on="mes")
    df = df.merge(cpi_eua[["mes", "inflacao_eua"]], on="mes")

    df = df[
        (df["mes"] >= "2015-01-01")
        & (df["mes"] <= "2025-12-01")
    ].copy()

    # Mesmas variáveis criadas no notebook
    df["ipca_anterior"] = df["ipca"].shift(1)
    df["selic_anterior"] = df["selic"].shift(1)

    df = df[
        [
            "mes",
            "ipca",
            "ipca_anterior",
            "inflacao_eua",
            "var_dolar",
            "selic_anterior",
            "var_ibc_br",
        ]
    ]

    df = df.dropna().reset_index(drop=True)

    return df



# Carregamento / treinamento


@st.cache_data(show_spinner=False)
def carregar_base():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        df["mes"] = pd.to_datetime(df["mes"])
        return df

    df = construir_base()

    try:
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
    except OSError:
        # Em alguns ambientes publicados o sistema de arquivos pode ser somente leitura.
        pass

    return df


def separar_treino_teste(df):
    X = df[VARIAVEIS]
    y = df["ipca"]

    corte = int(len(df) * 0.70)

    X_treino = X.iloc[:corte]
    X_teste = X.iloc[corte:]

    y_treino = y.iloc[:corte]
    y_teste = y.iloc[corte:]

    return X_treino, X_teste, y_treino, y_teste


@st.cache_resource(show_spinner=False)
def carregar_modelo():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    df = carregar_base()
    X_treino, _, y_treino, _ = separar_treino_teste(df)

    modelo = LinearRegression()
    modelo.fit(X_treino, y_treino)

    try:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(modelo, MODEL_PATH)
        joblib.dump(VARIAVEIS, VARIABLES_PATH)
    except OSError:
        pass

    return modelo


def avaliar_modelo(modelo, df):
    _, X_teste, _, y_teste = separar_treino_teste(df)

    previsoes = modelo.predict(X_teste)
    residuos = y_teste.to_numpy() - previsoes

    mae = mean_absolute_error(y_teste, previsoes)
    rmse = np.sqrt(mean_squared_error(y_teste, previsoes))
    r2 = r2_score(y_teste, previsoes)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "X_teste": X_teste,
        "y_teste": y_teste,
        "previsoes": previsoes,
        "residuos": residuos,
    }


def formatar_base(df):
    exibicao = df.copy()
    exibicao["mes"] = exibicao["mes"].dt.strftime("%d/%m/%Y")
    return exibicao



# Interface



# Carregar CSS


def carregar_css(arquivo):
    with open(arquivo, "r", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

carregar_css("style.css")

with st.sidebar:
    st.markdown("## InflationScope")
    st.caption("Painel de análise da inflação brasileira")
    st.markdown("---")
    st.markdown("**FIAP 2026**")
    st.caption("Data Science & Statistical Computing")
    st.markdown("**Período:** 2015 — 2025")

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Painel econômico • Brasil</div>
        <div class="hero-title">Inflação brasileira em perspectiva</div>
        <div class="hero-text">
            O InflationScope acompanha o comportamento mensal do IPCA e compara
            a inflação brasileira com indicadores econômicos nacionais e internacionais.
        </div>
        <span class="chip">2015 — 2025</span>
        <span class="chip">129 meses</span>
        <span class="chip">Regressão Linear Múltipla</span>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    with st.spinner("Carregando dados e modelo..."):
        df = carregar_base()
        modelo = carregar_modelo()
        avaliacao = avaliar_modelo(modelo, df)
except Exception as erro:
    st.error(
        "Não foi possível carregar os dados. "
        "Verifique sua conexão com a internet ou execute o notebook revisado "
        "e mantenha o arquivo `dados/base_tratada.csv` dentro do projeto."
    )
    st.exception(erro)
    st.stop()


aba_visao, aba_modelo, aba_previsao = st.tabs(
    ["📊 Panorama", "🧠 Modelo", "🎯 Simulador"]
)



# Aba 1 — Visão geral


with aba_visao:
    st.subheader("Panorama do período")

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("IPCA médio", f"{df['ipca'].mean():.2f}%")
    p2.metric("Maior IPCA", f"{df['ipca'].max():.2f}%")
    p3.metric("Menor IPCA", f"{df['ipca'].min():.2f}%")
    p4.metric("Meses analisados", f"{len(df)}")

    st.markdown(
        """
        <div class="section-box">
            <h4>O que este painel mostra?</h4>
            <p>
                Primeiro observamos como a inflação brasileira variou ao longo do tempo.
                Depois comparamos o IPCA com indicadores econômicos para verificar quais
                apresentaram maior relação com suas mudanças mensais.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Problema e dados")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(
            """
            **Variável resposta (y)**  
            - IPCA mensal do Brasil (%)

            **Variáveis usadas na previsão (X)**  
            - IPCA do mês anterior  
            - Inflação dos Estados Unidos  
            - Variação mensal do dólar  
            - Selic do mês anterior  
            - Variação mensal do IBC-Br
            """
        )

    with col_b:
        st.markdown(
            """
            **Fontes oficiais**
            - Banco Central do Brasil (SGS): IPCA, dólar, Selic e IBC-Br
            - Bureau of Labor Statistics (BLS): CPI dos Estados Unidos

            **Período:** 2015 a 2025  
            **Unidade de observação:** mês
            """
        )

    st.markdown(
        """
        - Banco Central do Brasil: https://api.bcb.gov.br/dados/serie/
        - Bureau of Labor Statistics: https://api.bls.gov/publicAPI/v1/timeseries/data/
        """
    )

    st.subheader("Amostra da base")
    amostra = formatar_base(df.head(10)).rename(columns={
        "mes": "Data",
        "ipca": "IPCA (%)",
        "ipca_anterior": "IPCA anterior (%)",
        "inflacao_eua": "Inflação EUA (%)",
        "var_dolar": "Variação dólar (%)",
        "selic_anterior": "Selic anterior (%)",
        "var_ibc_br": "Variação IBC-Br (%)",
    })
    st.dataframe(
        amostra.round(3),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Estatísticas descritivas")
    estatisticas = (
        df.select_dtypes(include="number")
        .describe()
        .T
        .round(3)
    )
    st.dataframe(estatisticas, use_container_width=True)

    st.subheader("Análise exploratória")

    # Gráfico 1
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(df["mes"], df["ipca"])
    ax1.set_title("Evolução mensal do IPCA - 2015 a 2025")
    ax1.set_xlabel("Ano")
    ax1.set_ylabel("IPCA mensal (%)")
    ax1.grid(alpha=0.3)
    st.pyplot(fig1)
    plt.close(fig1)

    st.caption(
        "O gráfico mostra períodos de inflação mais elevada e também meses "
        "com valores negativos, caracterizando deflação."
    )

    col1, col2 = st.columns(2)

    with col1:
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.hist(df["ipca"], bins=20, edgecolor="black")
        ax2.set_title("Distribuição do IPCA mensal")
        ax2.set_xlabel("IPCA mensal (%)")
        ax2.set_ylabel("Número de meses")
        st.pyplot(fig2)
        plt.close(fig2)

        st.caption(
            "A maior parte dos meses está concentrada em faixas relativamente "
            "baixas de inflação, com alguns valores mais extremos."
        )

    with col2:
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        ax3.scatter(df["ipca_anterior"], df["ipca"])
        ax3.set_title("IPCA anterior x IPCA atual")
        ax3.set_xlabel("IPCA do mês anterior (%)")
        ax3.set_ylabel("IPCA atual (%)")
        ax3.grid(alpha=0.3)
        st.pyplot(fig3)
        plt.close(fig3)

        st.caption(
            "Existe tendência positiva entre o IPCA anterior e o atual, "
            "embora haja dispersão entre os pontos."
        )



# Aba 2 — Modelo


with aba_modelo:
    st.subheader("Desempenho do modelo final")

    c1, c2, c3 = st.columns(3)

    c1.metric("MAE", f"{avaliacao['mae']:.3f} p.p.")
    c2.metric("RMSE", f"{avaliacao['rmse']:.3f} p.p.")
    c3.metric("R²", f"{avaliacao['r2']:.3f}")

    st.info(
        "O MAE indica o erro absoluto médio em pontos percentuais. "
        "O R² relativamente baixo mostra que o modelo explica apenas parte "
        "do comportamento mensal da inflação."
    )

    st.subheader("IPCA real x IPCA previsto")

    y_teste = avaliacao["y_teste"]
    previsoes = avaliacao["previsoes"]

    fig4, ax4 = plt.subplots(figsize=(7, 4))
    ax4.scatter(y_teste, previsoes)

    minimo = min(float(y_teste.min()), float(previsoes.min()))
    maximo = max(float(y_teste.max()), float(previsoes.max()))

    ax4.plot(
        [minimo, maximo],
        [minimo, maximo],
        linestyle="--",
    )

    ax4.set_title("IPCA real x IPCA previsto")
    ax4.set_xlabel("IPCA real (%)")
    ax4.set_ylabel("IPCA previsto (%)")
    ax4.grid(alpha=0.3)

    st.pyplot(fig4)
    plt.close(fig4)

    st.caption(
        "Quanto mais próximo um ponto estiver da linha diagonal, "
        "mais próxima a previsão ficou do valor real."
    )

    st.subheader("Resíduos")

    residuos = avaliacao["residuos"]

    fig5, ax5 = plt.subplots(figsize=(7, 4))
    ax5.scatter(previsoes, residuos)
    ax5.axhline(0, linestyle="--")
    ax5.set_title("Resíduos x valores previstos")
    ax5.set_xlabel("IPCA previsto (%)")
    ax5.set_ylabel("Resíduo")
    ax5.grid(alpha=0.3)

    st.pyplot(fig5)
    plt.close(fig5)

    st.caption(
        "Os resíduos representam a diferença entre o IPCA real e o previsto. "
        "O gráfico ajuda a verificar padrões nos erros."
    )

    st.subheader("Exemplos no conjunto de teste")

    indices_teste = avaliacao["X_teste"].index

    exemplos = pd.DataFrame(
        {
            "Data": df.loc[indices_teste, "mes"].dt.strftime("%d/%m/%Y").values,
            "IPCA real (%)": avaliacao["y_teste"].values,
            "IPCA previsto (%)": previsoes,
            "Erro absoluto (p.p.)": np.abs(residuos),
        }
    )

    st.dataframe(
        exemplos.round(3),
        use_container_width=True,
        hide_index=True,
    )



# Aba 3 — Previsão


with aba_previsao:
    st.subheader("Simulador de IPCA")

    st.write(
        """
        Informe os cinco indicadores abaixo. O modelo utilizará a mesma ordem
        de variáveis usada no notebook para estimar o **IPCA mensal (%)**.
        """
    )

    valores = {}

    col_esq, col_dir = st.columns(2)

    for i, variavel in enumerate(VARIAVEIS):
        minimo = float(df[variavel].min())
        maximo = float(df[variavel].max())
        mediana = float(df[variavel].median())

        coluna = col_esq if i % 2 == 0 else col_dir

        with coluna:
            valores[variavel] = st.number_input(
                ROTULOS[variavel],
                value=round(mediana, 3),
                step=0.01,
                format="%.3f",
                help=(
                    f"Faixa observada na base: "
                    f"{minimo:.3f} até {maximo:.3f}"
                ),
                key=f"entrada_{variavel}",
            )

    if st.button("Prever IPCA", type="primary", use_container_width=True):
        alertas = []

        for variavel, valor in valores.items():
            minimo = float(df[variavel].min())
            maximo = float(df[variavel].max())

            if valor < minimo or valor > maximo:
                alertas.append(
                    f"**{ROTULOS[variavel]}**: {valor:.3f} está fora "
                    f"do intervalo observado ({minimo:.3f} a {maximo:.3f})."
                )

        if alertas:
            st.warning(
                "Atenção: há entrada(s) fora do intervalo observado na base. "
                "Isso caracteriza extrapolação e a previsão pode ser menos confiável."
            )

            for alerta in alertas:
                st.markdown(f"- {alerta}")
        else:
            st.success(
                "Todos os valores informados estão dentro das faixas "
                "observadas na base histórica."
            )

        entrada = pd.DataFrame(
            [[valores[v] for v in VARIAVEIS]],
            columns=VARIAVEIS,
        )

        previsao = float(modelo.predict(entrada)[0])

        st.markdown(
            f"""
            <div style="background:linear-gradient(120deg,#FFFFFF,#EEF4F3); border:2px solid #3B7A68; border-radius:18px; padding:1.4rem; text-align:center; margin-top:1rem;">
                <div style="font-size:.8rem; font-weight:800; color:#567066; text-transform:uppercase; letter-spacing:.08em;">IPCA mensal estimado</div>
                <div style="font-size:2.7rem; font-weight:900; color:#1E5C4B;">{previsao:.3f}%</div>
                <div style="font-size:.82rem; color:#698078;">Estimativa gerada pelo modelo final</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "A previsão representa uma estimativa estatística e não deve ser "
            "interpretada como garantia do IPCA futuro ou como relação causal."
        )

        with st.expander("Ver valores utilizados"):
            tabela_entrada = pd.DataFrame(
                {
                    "Variável": [ROTULOS[v] for v in VARIAVEIS],
                    "Valor informado": [valores[v] for v in VARIAVEIS],
                    "Mínimo observado": [float(df[v].min()) for v in VARIAVEIS],
                    "Máximo observado": [float(df[v].max()) for v in VARIAVEIS],
                }
            )
            st.dataframe(
                tabela_entrada.round(3),
                use_container_width=True,
                hide_index=True,
            )


st.divider()
st.caption(
    "InflationScope Brasil • FIAP 2026 • "
    "Projeto acadêmico de Data Science & Statistical Computing"
)
