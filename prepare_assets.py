"""
Gera a base tratada e os artefatos do modelo usados pelo Streamlit.

Execute:
    python prepare_assets.py

Depois:
    streamlit run app.py
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import requests

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
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

    df = pd.DataFrame(resposta.json())
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

    payload = {
        "seriesid": ["CUSR0000SA0"],
        "startyear": str(ano_inicial),
        "endyear": str(ano_final),
    }

    resposta = requests.post(url, json=payload, timeout=60)
    resposta.raise_for_status()

    serie = resposta.json()["Results"]["series"][0]["data"]

    df = pd.DataFrame(serie)
    df = df[df["period"].str.match(r"M(0[1-9]|1[0-2])")].copy()

    df["mes"] = pd.to_datetime(
        df["year"] + "-" + df["period"].str.replace("M", "", regex=False) + "-01"
    )

    df["cpi_eua"] = pd.to_numeric(df["value"], errors="coerce")

    return df[["mes", "cpi_eua"]]


def construir_base():
    ipca = buscar_bcb_completo("433", inicio="01/01/2015")
    ipca = ipca.rename(columns={"data": "mes", "valor": "ipca"})

    dolar_diario = buscar_bcb_completo("1", inicio="01/01/2014")
    dolar_diario["mes"] = dolar_diario["data"].dt.to_period("M").dt.to_timestamp()

    dolar = (
        dolar_diario.groupby("mes", as_index=False)["valor"]
        .mean()
        .rename(columns={"valor": "dolar"})
    )
    dolar["var_dolar"] = dolar["dolar"].pct_change(fill_method=None) * 100

    selic = buscar_bcb_completo("4390", inicio="01/01/2015")
    selic = selic.rename(columns={"data": "mes", "valor": "selic"})

    ibc = buscar_bcb_completo("24364", inicio="01/01/2014")
    ibc = ibc.rename(columns={"data": "mes", "valor": "ibc_br"})
    ibc["var_ibc_br"] = ibc["ibc_br"].pct_change(fill_method=None) * 100

    cpi = pd.concat(
        [
            buscar_cpi_eua(2014, 2020),
            buscar_cpi_eua(2021, 2025),
        ],
        ignore_index=True,
    )

    cpi = (
        cpi.drop_duplicates(subset="mes")
        .sort_values("mes")
        .reset_index(drop=True)
    )
    cpi["inflacao_eua"] = cpi["cpi_eua"].pct_change(fill_method=None) * 100

    df = ipca.merge(dolar[["mes", "var_dolar"]], on="mes")
    df = df.merge(selic[["mes", "selic"]], on="mes")
    df = df.merge(ibc[["mes", "var_ibc_br"]], on="mes")
    df = df.merge(cpi[["mes", "inflacao_eua"]], on="mes")

    df = df[
        (df["mes"] >= "2015-01-01")
        & (df["mes"] <= "2025-12-01")
    ].copy()

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

    return df.dropna().reset_index(drop=True)


def main():
    print("Coletando e preparando os dados...")
    df = construir_base()

    X = df[VARIAVEIS]
    y = df["ipca"]

    corte = int(len(df) * 0.70)

    X_treino = X.iloc[:corte]
    X_teste = X.iloc[corte:]

    y_treino = y.iloc[:corte]
    y_teste = y.iloc[corte:]

    modelo = LinearRegression()
    modelo.fit(X_treino, y_treino)

    previsoes = modelo.predict(X_teste)

    metricas = {
        "MAE": float(mean_absolute_error(y_teste, previsoes)),
        "RMSE": float(np.sqrt(mean_squared_error(y_teste, previsoes))),
        "R2": float(r2_score(y_teste, previsoes)),
    }

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(DATA_PATH, index=False)
    joblib.dump(modelo, MODEL_PATH)
    joblib.dump(VARIAVEIS, VARIABLES_PATH)

    with METRICS_PATH.open("w", encoding="utf-8") as arquivo:
        json.dump(metricas, arquivo, ensure_ascii=False, indent=2)

    print("\nArquivos criados:")
    print(f"- {DATA_PATH}")
    print(f"- {MODEL_PATH}")
    print(f"- {VARIABLES_PATH}")
    print(f"- {METRICS_PATH}")

    print("\nMétricas:")
    print(f"MAE: {metricas['MAE']:.3f}")
    print(f"RMSE: {metricas['RMSE']:.3f}")
    print(f"R²: {metricas['R2']:.3f}")


if __name__ == "__main__":
    main()
