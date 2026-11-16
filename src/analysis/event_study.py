from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data" / "processed" / "financial_returns.csv"
TABLES_DIR = ROOT / "outputs" / "tables"
FIGURES_DIR = ROOT / "outputs" / "figures"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# def eventos e janelas do estudo
EVENTS = {
    "Publicação da MP": pd.Timestamp("2026-05-12"),
    "Aprovação no Congresso": pd.Timestamp("2026-09-03"),
}

ESTIMATION_WINDOW = 120
EVENT_WINDOW = 5


def load_data():

    df = pd.read_csv(INPUT_FILE, parse_dates=["Data"])

    required_columns = ["Data", "Ret_LREN3", "Ret_IBOV"]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Colunas não encontradas na base: {missing_columns}"
        )

    df = df[required_columns].copy()
    df = df.sort_values("Data")

    if df["Data"].duplicated().any():
        raise ValueError("Foram encontradas datas duplicadas na base.")

    df = df.dropna(
        subset=["Data", "Ret_LREN3", "Ret_IBOV"]
    ).reset_index(drop=True)

    return df


def estimate_market_model(df):
    # id a primeira data do evento
    first_event = EVENTS["Publicação da MP"]

    event_index = df.index[df["Data"] == first_event]

    if len(event_index) != 1:
        raise ValueError(
            f"A data do evento {first_event.date()} não foi encontrada exatamente uma vez."
        )

    event_index = event_index[0]

    # def a janela de estimaçao antes da primeira janela de evento
    end_index = event_index - EVENT_WINDOW
    start_index = end_index - ESTIMATION_WINDOW

    estimation_data = df.iloc[start_index:end_index].copy()

    if len(estimation_data) != ESTIMATION_WINDOW:
        raise ValueError(
            "A janela de estimação não contém 120 observações."
        )

    # estima o modelo de mercado por MQO
    X = sm.add_constant(estimation_data["Ret_IBOV"])
    y = estimation_data["Ret_LREN3"]

    model = sm.OLS(y, X).fit()

    alpha = model.params["const"]
    beta = model.params["Ret_IBOV"]
    r_squared = model.rsquared

    print("\nMODELO DE MERCADO")
    print(
        f"Período de estimação: "
        f"{estimation_data['Data'].min().date()} a "
        f"{estimation_data['Data'].max().date()}"
    )
    print(f"Número de observações: {len(estimation_data)}")
    print(f"Alfa: {alpha:.6f}")
    print(f"Beta: {beta:.6f}")
    print(f"R²: {r_squared:.4f}")

    return alpha, beta, r_squared


def calculate_event_window(df, event_name, event_date, alpha, beta):
    # localiza a data do evento na serie de retornos
    event_index = df.index[df["Data"] == event_date]

    if len(event_index) != 1:
        raise ValueError(
            f"A data do evento {event_date.date()} não foi encontrada exatamente uma vez."
        )

    event_index = event_index[0]

    # seleciona os cinco pregoes anteriores e posteriores ao evento
    start_index = event_index - EVENT_WINDOW
    end_index = event_index + EVENT_WINDOW + 1

    window = df.iloc[start_index:end_index].copy()

    if len(window) != 11:
        raise ValueError(
            f"A janela do evento {event_name} não contém 11 pregões."
        )

    window["Event"] = event_name
    window["Relative_Day"] = range(-EVENT_WINDOW, EVENT_WINDOW + 1)

    # calcula o retorno esperado pelo modelo de mercado
    window["Expected_Return"] = (
        alpha + beta * window["Ret_IBOV"]
    )

    # calcula os retornos anormais e os retornos anormais acumulados
    window["Abnormal_Return"] = (
        window["Ret_LREN3"] - window["Expected_Return"]
    )

    window["CAR"] = window["Abnormal_Return"].cumsum()

    return window


def main():
    # carrega os dados e estima o modelo de mercado
    df = load_data()

    alpha, beta, r_squared = estimate_market_model(df)

    event_results = []

    # calcula os retornos anormais para cada evento
    for event_name, event_date in EVENTS.items():
        window = calculate_event_window(
            df,
            event_name,
            event_date,
            alpha,
            beta,
        )

        event_results.append(window)

    daily_results = pd.concat(event_results, ignore_index=True)

    # organiza os resultados finais de cada evento
    summary = (
        daily_results.groupby("Event")
        .agg(
            CAR_Final=("CAR", "last"),
        )
        .reset_index()
    )

    summary["Alfa"] = alpha
    summary["Beta"] = beta
    summary["R_quadrado"] = r_squared

    summary = summary[
        ["Event", "Alfa", "Beta", "R_quadrado", "CAR_Final"]
    ]

    # salva as tabelas com os resultados
    daily_results.to_csv(
        TABLES_DIR / "event_study_daily.csv",
        index=False,
        decimal=".",
    )

    summary.to_csv(
        TABLES_DIR / "event_study_summary.csv",
        index=False,
        decimal=".",
    )

    # gera o grafico comparativo dos retornos anormais acumulados
    plt.figure(figsize=(10, 6))

    for event_name in EVENTS:
        event_data = daily_results[
            daily_results["Event"] == event_name
        ]

        plt.plot(
            event_data["Relative_Day"],
            event_data["CAR"],
            marker="o",
            label=event_name,
        )

    plt.axhline(y=0, linewidth=0.8)
    plt.axvline(x=0, linestyle="--", linewidth=0.8)

    plt.title("Retornos Anormais Acumulados - Estudo de Eventos")
    plt.xlabel("Pregões em Relação à Data do Evento")
    plt.ylabel("Retorno Anormal Acumulado (%)")
    plt.xticks(range(-5, 6))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "event_study_car.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("\nRESUMO DO ESTUDO DE EVENTOS")
    print(summary.to_string(index=False))

    print("\narquivos gerados com sucesso")
    print(TABLES_DIR / "event_study_daily.csv")
    print(TABLES_DIR / "event_study_summary.csv")
    print(FIGURES_DIR / "event_study_car.png")


if __name__ == "__main__":
    main()
