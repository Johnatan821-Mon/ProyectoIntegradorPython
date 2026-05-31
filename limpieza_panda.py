
from __future__ import annotations

import pandas as pd

from pathlib import Path

# Intentar importar herramientas de graficado; si faltan, omitir gráficos pero seguir el flujo
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except Exception:
    PLOTTING_AVAILABLE = False
    plt = None
    sns = None
    print("Aviso: matplotlib/seaborn no disponibles — se omiten gráficos.")

BASE_DIR = Path(__file__).resolve().parent
INPUT = BASE_DIR / "dataset_banco.csv"
OUTPUT_CSV = BASE_DIR / "dataset_banco_limpio_desde_secuencia.csv"
BOXPLOTS_FILE = BASE_DIR / "boxplots_cols_num.png"
COUNTPLOTS_FILE = BASE_DIR / "countplots_cols_cat.png"


def main() -> None:
    # Lectura
    data = pd.read_csv(INPUT)

    print("Dataset original:", data.shape)
    print(data.head().to_string())

    # Info
    print("\nInfo del dataset:")
    data.info()

    # 4.1 Datos faltantes
    print("\nEliminando filas con valores faltantes...")
    data.dropna(inplace=True)
    data.info()

    # 4.2 Columnas irrelevantes (conteo de niveles)
    cols_cat = [
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "poutcome",
        "y",
    ]

    print("\nNiveles por columna categórica:")
    for col in cols_cat:
        if col in data.columns:
            print(f"Columna {col}: {data[col].nunique()} subniveles")

    # data.describe()
    print("\nResumen numérico:")
    print(data.describe())

    # 4.3 Filas repetidas
    print(f"\nTamaño antes de eliminar duplicados: {data.shape}")
    data.drop_duplicates(inplace=True)
    print(f"Tamaño después de eliminar duplicados: {data.shape}")

    # 4.4 Outliers -> graficar boxplots (si está disponible)
    cols_num = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
    if PLOTTING_AVAILABLE:
        fig, ax = plt.subplots(nrows=len(cols_num), ncols=1, figsize=(8, 5 * len(cols_num)))
        fig.subplots_adjust(hspace=0.5)

        for i, col in enumerate(cols_num):
            if col in data.columns:
                sns.boxplot(x=col, data=data, ax=ax[i])
                ax[i].set_title(col)

        fig.savefig(BOXPLOTS_FILE)
        plt.close(fig)
        print(f"Boxplots guardados en: {BOXPLOTS_FILE}")
    else:
        print("Omisión: matplotlib/seaborn no disponibles — no se generaron boxplots.")

    # Observaciones y filtrado
    print("\nFiltrando: age<=100, duration>0, previous<=100")
    print(f"Tamaño antes de filtrar: {data.shape}")
    if "age" in data.columns:
        data = data[data["age"] <= 100]
    if "duration" in data.columns:
        data = data[data["duration"] > 0]
    if "previous" in data.columns:
        data = data[data["previous"] <= 100]
    print(f"Tamaño después de filtrar: {data.shape}")

    # 4.5 Errores tipográficos en categóricas
    # Graficar niveles por categoria
    if PLOTTING_AVAILABLE:
        fig2, ax2 = plt.subplots(nrows=len(cols_cat), ncols=1, figsize=(10, 3 * len(cols_cat)))
        fig2.subplots_adjust(hspace=1)
        for i, col in enumerate(cols_cat):
            if col in data.columns:
                sns.countplot(x=col, data=data, ax=ax2[i])
                ax2[i].set_title(col)
                # Rotar etiquetas de eje x de forma segura
                ax2[i].tick_params(axis="x", rotation=30)

        fig2.savefig(COUNTPLOTS_FILE)
        plt.close(fig2)
        print(f"Countplots guardados en: {COUNTPLOTS_FILE}")
    else:
        print("Omisión: matplotlib/seaborn no disponibles — no se generaron countplots.")

    # Normalizar subniveles
    for column in cols_cat:
        if column in data.columns:
            data[column] = data[column].astype(str).str.lower()

    if "job" in data.columns:
        data["job"] = data["job"].str.replace("admin.", "administrative", regex=False)
    if "marital" in data.columns:
        data["marital"] = data["marital"].str.replace("div.", "divorced", regex=False)
    if "education" in data.columns:
        data["education"] = data["education"].str.replace("sec.", "secondary", regex=False)
        data.loc[data["education"] == "unk", "education"] = "unknown"
    if "contact" in data.columns:
        data.loc[data["contact"] == "phone", "contact"] = "telephone"
        data.loc[data["contact"] == "mobile", "contact"] = "cellular"
    if "poutcome" in data.columns:
        data.loc[data["poutcome"] == "unk", "poutcome"] = "unknown"

    print("\nSubniveles normalizados.")
    print(data.shape)

    # Guardar resultado final
    data.to_csv(OUTPUT_CSV, index=False)
    print(f"CSV limpio guardado en: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
