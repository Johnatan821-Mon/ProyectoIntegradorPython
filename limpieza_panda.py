
from __future__ import annotations  # Habilita anotaciones de tipo futuras (compatibilidad)

import pandas as pd  # Importa pandas para manejo de DataFrame

from pathlib import Path  # Importa Path para manejar rutas de archivos

# Intento de importar librerías de graficado; si fallan, el script sigue sin gráficos
try:
    import matplotlib  # biblioteca base para graficado
    matplotlib.use("Agg")  # usa backend no interactivo para guardar figuras en disco
    import matplotlib.pyplot as plt  # interfaz de pyplot para crear figuras
    import seaborn as sns  # seaborn para gráficos estadísticos más bonitos
    PLOTTING_AVAILABLE = True  # bandera que indica que podemos graficar
except Exception:
    PLOTTING_AVAILABLE = False  # si hay cualquier error, no graficamos
    plt = None  # dejamos plt en None para evitar referencias fallidas
    sns = None  # idem para seaborn
    print("Aviso: matplotlib/seaborn no disponibles — se omiten gráficos.")  # aviso al usuario

# Definición de rutas y archivos de entrada/salida
BASE_DIR = Path(__file__).resolve().parent  # directorio base del script
INPUT = BASE_DIR / "dataset_banco.csv"  # archivo CSV original de entrada
OUTPUT_CSV = BASE_DIR / "dataset_banco_limpio_desde_secuencia.csv"  # CSV de salida limpio
BOXPLOTS_FILE = BASE_DIR / "boxplots_cols_num.png"  # ruta para guardar boxplots
COUNTPLOTS_FILE = BASE_DIR / "countplots_cols_cat.png"  # ruta para guardar countplots


def main() -> None:
    # Lectura del archivo CSV de entrada en un DataFrame
    data = pd.read_csv(INPUT)  # carga los datos desde la ruta INPUT

    print("Dataset original:", data.shape)  # muestra número de filas y columnas
    print(data.head().to_string())  # muestra las primeras filas en formato texto

    # Info del DataFrame: tipos y valores no nulos
    print("\nInfo del dataset:")
    data.info()  # imprime info sobre columnas, tipos y nulos

    # 4.1 Datos faltantes: eliminar filas que contengan NaN
    print("\nEliminando filas con valores faltantes...")
    data.dropna(inplace=True)  # elimina filas con cualquier valor faltante
    data.info()  # muestra info actualizada tras dropna

    # 4.2 Columnas categóricas a inspeccionar (conteo de niveles)
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
    ]  # lista de columnas categóricas de interés

    print("\nNiveles por columna categórica:")
    for col in cols_cat:
        if col in data.columns:
            print(f"Columna {col}: {data[col].nunique()} subniveles")  # cuenta subniveles únicos

    # Resumen estadístico numérico básico
    print("\nResumen numérico:")
    print(data.describe())  # describe estadísticas para columnas numéricas

    # 4.3 Filas repetidas: eliminar duplicados exactos
    print(f"\nTamaño antes de eliminar duplicados: {data.shape}")
    data.drop_duplicates(inplace=True)  # elimina filas duplicadas
    print(f"Tamaño después de eliminar duplicados: {data.shape}")

    # 4.4 Outliers -> graficar boxplots para columnas numéricas si es posible
    cols_num = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
    if PLOTTING_AVAILABLE:
        fig, ax = plt.subplots(nrows=len(cols_num), ncols=1, figsize=(8, 5 * len(cols_num)))
        fig.subplots_adjust(hspace=0.5)  # espacio vertical entre subplots

        for i, col in enumerate(cols_num):
            if col in data.columns:
                sns.boxplot(x=col, data=data, ax=ax[i])  # dibuja boxplot para la columna
                ax[i].set_title(col)  # título con el nombre de la columna

        fig.savefig(BOXPLOTS_FILE)  # guarda la figura en disco
        plt.close(fig)  # cierra la figura para liberar memoria
        print(f"Boxplots guardados en: {BOXPLOTS_FILE}")
    else:
        print("Omisión: matplotlib/seaborn no disponibles — no se generaron boxplots.")

    # Observaciones y filtrado: eliminar valores no plausibles en columnas clave
    print("\nFiltrando: age<=100, duration>0, previous<=100")
    print(f"Tamaño antes de filtrar: {data.shape}")
    if "age" in data.columns:
        data = data[data["age"] <= 100]  # filtra edades mayores a 100
    if "duration" in data.columns:
        data = data[data["duration"] > 0]  # filtra duraciones no positivas
    if "previous" in data.columns:
        data = data[data["previous"] <= 100]  # filtra valores previos excesivos
    print(f"Tamaño después de filtrar: {data.shape}")

    # 4.5 Errores tipográficos en categóricas: visualizar niveles por categoría
    if PLOTTING_AVAILABLE:
        fig2, ax2 = plt.subplots(nrows=len(cols_cat), ncols=1, figsize=(10, 3 * len(cols_cat)))
        fig2.subplots_adjust(hspace=1)  # ajusta el espacio entre subplots
        for i, col in enumerate(cols_cat):
            if col in data.columns:
                sns.countplot(x=col, data=data, ax=ax2[i])  # cuenta y grafica frecuencias
                ax2[i].set_title(col)  # título con el nombre de la columna
                # Rotar etiquetas de eje x de forma segura
                ax2[i].tick_params(axis="x", rotation=30)  # rota etiquetas para legibilidad

        fig2.savefig(COUNTPLOTS_FILE)  # guarda los countplots
        plt.close(fig2)  # cierra la figura
        print(f"Countplots guardados en: {COUNTPLOTS_FILE}")
    else:
        print("Omisión: matplotlib/seaborn no disponibles — no se generaron countplots.")

    # Normalizar subniveles: convertir a minúsculas para unificación
    for column in cols_cat:
        if column in data.columns:
            data[column] = data[column].astype(str).str.lower()  # pasa todo a minúsculas

    # Correcciones puntuales sobre etiquetas conocidas para homogeneizar
    if "job" in data.columns:
        data["job"] = data["job"].str.replace("admin.", "administrative", regex=False)  # corrige abreviatura
    if "marital" in data.columns:
        data["marital"] = data["marital"].str.replace("div.", "divorced", regex=False)  # corrige abreviatura
    if "education" in data.columns:
        data["education"] = data["education"].str.replace("sec.", "secondary", regex=False)  # corrige abreviatura
        data.loc[data["education"] == "unk", "education"] = "unknown"  # reemplaza 'unk' por 'unknown'
    if "contact" in data.columns:
        data.loc[data["contact"] == "phone", "contact"] = "telephone"  # unifica 'phone'
        data.loc[data["contact"] == "mobile", "contact"] = "cellular"  # unifica 'mobile'
    if "poutcome" in data.columns:
        data.loc[data["poutcome"] == "unk", "poutcome"] = "unknown"  # unifica 'unk' a 'unknown'

    print("\nSubniveles normalizados.")
    print(data.shape)  # muestra la forma final del DataFrame

    # Guardar resultado final en un CSV sin índice
    data.to_csv(OUTPUT_CSV, index=False)  # escribe datos limpios en OUTPUT_CSV
    print(f"CSV limpio guardado en: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
