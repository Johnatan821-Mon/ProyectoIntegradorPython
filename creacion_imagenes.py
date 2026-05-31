from __future__ import annotations
from pathlib import Path

import pandas as pd
import matplotlib
# Usamos el backend 'Agg' para poder generar archivos de imagen sin necesidad de
# una interfaz gráfica (útil en servidores o entornos sin DISPLAY).
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Directorio base del script y rutas a los datasets (original y limpio)
BASE_DIR = Path(__file__).resolve().parent
ORIGINAL = BASE_DIR / "dataset_banco.csv"
LIMPIO = BASE_DIR / "dataset_banco_limpio.csv"
GRAFICAS_DIR = BASE_DIR / "graficas"
GRAFICAS_DIR.mkdir(exist_ok=True)

raw = pd.read_csv(ORIGINAL)
clean = pd.read_csv(LIMPIO)


def annotate_bars(ax, fmt="{:.1f}%", offset=0.3, fontsize=9):
    """Anota cada barra en un `Axes` con su altura.

    - `ax`: objeto Axes de matplotlib que contiene las barras.
    - `fmt`: formato para el texto (p. ej. "{:.1f}%" o "€{:,.0f}").
    - `offset`: desplazamiento vertical para la etiqueta.
    - `fontsize`: tamaño de la fuente de la anotación.
    """
    for bar in ax.patches:
        # Calcular posición x (centro de la barra) y y (alto + offset)
        ax.annotate(
            fmt.format(bar.get_height()),
            (bar.get_x() + bar.get_width() / 2, bar.get_height() + offset),
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def conv_rate(df, col):
    """Calcula la tasa de conversión (%) por categoría de la columna `col`.

    Devuelve un DataFrame con columnas `[col, 'tasa']` ordenado descendente.
    """
    result = (
        df.groupby(col)["y"]
        .apply(lambda s: (s == "yes").mean() * 100)  # proporción de 'yes' * 100
        .sort_values(ascending=False)
        .reset_index()
    )
    result.columns = [col, "tasa"]
    return result


def save_fig(fig, filename):
    """Guarda la figura `fig` en `GRAFICAS_DIR/filename` y la cierra en memoria."""
    fig.savefig(GRAFICAS_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"Grafico guardado: graficas/{filename}")


# ---------------------------------------------------------------------------
# 1. COMPARACION: dataset original vs limpio
# ---------------------------------------------------------------------------
print("=" * 60)
print("COMPARACION ORIGINAL vs LIMPIO")
print("=" * 60)

nulos_raw = raw.isnull().sum().sum()      # total de celdas nulas en raw
nulos_clean = clean.isnull().sum().sum()  # total de celdas nulas en clean
dup_raw = raw.duplicated().sum()          # filas duplicadas en raw
dup_clean = clean.duplicated().sum()      # filas duplicadas en clean

print(f"{'Metrica':<35} {'Original':>10} {'Limpio':>10}")
print("-" * 57)
print(f"{'Filas':<35} {raw.shape[0]:>10,} {clean.shape[0]:>10,}")
print(f"{'Columnas':<35} {raw.shape[1]:>10} {clean.shape[1]:>10}")
print(f"{'Celdas nulas':<35} {nulos_raw:>10} {nulos_clean:>10}")
print(f"{'Filas duplicadas':<35} {dup_raw:>10} {dup_clean:>10}")
print(f"{'Filas eliminadas (total)':<35} {raw.shape[0]-clean.shape[0]:>10,} {'—':>10}")

# Columnas categóricas de interés para comparar niveles únicos
cols_cat = ["job", "marital", "education", "contact", "poutcome"]
print("\nNiveles unicos por columna categorica:")
print(f"{'Columna':<15} {'Original':>10} {'Limpio':>10}")
print("-" * 37)
for col in cols_cat:
    # Solo imprimimos si la columna existe en ambos DataFrames
    if col in raw.columns and col in clean.columns:
        print(f"{col:<15} {raw[col].nunique():>10} {clean[col].nunique():>10}")

# Gráficos comparativos: Filas, Nulos, Duplicados (Original vs Limpio)
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Comparacion Dataset Original vs Limpio", fontsize=14, fontweight="bold")
metricas = ["Filas", "Nulos", "Duplicados"]
vals_raw = [raw.shape[0], nulos_raw, dup_raw]
vals_clean = [clean.shape[0], nulos_clean, dup_clean]
colors = ["#4C72B0", "#55A868"]
for i, (metrica, vr, vc) in enumerate(zip(metricas, vals_raw, vals_clean)):
    ax = axes[i]
    bars = ax.bar(["Original", "Limpio"], [vr, vc], color=colors, edgecolor="white")
    ax.set_title(metrica)
    ax.bar_label(bars, fmt="{:,.0f}", padding=3)  # etiquetas con separador de miles
    ax.set_ylabel("Cantidad")
    ax.set_ylim(0, max(vr, vc) * 1.2 + 1)
plt.tight_layout()
save_fig(fig, "01_comparacion_datasets.png")


# ---------------------------------------------------------------------------
# 2. DISTRIBUCION DE LA VARIABLE OBJETIVO (y)
# ---------------------------------------------------------------------------
conv = clean["y"].value_counts(normalize=True).mul(100).round(1)  # porcentaje por clase
print(f"\nTasa de conversion global: {conv.get('yes', 0):.1f}% SI | {conv.get('no', 0):.1f}% NO")

fig, ax = plt.subplots(figsize=(6, 5))
# Pie chart con agujero central (donut-like) para la tasa global
wedges, texts, autotexts = ax.pie(
    conv.values,
    labels=conv.index.str.upper(),
    autopct="%1.1f%%",
    colors=["#55A868", "#C44E52"],
    startangle=90,
    pctdistance=0.75,
    wedgeprops=dict(width=0.5),
)
for t in autotexts:
    t.set_fontsize(13)
ax.set_title("Tasa de Conversion Global\n(suscripcion a deposito a termino)", fontsize=12)
save_fig(fig, "02_conversion_global.png")


# ---------------------------------------------------------------------------
# 3. TASA DE CONVERSION POR TIPO DE TRABAJO
# ---------------------------------------------------------------------------
job_conv = conv_rate(clean, "job")  # DataFrame con columnas ['job', 'tasa']

fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(data=job_conv, x="job", y="tasa", hue="job", palette="Blues_d", legend=False, ax=ax, order=job_conv["job"])
ax.set_title("Tasa de Conversion por Tipo de Trabajo (%)", fontsize=13, fontweight="bold")
ax.set_xlabel("Ocupacion")
ax.set_ylabel("% que suscribio")
ax.tick_params(axis="x", rotation=30)
annotate_bars(ax, fontsize=8)
plt.tight_layout()
save_fig(fig, "03_conversion_por_trabajo.png")

print(f"\nTop 3 ocupaciones con mayor conversion:")
for _, row in job_conv.head(3).iterrows():
    print(f"  {row['job']:<20} {row['tasa']:.1f}%")


# ---------------------------------------------------------------------------
# 4. DISTRIBUCION POR GRUPO DE EDAD y CONVERSION
# ---------------------------------------------------------------------------
# Definimos intervalos de edad y etiquetas
bins = [0, 25, 35, 45, 55, 65, 120]
labels = ["<25", "25-35", "35-45", "45-55", "55-65", "65+"]
clean = clean.copy()  # trabajamos sobre una copia para no modificar el original
clean["grupo_edad"] = pd.cut(clean["age"], bins=bins, labels=labels, right=False)

age_total = clean["grupo_edad"].value_counts().reindex(labels)  # total por grupo
age_conv = clean[clean["y"] == "yes"]["grupo_edad"].value_counts().reindex(labels).fillna(0)
age_rate = (age_conv / age_total * 100).round(1)  # tasa % por grupo de edad

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Analisis por Grupo de Edad", fontsize=13, fontweight="bold")

age_total_df = age_total.rename_axis("grupo_edad").reset_index(name="count")
age_rate_df = age_rate.rename_axis("grupo_edad").reset_index(name="tasa")

sns.barplot(data=age_total_df, x="grupo_edad", y="count", hue="grupo_edad", palette="Blues_d", legend=False, ax=axes[0])
axes[0].set_title("Cantidad de Clientes por Grupo de Edad")
axes[0].set_xlabel("Grupo de Edad")
axes[0].set_ylabel("Numero de Clientes")
annotate_bars(axes[0], fmt="{:,.0f}", offset=50)

sns.barplot(data=age_rate_df, x="grupo_edad", y="tasa", hue="grupo_edad", palette="Greens_d", legend=False, ax=axes[1])
axes[1].set_title("Tasa de Conversion por Grupo de Edad (%)")
axes[1].set_xlabel("Grupo de Edad")
axes[1].set_ylabel("% que suscribio")
annotate_bars(axes[1])

plt.tight_layout()
save_fig(fig, "04_analisis_por_edad.png")

print(f"\nTasa de conversion por grupo de edad:")
for g, r in age_rate.items():
    print(f"  {g:<8} {r:.1f}%")


# ---------------------------------------------------------------------------
# 5. CONVERSION POR EDUCACION Y ESTADO CIVIL
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Conversion por Educacion y Estado Civil", fontsize=13, fontweight="bold")

for ax, col, title in zip(axes, ["education", "marital"], ["Educacion", "Estado Civil"]):
    df_g = conv_rate(clean, col)  # tasa por nivel de la columna
    sns.barplot(data=df_g, x=col, y="tasa", hue=col, palette="Purples_d", legend=False, ax=ax)
    ax.set_title(f"Tasa de Conversion por {title} (%)")
    ax.set_xlabel(title)
    ax.set_ylabel("% que suscribio")
    ax.tick_params(axis="x", rotation=20)
    annotate_bars(ax)

plt.tight_layout()
save_fig(fig, "05_conversion_educacion_civil.png")


# ---------------------------------------------------------------------------
# 6. BALANCE PROMEDIO DE CLIENTES QUE SI vs NO SUSCRIBEN
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Saldo Promedio Anual (balance) por Resultado", fontsize=13, fontweight="bold")
yn_map = {"yes": "Si", "no": "No"}
palette_yn = {"Si": "#55A868", "No": "#C44E52"}

clean_disp = clean.assign(y=clean["y"].map(yn_map))

# Boxplot: muestra distribución (mediana, IQR, outliers) del balance por resultado
sns.boxplot(data=clean_disp, x="y", y="balance", hue="y", palette=palette_yn, legend=False, ax=axes[0])
axes[0].set_title("Distribucion del Saldo por Resultado")
axes[0].set_xlabel("Suscribio?")
axes[0].set_ylabel("Saldo promedio anual (EUR)")

# Barra: promedio del balance por resultado
bal_mean = clean_disp.groupby("y")["balance"].mean().reset_index()
bal_mean.columns = ["y", "balance_medio"]
sns.barplot(data=bal_mean, x="y", y="balance_medio", hue="y", palette=palette_yn, legend=False, ax=axes[1])
axes[1].set_title("Saldo Promedio por Resultado")
axes[1].set_xlabel("Suscribio?")
axes[1].set_ylabel("Saldo promedio (EUR)")
annotate_bars(axes[1], fmt="€{:,.0f}", offset=20, fontsize=10)

plt.tight_layout()
save_fig(fig, "06_balance_por_resultado.png")


# ---------------------------------------------------------------------------
# 7. DURACION DEL CONTACTO vs CONVERSION
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Duracion del Contacto Telefonico vs Conversion", fontsize=13, fontweight="bold")

for label, color in [("yes", "#55A868"), ("no", "#C44E52")]:
    # Histogramas superpuestos de la duración (convertimos a minutos)
    subset = clean[clean["y"] == label]["duration"] / 60
    axes[0].hist(subset, bins=40, alpha=0.6, label="Si" if label == "yes" else "No", color=color)
axes[0].set_title("Distribucion de la Duracion (minutos)")
axes[0].set_xlabel("Duracion (min)")
axes[0].set_ylabel("Frecuencia")
axes[0].legend()
axes[0].set_xlim(0, 30)

# Duracion promedio por resultado (en minutos)
dur_mean = clean_disp.groupby("y")["duration"].mean().div(60).reset_index()
dur_mean.columns = ["y", "duracion_min"]
sns.barplot(data=dur_mean, x="y", y="duracion_min", hue="y", palette=palette_yn, legend=False, ax=axes[1])
axes[1].set_title("Duracion Promedio del Contacto")
axes[1].set_xlabel("Suscribio?")
axes[1].set_ylabel("Duracion promedio (min)")
annotate_bars(axes[1], fmt="{:.1f} min", offset=0.1, fontsize=10)

plt.tight_layout()
save_fig(fig, "07_duracion_vs_conversion.png")


# ---------------------------------------------------------------------------
# 8. RESULTADO CAMPANA ANTERIOR (poutcome) vs CONVERSION ACTUAL
# ---------------------------------------------------------------------------
pout_conv = conv_rate(clean, "poutcome")

fig, ax = plt.subplots(figsize=(7, 5))
sns.barplot(data=pout_conv, x="poutcome", y="tasa", hue="poutcome", palette="Oranges_d", legend=False, ax=ax)
ax.set_title("Conversion segun Resultado Campana Anterior", fontsize=12, fontweight="bold")
ax.set_xlabel("Resultado campana anterior")
ax.set_ylabel("% que suscribio en campana actual")
ax.tick_params(axis="x", rotation=15)
annotate_bars(ax, offset=0.5, fontsize=10)
plt.tight_layout()
save_fig(fig, "08_poutcome_vs_conversion.png")


# ---------------------------------------------------------------------------
# 9. HEATMAP: Conversion por Trabajo y Educacion
# ---------------------------------------------------------------------------
# Pivot table con % de conversión para cada combinación (job x education)
pivot = (
    clean.groupby(["job", "education"]) ["y"]
    .apply(lambda s: (s == "yes").mean() * 100)
    .unstack(fill_value=0)
    .round(1)
)

fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax, linewidths=0.5, cbar_kws={"label": "% conversion"})
ax.set_title("Tasa de Conversion (%) por Trabajo y Nivel Educativo", fontsize=13, fontweight="bold")
ax.set_xlabel("Nivel Educativo")
ax.set_ylabel("Ocupacion")
ax.tick_params(axis="x", rotation=20)
ax.tick_params(axis="y", rotation=0)
plt.tight_layout()
save_fig(fig, "09_heatmap_trabajo_educacion.png")


# ---------------------------------------------------------------------------
# 10. CANAL DE CONTACTO vs CONVERSION
# ---------------------------------------------------------------------------
contact_conv = conv_rate(clean, "contact")
contact_vol = clean["contact"].value_counts().reset_index()
contact_vol.columns = ["contact", "cantidad"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Canal de Contacto: Volumen y Conversion", fontsize=13, fontweight="bold")

sns.barplot(data=contact_vol, x="contact", y="cantidad", hue="contact", palette="Blues_d", legend=False, ax=axes[0])
axes[0].set_title("Clientes Contactados por Canal")
axes[0].set_xlabel("Canal")
axes[0].set_ylabel("Cantidad de clientes")
annotate_bars(axes[0], fmt="{:,.0f}", offset=50)

sns.barplot(data=contact_conv, x="contact", y="tasa", hue="contact", palette="Greens_d", legend=False, ax=axes[1])
axes[1].set_title("Tasa de Conversion por Canal (%)")
axes[1].set_xlabel("Canal")
axes[1].set_ylabel("% que suscribio")
annotate_bars(axes[1])

plt.tight_layout()
save_fig(fig, "10_canal_contacto.png")


# ---------------------------------------------------------------------------
# RESUMEN FINAL
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("RESUMEN DEL PERFIL DE CLIENTE CON MAYOR CONVERSION")
print("=" * 60)

# Seleccionamos los mejores/mas representativos por cada dimensión analizada
best_job = job_conv.iloc[0]
best_age = age_rate.idxmax()
best_edu = (clean.groupby("education")["y"].apply(lambda s: (s == "yes").mean() * 100).idxmax())
best_civil = (clean.groupby("marital")["y"].apply(lambda s: (s == "yes").mean() * 100).idxmax())
best_pout = pout_conv.iloc[0]
best_cont = contact_conv.iloc[0]

print(f"  Ocupacion mas propensa   : {best_job['job']} ({best_job['tasa']:.1f}% conversion)")
print(f"  Grupo de edad mas propenso: {best_age} años")
print(f"  Educacion mas propensa   : {best_edu}")
print(f"  Estado civil mas propenso: {best_civil}")
print(f"  Campana anterior exitosa : {best_pout['poutcome']} ({best_pout['tasa']:.1f}% conversion)")
print(f"  Mejor canal de contacto  : {best_cont['contact']} ({best_cont['tasa']:.1f}% conversion)")
print("\nTodos los graficos generados exitosamente.")