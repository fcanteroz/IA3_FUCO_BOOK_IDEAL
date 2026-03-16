import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def visualizacion_uni_hist_boxplot(df: pd.DataFrame, col: str, bins="auto") -> None:
    """
    Muestra métricas tipo df.describe() (en texto) + histograma y boxplot
    para una columna numérica.

    Params
    ------
    df : DataFrame
    col : str
        Nombre de la columna a analizar
    bins : int | str
        Bins para el histograma (por defecto 'auto')
    """
    if col not in df.columns:
        raise ValueError(f"La columna '{col}' no existe en el DataFrame.")

    # Acepta int/float; si viene como string numérico, intenta convertir
    s = df[col]
    if not pd.api.types.is_numeric_dtype(s):
        s = pd.to_numeric(s, errors="coerce")

    x = s.dropna()
    if x.empty:
        raise ValueError(f"La columna '{col}' no tiene valores numéricos válidos (todo NaN o no numérico).")

    # 1) Métricas (describe) en texto
    desc = x.describe()
    print(f"=== ESTADISTICA DESCRIPTIVA '{col}' ===")
    print(desc.to_string())

    # 2) Visualización: histograma + boxplot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [3, 1]})

    axes[0].hist(x, bins=bins, edgecolor="black", linewidth=1.0)
    axes[0].set_title(f"Histograma: {col}")
    axes[0].set_xlabel(col)
    axes[0].set_ylabel("Frecuencia")

    axes[1].boxplot(x, vert=True)
    axes[1].set_title(f"Boxplot: {col}")
    axes[1].set_xticks([])

    plt.tight_layout()
    plt.show()
    
def plot_corr_matrix(df: pd.DataFrame, method: str = "pearson", annot_fmt: str = ".2f") -> pd.DataFrame:
    """
    Calcula y muestra la matriz de correlación para columnas numéricas.
    - Colormap: tonos pastel azul
    - Anota el valor de correlación en cada celda

    Devuelve el DataFrame de correlaciones.
    """
    corr = df.corr(numeric_only=True, method=method)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="Blues")  # azul tipo pastel

    # Ticks con nombres
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"Correlación ({method})")

    # Anotaciones por celda
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            val = corr.iloc[i, j]
            ax.text(j, i, format(val, annot_fmt), ha="center", va="center", fontsize=9)

    ax.set_title(f"Matriz de correlación ({method})")
    plt.tight_layout()
    plt.show()


def scatter_feature_vs_target(df: pd.DataFrame, feature: str, target: str) -> None:
    """
    Scatterplot de una sola variable (feature) vs la variable objetivo (target).
    """
    if feature not in df.columns:
        raise ValueError(f"La feature '{feature}' no existe en el DataFrame.")
    if target not in df.columns:
        raise ValueError(f"La variable objetivo '{target}' no existe en el DataFrame.")

    # Intentar asegurar que sean numéricas (sin modificar el df original)
    x = pd.to_numeric(df[feature], errors="coerce")
    y = pd.to_numeric(df[target], errors="coerce")

    tmp = pd.DataFrame({feature: x, target: y}).dropna()
    if tmp.empty:
        raise ValueError("No hay datos numéricos válidos para graficar (todo NaN tras conversión).")

    plt.figure(figsize=(7, 5))
    plt.scatter(tmp[feature], tmp[target], alpha=0.6)
    plt.title(f"{feature} vs {target}")
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.tight_layout()
    plt.show()
