import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib import ticker
from mpl_scatter_density import ScatterDensityArtist

# ----------------------- Utilities -----------------------

def pretty_model_name(model_name):
    return {
        "BayesDiagSheaf": "Diag-BSNN",
        "BayesBundleSheaf": "SO(d)-BSNN",
        "BayesGeneralSheaf": "Gen-BSNN"
    }.get(model_name, model_name)

def load_prediction_level_data(base_path="uq_plot"):
    custom_dataset_order = ["Texas", "Wisconsin", "Film", "Cornell", "Citeseer", "Pubmed", "Cora"]
    all_data = {}

    models = sorted(os.listdir(base_path))
    for model in models:
        model_path = os.path.join(base_path, model)
        if not os.path.isdir(model_path):
            continue
        for dataset in custom_dataset_order:
            dataset_path = os.path.join(model_path, dataset.lower(), "uq_results.csv")
            if os.path.exists(dataset_path):
                df = pd.read_csv(dataset_path)
                all_data[(dataset, model)] = df
    return all_data

# ------------------- Plot: Entropy vs Epistemic Variance -------------------

def plot_entropy_variance_scatter_density(all_data_dict, save_path="plots/ev_density_grid.png", dpi=600):
    sns.set_theme(style="white", font_scale=0.8)
    plt.rcParams.update({
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'legend.frameon': False
    })

    datasets = sorted(set(k[0] for k in all_data_dict))
    models = sorted(set(k[1] for k in all_data_dict))
    rows, cols = len(datasets), len(models)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.5 * rows), subplot_kw={'projection': 'scatter_density'})
    fig.subplots_adjust(wspace=0.2, hspace=0.3)

    all_variances = [v for (_, _), df in all_data_dict.items() for v in df["variance"]]
    all_entropies = [e for (_, _), df in all_data_dict.items() for e in df["entropy"]]
    global_x_lim = (min(all_variances), max(all_variances))
    global_y_lim = (min(all_entropies), max(all_entropies))

    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            ax = axes[i, j]
            df = all_data_dict.get((dataset, model))
            if df is None:
                ax.axis("off")
                continue
            density_artist = ax.scatter_density(df["variance"], df["entropy"], cmap="viridis", norm=LogNorm(vmin=1, vmax=1e3))
            ax.set_xlim(*global_x_lim)
            ax.set_ylim(*global_y_lim)
            if i == rows - 1:
                ax.set_xlabel("Epistemic Variance")
            if j == 0:
                ax.set_ylabel("Entropy")
            if i == 0:
                ax.set_title(pretty_model_name(model), fontsize=11)

    cbar = fig.colorbar(density_artist, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.02, pad=0.03)
    cbar.set_label("Density (log scale)", fontsize=10)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()

# ------------------- Plot: MI Histogram -------------------

def plot_mi_histogram_grid(all_data_dict, save_path="plots/mi_histogram_grid.png", dpi=600):
    sns.set_theme(style="white", font_scale=0.8)
    datasets = sorted(set(k[0] for k in all_data_dict))
    models = sorted(set(k[1] for k in all_data_dict))
    rows, cols = len(datasets), len(models)

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.5 * rows))
    fig.subplots_adjust(wspace=0.2, hspace=0.3)

    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            ax = axes[i, j]
            df = all_data_dict.get((dataset, model))
            if df is None or df["mutual_information"].empty:
                ax.axis("off")
                continue
            counts, bins = np.histogram(df["mutual_information"], bins=20)
            rel_freq = counts / counts.sum()
            ax.bar(bins[:-1], rel_freq, width=(bins[1] - bins[0]), color='teal', edgecolor='teal', linewidth=0.8)
            if i == rows - 1:
                ax.set_xlabel("Mutual Information")
            if j == 0:
                ax.set_ylabel("Proportion")
            if i == 0:
                ax.set_title(pretty_model_name(model), fontsize=11)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, min_n_ticks=2))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()

# ------------------- Table: Averages -------------------

def create_uq_table(base_path="uq_plot"):
    all_data = load_prediction_level_data(base_path)
    model_data = {}

    for (dataset, model), df in all_data.items():
        pretty_name = pretty_model_name(model)
        model_data.setdefault(pretty_name, []).append(df)

    rows = []
    for model, dfs in model_data.items():
        entropies = [df["entropy"].mean() for df in dfs]
        variances = [df["variance"].mean() for df in dfs]
        rows.append({
            "Model": model,
            "Average Entropy": round(sum(entropies) / len(entropies), 6),
            "Average Epistemic Variance": round(sum(variances) / len(variances), 6),
        })

    order = ["Diag-BSNN", "Gen-BSNN", "SO(d)-BSNN"]
    df_summary = pd.DataFrame(rows).set_index("Model").loc[order].reset_index()
    return df_summary

# ------------------- Main -------------------

if __name__ == "__main__":
    base_path = "uq_plot"
    all_data_dict = load_prediction_level_data(base_path)

    plot_entropy_variance_scatter_density(all_data_dict, save_path="plots/ev_density_grid.png")
    plot_mi_histogram_grid(all_data_dict, save_path="plots/mi_histogram_grid.png")

    summary_table = create_uq_table(base_path)
    print(summary_table.to_string(index=False))
