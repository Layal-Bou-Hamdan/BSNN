"""Generate Figures 3–4 and Tables 2–3 from per-trial UQ outputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS = ['BayesDiagSheaf', 'BayesGeneralSheaf', 'BayesBundleSheaf']
LABELS = ['Diag-BSNN', 'Gen-BSNN', 'SO(d)-BSNN']
DATASETS = ['texas', 'wisconsin', 'film', 'cornell', 'citeseer', 'pubmed', 'cora']
ROOT = Path(__file__).resolve().parents[2]


def generate(base, output):
    data, summaries = {}, []
    for model in MODELS:
        for dataset in DATASETS:
            trials = []
            for path in sorted((base/model/dataset).glob('seed*/uq_results.csv')):
                frame = pd.read_csv(path)
                if frame.empty:
                    raise ValueError(f'Empty trial: {path}')
                summary = json.loads(path.with_name('summary.json').read_text())
                summaries.append(dict(model=model, dataset=dataset,
                                      entropy=frame.entropy.mean(), variance=frame.variance.mean(),
                                      mutual_information=frame.mutual_information.mean(), **summary))
                trials.append(frame)
            if trials:
                data[dataset, model] = pd.concat(trials, ignore_index=True)
    if not data:
        raise ValueError(f'No per-trial UQ results found in {base}')
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output/'trial_metrics.csv', index=False)
    # First average trials within datasets, then give each dataset equal weight.
    ds = summary.groupby(['model','dataset'])[['entropy','variance','ece']].mean()
    ds.groupby('model')[['entropy','variance']].mean().to_csv(output/'table2.csv')
    summary.groupby(['model','dataset']).ece.agg(['mean','std','count']).to_csv(output/'table3.csv')
    missing = [f'{m}/{d}' for m in MODELS for d in DATASETS if (d,m) not in data]
    (output/'coverage.json').write_text(json.dumps(dict(missing=missing, trials=len(summary)), indent=2))
    if missing:
        print('Incomplete paper coverage:', ', '.join(missing))
    for kind in ['entropy_variance', 'mutual_information']:
        fig, axes = plt.subplots(len(DATASETS), len(MODELS), squeeze=False,
                                 figsize=(10, 15), constrained_layout=True)
        # Shared scales make comparisons meaningful; histogram heights are proportions.
        xmax = max(float(df.variance.max()) for df in data.values()) or 1e-6
        ymax = max(float(df.entropy.max()) for df in data.values()) or 1e-6
        mi_max = max(float(df.mutual_information.max()) for df in data.values()) or 1e-6
        bins = np.linspace(0, mi_max*1.01, 21)
        density_max = max(len(df) for df in data.values())
        artist = None
        for i, dataset in enumerate(DATASETS):
            for j, model in enumerate(MODELS):
                ax = axes[i,j]
                if i == 0:
                    ax.set_title(LABELS[j])
                if j == 0:
                    ax.set_ylabel(dataset.title()+'\n'+('Predictive entropy (nats)' if kind == 'entropy_variance' else 'Proportion'))
                frame = data.get((dataset,model))
                if frame is None:
                    ax.text(.5,.5,'Missing results',ha='center',transform=ax.transAxes)
                    continue
                if kind == 'entropy_variance':
                    artist = ax.hexbin(frame.variance, frame.entropy, gridsize=35, mincnt=1,
                              bins='log', extent=(0,xmax*1.01,0,ymax*1.01), cmap='viridis', vmin=1, vmax=max(2, density_max))
                    ax.set_xlim(0,xmax*1.01); ax.set_ylim(0,ymax*1.01)
                    label = 'Epistemic variance'
                else:
                    ax.hist(frame.mutual_information, bins=bins,
                            weights=np.ones(len(frame))/len(frame), color='teal')
                    ax.set_xlim(0,bins[-1]); ax.set_ylim(0,1)
                    label = 'Mutual information (nats)'
                if i == len(DATASETS)-1:
                    ax.set_xlabel(label)
                ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        if kind == 'entropy_variance' and artist is not None:
            fig.colorbar(artist, ax=axes.ravel().tolist(), label='Node-trial count (log scale)', shrink=.5)
        for extension in ['pdf','png']:
            fig.savefig(output/f'{kind}.{extension}', dpi=300)
        plt.close(fig)
    print(f'Wrote plots and tables to {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'uq')
    parser.add_argument('--output', type=Path, default=ROOT/'plots')
    args = parser.parse_args()
    generate(args.input, args.output)


if __name__ == '__main__':
    main()
