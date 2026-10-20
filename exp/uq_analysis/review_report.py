"""Produce reviewable tables and scientific figures from completed v2 trials."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


METRICS = ['accuracy','ece','nll','brier','entropy','expected_entropy','mutual_information','variance']


def entropy(p):
    return -(p*np.log(np.clip(p, 1e-300, 1))).sum(-1)


def generate(output):
    output = Path(output)
    figures = output/'report'
    figures.mkdir(parents=True, exist_ok=True)
    paths = sorted(output.glob('*/*/depth*/split*_seed*/metrics.json'))
    if not paths:
        raise ValueError(f'No completed trials under {output}')
    frame = pd.DataFrame([row for p in paths for row in json.loads(p.read_text())])
    keys = ['dataset','model','layers','split','mode','samples']
    if frame.duplicated(keys+['seed']).any():
        raise ValueError('Duplicate independent trial identifiers')
    frame.to_csv(figures/'trial_metrics.csv', index=False)
    status_rows=[]
    for p in paths:
        rows=json.loads(p.read_text())
        row=rows[0]
        event_path=p.with_name('numerical_guard.json')
        event=json.loads(event_path.read_text()) if event_path.exists() else {}
        status_rows.append({**{k:row[k] for k in ['dataset','model','layers','split','seed']},
                            'termination':row.get('training_termination','normal_unguarded'),
                            'guard_version':row.get('training_guard_version',0),
                            'guard_epoch':event.get('epoch'), 'guard_stage':event.get('stage'),
                            'guard_reason':event.get('reason')})
    pd.DataFrame(status_rows).to_csv(figures/'training_status.csv',index=False)
    summary = frame.groupby(keys)[METRICS].agg(['mean','std','count'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.to_csv(figures/'comparison.csv')
    paired, nested, sheaf, convergence = [], [], [], []
    for path in paths:
        metrics = json.loads(path.read_text())
        for mode in sorted({r['mode'] for r in metrics}):
            with np.load(path.with_name(f'{mode}_predictions.npz')) as saved:
                p = saved['probabilities'].astype(np.float64)
                mean = p.mean(0)
                mi = entropy(mean)-entropy(p).mean(0)
                variance = p.var(0).mean(-1)
                for row in metrics:
                    if row['mode'] != mode:
                        continue
                    q = p[:row['samples']]
                    qmean = q.mean(0)
                    qmi = entropy(qmean)-entropy(q).mean(0)
                    convergence.append({**{k:row[k] for k in keys+['seed']},
                                        'reference_samples':len(p),
                                        'mean_probability_l1':float(np.abs(qmean-mean).sum(-1).mean()),
                                        'mi_mae':float(np.abs(qmi-mi).mean()),
                                        'variance_mae':float(np.abs(q.var(0).mean(-1)-variance).mean())})
    pd.DataFrame(convergence).to_csv(figures/'convergence.csv', index=False)
    # Pair every Bayesian result with the same family, seed, split, depth and budget.
    # Deterministic one-pass accuracy/calibration is compared at all BSNN budgets,
    # with the baseline budget explicitly recorded as one.
    for row in frame.loc[frame['mode'] == 'posterior'].to_dict('records'):
        base = frame.loc[(frame.model == row['model'].replace('Bayes','')) &
                         (frame.dataset == row['dataset']) & (frame.layers == row['layers']) &
                         (frame.split == row['split']) & (frame.seed == row['seed'])]
        for mode in ['deterministic','mc_dropout']:
            match = base.loc[(base['mode'] == mode) & (base.samples == (1 if mode == 'deterministic' else row['samples']))]
            if len(match) != 1:
                continue
            baseline = match.iloc[0]
            paired.append(dict(dataset=row['dataset'], model=row['model'], layers=row['layers'],
                               split=row['split'], seed=row['seed'], samples=row['samples'], baseline=mode,
                               baseline_samples=int(baseline.samples),
                               **{f'delta_{m}': row[m]-baseline[m] for m in METRICS}))
    if paired:
        pairs = pd.DataFrame(paired)
        pairs.to_csv(figures/'paired_differences.csv', index=False)
        group = ['dataset','model','layers','split','samples','baseline','baseline_samples']
        delta_columns = [f'delta_{m}' for m in METRICS]
        aggregate = pairs.groupby(group)[delta_columns].agg(['mean','std','count'])
        aggregate.columns = ['_'.join(c) for c in aggregate.columns]
        aggregate.to_csv(figures/'paired_summary.csv')

    # Between-run agreement is reported separately from within-run posterior MI.
    group_keys = ['dataset','model','layers','split','mode']
    for values, group in frame.groupby(group_keys):
        maximum = group.samples.max()
        trials = group.loc[group.samples == maximum]
        means, expected, within, references = [], [], [], []
        for row in trials.to_dict('records'):
            path = output/row['dataset']/row['model']/f"depth{row['layers']}"/f"split{row['split']}_seed{row['seed']}"
            with np.load(path/f"{row['mode']}_predictions.npz") as data:
                probabilities = data['probabilities'].astype(np.float64)
                reference = (data['node_idx'], data['true'])
                if references:
                    if not all(np.array_equal(a,b) for a,b in zip(reference,references[0])):
                        raise ValueError('Node/label alignment mismatch between trials')
                references.append(reference)
                mean = probabilities.mean(0)
                cond = entropy(probabilities).mean(0)
                means.append(mean)
                expected.append(cond)
                within.append(entropy(mean)-cond)
        means, expected, within = np.stack(means), np.stack(expected), np.stack(within)
        total = entropy(means.mean(0))
        between = total-entropy(means).mean(0)
        residual = total-expected.mean(0)-within.mean(0)-between
        nested.append(dict(zip(group_keys, values), trials=len(trials), samples=int(maximum),
                           pooled_predictive_entropy=float(total.mean()),
                           expected_conditional_entropy=float(expected.mean()),
                           within_trial_mi=float(within.mean()),
                           between_run_disagreement=float(between.mean()) if len(trials)>1 else np.nan,
                           max_identity_residual=float(np.abs(residual).max())))
    pd.DataFrame(nested).to_csv(figures/'nested_uncertainty.csv', index=False)

    for path in paths:
        layer_path = path.with_name('sheaf_layers.csv')
        if layer_path.exists():
            cfg = json.loads(path.read_text())[0]
            layers = pd.read_csv(layer_path)
            for k in ['model','dataset','seed','split','layers']:
                layers[k] = cfg[k]
            sheaf.append(layers)
    if sheaf:
        pd.concat(sheaf).to_csv(figures/'sheaf_output_correlations.csv', index=False)

    manifest = json.loads((output/'manifest.json').read_text())
    expected = len(manifest['configurations'])
    coverage = dict(expected_training_trials=expected, completed_training_trials=len(paths),
                    complete=len(paths)==expected, planned_samples=manifest['sample_counts'],
                    numerically_stopped_trials=sum(r['termination']=='numerical_guard' for r in status_rows),
                    note='Counts are training trials, not forward passes; no independent hyperparameter tuning.')
    (figures/'coverage.json').write_text(json.dumps(coverage, indent=2)+'\n')
    # A portable table for the standalone revision supplement. Keep depth and
    # partition identifiers visible; do not fold them into seed error bars.
    largest=frame.loc[(frame.samples==frame.groupby(group_keys).samples.transform('max'))]
    table=largest.groupby(['dataset','model','mode','layers','split'])[['accuracy','ece','nll','brier']].agg(['mean','std'])
    tex=[r'\begin{longtable}{lllrrrrrr}',r'Dataset & Model & Mode & Depth & Split & Accuracy & ECE & NLL & Brier \\',r'\hline']
    for key,values in table.iterrows():
        cells=[str(x).replace('_',r'\_') for x in key]
        for metric in ['accuracy','ece','nll','brier']:
            mean,std=values[(metric,'mean')],values[(metric,'std')]
            cells.append(f'${mean:.3f} \\pm {std:.3f}$' if np.isfinite(std) else f'${mean:.3f}$')
        tex.append(' & '.join(cells)+r' \\')
    tex.append(r'\end{longtable}')
    (figures/'comparison.tex').write_text('\n'.join(tex)+'\n')

    # Each page compares one dataset/depth/split without pooling unequal partitions.
    with PdfPages(figures/'calibration_and_convergence.pdf') as pdf:
        for (dataset, depth, split), group in frame.groupby(['dataset','layers','split']):
            fig, axes = plt.subplots(2,3,figsize=(13,7), constrained_layout=True)
            for (model, mode), trials in group.groupby(['model','mode']):
                label=f'{model}/{mode}'
                for ax, metric in zip(axes[0], ['accuracy','ece','nll']):
                    values = trials.groupby('samples')[metric].agg(['mean','std'])
                    ax.errorbar(values.index, values['mean'], yerr=values['std'].fillna(0), label=label,
                                marker='.', capsize=2)
                    ax.set(xlabel='Inference draws', ylabel=metric, xscale='log')
                for ax, metric in zip(axes[1,:2], ['mutual_information','expected_entropy']):
                    values=trials.groupby('samples')[metric].mean()
                    ax.plot(values.index,values.values,marker='.',label=label)
                    ax.set(xlabel='Inference draws', ylabel=metric+' (nats)', xscale='log')
                maximum=trials.samples.max()
                selected=trials.loc[trials.samples==maximum]
                bins=[]
                for row in selected.to_dict('records'):
                    p=output/dataset/model/f'depth{depth}'/f"split{split}_seed{row['seed']}"/f'{mode}_calibration.csv'
                    bins.append(pd.read_csv(p))
                b=pd.concat(bins)
                b=b.loc[b['count']>0].copy()
                b['weighted_confidence']=b.confidence*b['count']
                b['weighted_accuracy']=b.accuracy*b['count']
                sums=b.groupby('bin')[['count','weighted_confidence','weighted_accuracy']].sum()
                axes[1,2].plot(sums.weighted_confidence/sums['count'],sums.weighted_accuracy/sums['count'],'.-',label=label)
            axes[1,2].plot([0,1],[0,1],'--',color='gray')
            axes[1,2].set(xlabel='Mean confidence',ylabel='Accuracy',xlim=(0,1),ylim=(0,1))
            axes[0,0].legend(fontsize=5)
            fig.suptitle(f'{dataset}; depth={depth}; split={split}; bars: seed SD; reliability: pooled node/trials')
            pdf.savefig(fig)
            plt.close(fig)

    with PdfPages(figures/'predictive_distributions.pdf') as pdf:
        # Select by fixed rules: high/low MI within correct/incorrect strata.
        # One seed per setting keeps the artifact readable; all probabilities are saved.
        examples=frame.loc[(frame.samples==frame.groupby(group_keys).samples.transform('max'))]
        examples=examples.sort_values('seed').drop_duplicates(group_keys)
        selection=[]
        for row in examples.to_dict('records'):
            path=output/row['dataset']/row['model']/f"depth{row['layers']}"/f"split{row['split']}_seed{row['seed']}"
            data=np.load(path/f"{row['mode']}_predictions.npz")
            p=data['probabilities'].astype(np.float64)
            mean=p.mean(0)
            mi=entropy(mean)-entropy(p).mean(0)
            correct=mean.argmax(-1)==data['true']
            fig, axes=plt.subplots(1,4,figsize=(13,3),constrained_layout=True)
            for ax,(is_correct,high) in zip(axes,[(True,False),(True,True),(False,False),(False,True)]):
                candidates=np.flatnonzero(correct==is_correct)
                title=('Correct' if is_correct else 'Incorrect')+('; high MI' if high else '; low MI')
                if not len(candidates):
                    ax.set_title(title+'\nNo nodes')
                    continue
                index=candidates[np.argmax(mi[candidates]) if high else np.argmin(mi[candidates])]
                x=np.arange(mean.shape[-1])
                ax.bar(x,mean[index],yerr=p[:,index,:].std(0),capsize=2)
                # Individual draws for selected nodes reveal multimodality hidden by bars.
                ax.plot(np.tile(x,(len(p),1)).T,p[:,index,:].T,'.',color='black',alpha=.12,markersize=2)
                ax.set(ylim=(0,1.05),xlabel='Class',ylabel='Probability',
                       title=title+f"\nnode {data['node_idx'][index]}, true={data['true'][index]}, MI={mi[index]:.3g}")
                selection.append({**{k:row[k] for k in group_keys+['seed']}, 'node_idx':int(data['node_idx'][index]),'rule':title})
            fig.suptitle(f"{row['dataset']} / {row['model']} / {row['mode']}, depth {row['layers']}, seed {row['seed']}")
            pdf.savefig(fig)
            plt.close(fig)
            data.close()
        pd.DataFrame(selection).to_csv(figures/'example_selection.csv', index=False)

    with PdfPages(figures/'depth_and_sheaf_uncertainty.pdf') as pdf:
        for (dataset,split),group in frame.groupby(['dataset','split']):
            fig,axes=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
            max_group=group.loc[group.samples==group.groupby(['model','mode']).samples.transform('max')]
            for (model,mode),trials in max_group.groupby(['model','mode']):
                for ax,metric in zip(axes,['variance','mutual_information','accuracy']):
                    stats=trials.groupby('layers')[metric].agg(['mean','std'])
                    ax.errorbar(stats.index,stats['mean'],yerr=stats['std'].fillna(0),marker='.',label=f'{model}/{mode}')
                    ax.set(xlabel='Layers',ylabel=metric)
            axes[0].legend(fontsize=5)
            fig.suptitle(f'{dataset}, split {split}; fixed settings except depth; seed SD')
            pdf.savefig(fig); plt.close(fig)
        for path in paths:
            if not path.with_name('sheaf_nodes.csv').exists():
                continue
            metadata=json.loads(path.read_text())[0]
            if metadata['seed']!=min(frame.seed):
                continue
            nodes=pd.read_csv(path.with_name('sheaf_nodes.csv'))
            fig,axes=plt.subplots(1,2,figsize=(9,3.5),constrained_layout=True)
            for layer,group in nodes.groupby('layer'):
                axes[0].scatter(group.sheaf_spread,group.mutual_information,s=5,alpha=.5,label=f'Layer {layer}')
                axes[1].scatter(group.sheaf_spread,group.output_variance,s=5,alpha=.5)
            axes[0].set(xlabel='Incident-map spread',ylabel='Output MI (nats)')
            axes[1].set(xlabel='Incident-map spread',ylabel='Output variance')
            axes[0].legend(fontsize=6)
            fig.suptitle(f"{metadata['dataset']}, {metadata['model']}, depth {metadata['layers']}, seed {metadata['seed']}")
            pdf.savefig(fig); plt.close(fig)

    note = f"""# Corrected matched study

Completed {len(paths)} of {expected} planned training trials. Data and checkpoints use corrected Gaussian log-variance sampling. This study uses fixed matched configurations, newly constructed partitions, and validation-selected epochs; it does not establish independently tuned state-of-the-art performance.

- `comparison.csv`: mean, sample SD and trial count, keeping datasets, splits, depths, budgets and inference modes separate.
- `training_status.csv`: numerical termination flags and guard events. Guarded trials use the best finite validation checkpoint from before the failure, not substituted NaNs; they must be disclosed as numerically truncated training. Previously completed trials are unchanged.
- `paired_differences.csv` and `paired_summary.csv`: BSNN minus matched NSD or MC-dropout NSD. Positive accuracy is favorable; negative ECE/NLL/Brier is favorable. SD across training seeds on a fixed split is not uncertainty across partitions.
- `nested_uncertainty.csv`: expected conditional entropy, within-trial MI, and separately labelled between-run disagreement. Between-run disagreement is a diagnostic across independently fitted models, not posterior sampling from one model.
- `convergence.csv`: per-node mean-probability L1, MI absolute error and variance absolute error against the largest sampled prefix; the reference is a finite Monte Carlo estimate, not ground truth.
- `sheaf_output_correlations.csv`: descriptive within-trial node correlations; no node-independence significance claim. Map spread uses squared Frobenius/chordal distance in learned coordinates, before edge weighting and Laplacian normalization.
- `calibration_and_convergence.pdf`: matched accuracy/calibration, expected entropy and MI versus inference budget, plus reliability diagrams.
- `predictive_distributions.pdf`: class probabilities, sample spread and individual draws for examples selected by fixed correctness/MI rules.
- `depth_and_sheaf_uncertainty.pdf`: controlled depth comparisons and incident-map spread versus output uncertainty.

One-pass references cannot estimate epistemic variation. Larger-sample estimates provide convergence diagnostics rather than ground truth. Network weights remain point estimates. Expected conditional entropy is model conditional ambiguity, not separately measured physical label noise. Equal inference draws do not guarantee equal training cost or model capacity; per-trial costs and parameter counts are saved. Non-Bayesian dropout is active only in functional dropout sites; module evaluation mode remains enabled. Deep ensembles are not part of this fixed-training-cost comparison.
"""
    (figures/'README.md').write_text(note)
    print('Wrote review report:',figures,flush=True)
