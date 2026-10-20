"""Matched baseline, convergence, nested uncertainty, and sheaf diagnostics.

Each checkpoint is an independent trial. Posterior/dropout draws never become
independent training trials. All data are kept under a separate versioned root.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from exp.inference import inference_mode, log_probabilities
from exp.uq_analysis.compute_uq_metrics import calibration_bins, ece_score, predictive_statistics
from models.registry import MODELS
from utils.heterophilic import get_dataset, get_fixed_splits

VERSION = 'review-v2-log-variance'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def entropy(p):
    return -(p * np.log(np.clip(p, np.finfo(float).tiny, 1))).sum(-1)


def scores(samples, true):
    p = np.asarray(samples, dtype=np.float64)
    # A one-pass reference has no estimated posterior variation.
    stats = predictive_statistics(p if len(p) > 1 else np.repeat(p, 2, axis=0))
    mean = p.mean(0)
    return dict(accuracy=float((stats['pred'] == true).mean()),
                ece=ece_score(stats['confidence'], stats['pred'], true),
                nll=float(-np.log(np.clip(mean[np.arange(len(true)), true], 1e-300, 1)).mean()),
                brier=float(np.square(mean - np.eye(mean.shape[-1])[true]).sum(-1).mean()),
                entropy=float(stats['entropy'].mean()),
                expected_entropy=float(stats['expected_entropy'].mean()),
                mutual_information=float(stats['mutual_information'].mean()),
                variance=float(stats['variance'].mean()))


def rank_correlation(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(spearmanr(a, b).statistic)


class MapMoments:
    """Online per-layer map moments; bounded memory independent of draws."""
    def __init__(self):
        self.mean, self.m2, self.count = {}, {}, {}
        self.examples = {}

    def add(self, layer, maps):
        x = maps.detach().cpu().numpy().astype(np.float64)
        n = self.count.get(layer, 0) + 1
        if n == 1:
            self.mean[layer], self.m2[layer] = x.copy(), np.zeros_like(x)
            self.examples[layer] = []
        else:
            delta = x - self.mean[layer]
            self.mean[layer] += delta / n
            self.m2[layer] += delta * (x - self.mean[layer])
        self.count[layer] = n
        self.examples[layer].append(x[:min(16, len(x))].astype(np.float32))

    def save(self, path, source_nodes, test_nodes, stats, graph_size, rotation=False):
        rows, nodes, arrays = [], [], {}
        for layer in sorted(self.mean):
            # Squared chordal/Frobenius spread about the Euclidean mean.
            # For rotations this avoids treating latent angle coordinates as Euclidean.
            variance = self.m2[layer] / self.count[layer]
            spread = variance.reshape(len(source_nodes), -1).sum(-1)
            totals = np.bincount(source_nodes, weights=spread, minlength=graph_size)
            degrees = np.bincount(source_nodes, minlength=graph_size)
            node_spread = totals / np.maximum(degrees, 1)
            selected = node_spread[test_nodes]
            rows.append(dict(layer=layer, draws=self.count[layer],
                             mean_map_spread=float(spread.mean()),
                             map_metric='squared_chordal' if rotation else 'squared_frobenius',
                             spearman_mi=rank_correlation(selected, stats['mutual_information']),
                             spearman_output_variance=rank_correlation(selected, stats['variance'])))
            nodes.append(pd.DataFrame(dict(node_idx=test_nodes, layer=layer,
                                           sheaf_spread=selected,
                                           mutual_information=stats['mutual_information'],
                                           output_variance=stats['variance'])))
            arrays[f'layer{layer}_mean'] = self.mean[layer].astype(np.float32)
            arrays[f'layer{layer}_variance'] = variance.astype(np.float32)
            arrays[f'layer{layer}_sampled_first16'] = np.stack(self.examples[layer])
        if rows:
            pd.DataFrame(rows).to_csv(path/'sheaf_layers.csv', index=False)
            pd.concat(nodes).to_csv(path/'sheaf_nodes.csv', index=False)
            np.savez_compressed(path/'sheaf_maps.npz', source_nodes=source_nodes, **arrays)


def load_data(cfg, device):
    dataset = get_dataset(cfg['dataset'])
    split_dir = ROOT/cfg['split_dir']
    data = get_fixed_splits(dataset[0], cfg['dataset'], cfg['split_index'],
                            cfg['permute_masks'], str(split_dir)).to(device)
    settings = {**cfg, 'device': device, 'graph_size': data.num_nodes,
                'input_dim': dataset.num_features, 'output_dim': dataset.num_classes}
    return data, settings


def validation(model, data, draws, dropout):
    with inference_mode(model, dropout):
        probs = torch.stack([log_probabilities(model, data.x).exp() for _ in range(draws)]).mean(0)
    ids = data.val_mask
    return (float((probs[ids].argmax(-1) == data.y[ids]).float().mean()),
            float(F.nll_loss(probs[ids].clamp_min(1e-30).log(), data.y[ids])))


def require_finite_parameters(model, gradients=False):
    named = [(name, p.grad if gradients else p) for name, p in model.named_parameters()
             if not gradients or p.grad is not None]
    flags = torch.stack([torch.isfinite(value).all() for _, value in named])
    if not bool(flags.all()):
        bad = [name for (name, _), finite in zip(named, flags.tolist()) if not finite]
        kind = 'gradients' if gradients else 'parameters'
        raise FloatingPointError(f'Nonfinite {kind}: {bad}')


def train_trial(cfg, path, device):
    """No test access for checkpoint, candidate, stopping, or model selection."""
    seed_all(cfg['seed'])
    data, settings = load_data(cfg, device)
    model = MODELS[cfg['model']](data.edge_index, settings).to(device)
    sheaf, other = model.grouped_parameters()
    optimizer = torch.optim.Adam([{'params': sheaf, 'weight_decay': cfg['sheaf_decay']},
                                  {'params': other, 'weight_decay': cfg['weight_decay']}], lr=cfg['lr'])
    best_key, best, bad, history = None, None, 0, []
    start = time.perf_counter()
    bayes = cfg['model'].startswith('Bayes')
    guard_event = None
    for epoch in range(cfg['epochs']):
        stage = 'training_forward'
        try:
            train_metrics = guarded_step(model, optimizer, data, cfg, epoch, bayes)
            nll, kl, beta = train_metrics
            stage = 'validation'
            # Restore every RNG state even if validation fails.
            np_state, py_state = np.random.get_state(), random.getstate()
            devices = [device.index or 0] if device.type == 'cuda' else []
            try:
                with torch.random.fork_rng(devices=devices):
                    seed_all(1000000 + cfg['seed'] * 10000 + epoch)
                    val_acc, val_nll = validation(model, data, cfg['num_ensemble'] if bayes else 1, False)
            finally:
                np.random.set_state(np_state)
                random.setstate(py_state)
            if not np.isfinite([val_acc, val_nll]).all():
                raise FloatingPointError('Nonfinite validation metrics')
        except (FloatingPointError, AssertionError) as error:
            # Handle only numerical assertions; shape/programming assertions stay fatal.
            if isinstance(error, AssertionError) and not any(
                    'isfinite' in (entry.line or '') for entry in traceback.extract_tb(error.__traceback__)):
                raise
            guard_event = dict(epoch=epoch, stage=getattr(error, 'stage', stage),
                               reason=str(error) or 'Nonfinite intermediate tensor',
                               best_epoch=best_epoch if best is not None else None)
            path.mkdir(parents=True, exist_ok=True)
            (path/'numerical_guard.json').write_text(json.dumps(guard_event, indent=2)+'\n')
            print('Numerical guard:', path, guard_event, flush=True)
            break
        key = (val_acc, -val_nll)
        history.append(dict(epoch=epoch, train_nll=float(nll.detach()), kl=float(kl.detach()),
                            beta=float(beta), val_accuracy=val_acc, val_nll=val_nll))
        if best_key is None or key > best_key:
            best_key, bad = key, 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        else:
            bad += 1
        if bad >= cfg['early_stopping']:
            break
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(path/'training.csv', index=False)
    if best is None:
        raise FloatingPointError(f'No finite validation checkpoint for {path}: {guard_event}')
    checkpoint = dict(format_version=2, study_version=VERSION,
                      training_guard_version=1,
                      termination='numerical_guard' if guard_event else 'normal', guard_event=guard_event,
                      gaussian_scale_convention='log_variance', config=cfg,
                      state_dict=best, split=cfg['split_index'], best_epoch=best_epoch,
                      best_val_acc=best_key[0], best_val_nll=-best_key[1],
                      training_seconds=time.perf_counter()-start,
                      parameter_count=sum(p.numel() for p in model.parameters()),
                      split_sha256=digest(ROOT/cfg['split_dir']/f"{cfg['dataset']}_split_0.6_0.2_{cfg['split_index']}.npz"))
    torch.save(checkpoint, path/'checkpoint.pt')
    return checkpoint


def guarded_step(model, optimizer, data, cfg, epoch, bayes):
    """Reject bad gradients before Adam can corrupt parameters or its moments.

    No clipping, learning-rate changes, jitter changes, or NaN substitution.
    A numerical error ends that trial at its best finite validation checkpoint.
    """
    stage = 'training_forward'
    try:
        model.train()
        optimizer.zero_grad()
        result = model(data.x)
        logp, kl = result if bayes else (result, data.x.new_zeros(()))
        nll = F.nll_loss(logp[data.train_mask], data.y[data.train_mask])
        beta = torch.sigmoid(data.x.new_tensor((epoch % 40) / 2 - 10))
        loss = nll + beta * kl if bayes and cfg['use_kl'] else nll
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        stage = 'backward'
        loss.backward()
        stage = 'gradient_check_before_adam'
        require_finite_parameters(model, gradients=True)
        stage = 'adam_update'
        optimizer.step()
        require_finite_parameters(model)
        return nll, kl, beta
    except (FloatingPointError, AssertionError) as error:
        error.stage = stage
        raise


def evaluate(checkpoint, path, counts, device, capture=True):
    cfg = checkpoint['config']
    data, settings = load_data(cfg, device)
    expected_hash = digest(ROOT/cfg['split_dir']/f"{cfg['dataset']}_split_0.6_0.2_{cfg['split_index']}.npz")
    if expected_hash != checkpoint['split_sha256']:
        raise ValueError('Split file differs from training checkpoint')
    model = MODELS[cfg['model']](data.edge_index, settings).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    bayes = cfg['model'].startswith('Bayes')
    ids = data.test_mask.nonzero(as_tuple=True)[0]
    node_ids, true = ids.cpu().numpy(), data.y[ids].cpu().numpy()
    outputs = []
    modes = ['posterior'] if bayes else ['deterministic', 'mc_dropout']
    for mode in modes:
        seed_all(2000000 + cfg['seed'])
        samples, moments, layer_counter = [], MapMoments(), [0]
        builder = model.laplacian_builder
        handle = None
        if bayes and capture:
            builder.capture_samples = True
            def hook(module, inputs, result):
                layer = layer_counter[0]
                moments.add(layer, module.sampled_maps)
                # First draw parameters are sufficient: BSNN parameters are
                # deterministic at eval; layer samples share that distribution.
                if layer_counter[0] == 0 and len(samples) == 0:
                    params = {f'parameter{i}': x.detach().cpu().numpy() for i, x in enumerate(inputs)
                              if torch.is_tensor(x)}
                    np.savez_compressed(path/'sheaf_parameters.npz', **params)
                layer_counter[0] += 1
            handle = builder.register_forward_hook(hook)
        start = time.perf_counter()
        try:
            with inference_mode(model, mode == 'mc_dropout'):
                for _ in range(1 if mode == 'deterministic' else max(counts)):
                    layer_counter[0] = 0
                    samples.append(log_probabilities(model, data.x)[ids].exp().cpu().numpy())
        finally:
            if handle:
                handle.remove()
                builder.capture_samples = False
                del builder.sampled_maps
        seconds = time.perf_counter() - start
        samples = np.stack(samples)
        np.savez_compressed(path/f'{mode}_predictions.npz', probabilities=samples,
                            node_idx=node_ids, true=true, edge_index=data.edge_index.cpu().numpy())
        for count in ([1] if mode == 'deterministic' else counts):
            metrics = scores(samples[:count], true)
            outputs.append(dict(model=cfg['model'], dataset=cfg['dataset'], seed=cfg['seed'],
                                split=cfg['split_index'], layers=cfg['layers'], mode=mode,
                                samples=count, inference_seconds=seconds*count/len(samples),
                                training_seconds=checkpoint['training_seconds'],
                                parameter_count=checkpoint['parameter_count'], **metrics))
            outputs[-1]['training_termination'] = checkpoint.get('termination', 'normal_unguarded')
            outputs[-1]['training_guard_version'] = checkpoint.get('training_guard_version', 0)
        stats = predictive_statistics(samples if len(samples)>1 else np.repeat(samples, 2, 0))
        pd.DataFrame(dict(node_idx=node_ids, true=true, **stats)).to_csv(path/f'{mode}_nodes.csv', index=False)
        calibration_bins(stats['confidence'], stats['pred'] == true).to_csv(path/f'{mode}_calibration.csv', index=False)
        if bayes and capture:
            moments.save(path, data.edge_index[0].cpu().numpy(), node_ids, stats,
                         data.num_nodes, rotation=cfg['model'] == 'BayesBundleSheaf')
    return outputs


def cases(options):
    for dataset in options.datasets:
        for family in options.families:
            base = json.loads((ROOT/'config/paper_uq'/f'{dataset}_Bayes{family}Sheaf.json').read_text())
            for depth in options.depths:
                for split in options.splits:
                    for seed in options.seeds:
                        for prefix in ['Bayes', '']:
                            cfg = dict(base, model=f'{prefix}{family}Sheaf', layers=depth,
                                       seed=seed, split_index=split, folds=1,
                                       split_dir='splits/paper_48_32_20', gaussian_scale_convention='log_variance')
                            if options.epochs is not None:
                                cfg['epochs'] = options.epochs
                            if options.patience is not None:
                                cfg['early_stopping'] = options.patience
                            yield cfg


def case_path(output, cfg):
    return output/cfg['dataset']/cfg['model']/f"depth{cfg['layers']}"/f"split{cfg['split_index']}_seed{cfg['seed']}"


def report(output):
    from exp.uq_analysis.review_report import generate
    generate(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'review_results/v2')
    parser.add_argument('--datasets', nargs='+', default=['texas','wisconsin','film','cornell','citeseer','pubmed','cora'])
    parser.add_argument('--families', nargs='+', choices=['Diag','Bundle','General'], default=['Diag','Bundle','General'])
    parser.add_argument('--depths', nargs='+', type=int, default=[2,4,6])
    parser.add_argument('--splits', nargs='+', type=int, default=[0])
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(30)))
    parser.add_argument('--samples', nargs='+', type=int, default=[1,3,10,30,100])
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--skip-sheaf', action='store_true', help='Save disk/time by omitting map diagnostics')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--task-index', type=int, default=0, help='Disjoint shard index for batch arrays')
    parser.add_argument('--task-count', type=int, default=1)
    parser.add_argument('--no-report', action='store_true', help='Generate one report after all array tasks finish')
    parser.add_argument('--trial-indices', nargs='+', type=int, help='Optional 1-based manifest indices for diagnostics/retry')
    options = parser.parse_args()
    if options.report_only:
        report(options.output)
        return
    if min(options.samples) < 1 or min(options.depths) < 1 or (options.epochs is not None and options.epochs < 1):
        parser.error('Sample counts, depths and epochs must be positive')
    options.samples = sorted(set(options.samples))
    if max(options.samples) < 2:
        parser.error('Include at least one sample count >= 2 for UQ')
    if options.task_count < 1 or not 0 <= options.task_index < options.task_count:
        parser.error('Require 0 <= task-index < task-count')
    options.output.mkdir(parents=True, exist_ok=True)
    configurations = list(cases(options))
    manifest = dict(version=VERSION, configurations=configurations, sample_counts=options.samples,
                    capture_sheaf=not options.skip_sheaf,
                    protocol='Matched fixed configurations; no independent hyperparameter tuning. Validation selects epoch; test only evaluated after training.',
                    partitions='New 48/32/20 splits, not original paper partitions',
                    device=options.device, torch_version=torch.__version__, numpy_version=np.__version__)
    manifest_path = options.output/'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in ['version','configurations','sample_counts','capture_sheaf']:
            if previous[key] != manifest[key]:
                raise ValueError(f'{key} differs from existing study; choose a new output directory')
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'{len(configurations)} training trials; {len(options.samples)} inference budgets; output {options.output}', flush=True)
    if options.plan_only:
        return
    torch.set_num_threads(options.threads)
    device = torch.device(options.device)
    failed = []
    for number, cfg in enumerate(configurations, 1):
        if options.trial_indices is not None and number not in options.trial_indices:
            continue
        if (number-1) % options.task_count != options.task_index:
            continue
        path = case_path(options.output, cfg)
        path.mkdir(parents=True, exist_ok=True)
        try:
            ckpt_path = path/'checkpoint.pt'
            if ckpt_path.exists():
                checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                if checkpoint['config'] != cfg or checkpoint.get('study_version') != VERSION:
                    raise ValueError(f'Checkpoint configuration mismatch: {path}')
            elif options.evaluate_only:
                print('Missing checkpoint:', path, flush=True)
                continue
            else:
                print(f'Training {number}/{len(configurations)} {path}', flush=True)
                checkpoint = train_trial(cfg, path, device)
            metrics_path = path/'metrics.json'
            if not metrics_path.exists():
                metrics = evaluate(checkpoint, path, options.samples, device, not options.skip_sheaf)
                metrics_path.write_text(json.dumps(metrics, indent=2)+'\n')
                print('Evaluated:', path, flush=True)
        except (FloatingPointError, AssertionError) as error:
            if isinstance(error, AssertionError) and not any(
                    'isfinite' in (entry.line or '') for entry in traceback.extract_tb(error.__traceback__)):
                raise
            failure = dict(trial_index=number, config=cfg, error=str(error),
                           traceback=traceback.format_exc(), resolved=False)
            (path/'failure.json').write_text(json.dumps(failure, indent=2)+'\n')
            failed.append(str(path))
            print('Recorded failed trial; continuing:', path, error, flush=True)
    if not options.no_report:
        report(options.output)
    if failed:
        print("Unresolved numerical failures:", failed, flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
