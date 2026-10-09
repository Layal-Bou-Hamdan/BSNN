"""Compute paper UQ metrics separately for each trained seed."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch


def predictive_statistics(samples):
    """samples: posterior predictive probabilities [samples, nodes, classes]."""
    p = np.asarray(samples, dtype=np.float64)
    if p.ndim != 3 or p.shape[0] < 2 or p.shape[1] == 0:
        raise ValueError('Need at least two posterior samples and one node')
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(-1), 1, atol=1e-6):
        raise ValueError('Invalid probability vectors')
    def entropy(q):
        return -(q * np.log(np.clip(q, np.finfo(float).tiny, 1))).sum(-1)
    mean = p.mean(0)
    total = entropy(mean)
    expected = entropy(p).mean(0)
    return dict(entropy=total, expected_entropy=expected,
                mutual_information=np.maximum(total - expected, 0),
                variance=p.var(0).mean(-1), confidence=mean.max(-1),
                pred=mean.argmax(-1))


def calibration_bins(conf, correct, n_bins=15):
    conf, correct = np.asarray(conf), np.asarray(correct)
    if n_bins < 1 or conf.size == 0:
        raise ValueError('Need positive bin count and nonempty predictions')
    ids = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    rows = []
    for i in range(n_bins):
        mask = ids == i
        rows.append(dict(bin=i, lower=i/n_bins, upper=(i+1)/n_bins,
                         count=int(mask.sum()),
                         confidence=float(conf[mask].mean()) if mask.any() else np.nan,
                         accuracy=float(correct[mask].mean()) if mask.any() else np.nan))
    return pd.DataFrame(rows)


def ece_score(conf, pred, true, n_bins=15):
    bins = calibration_bins(conf, np.asarray(pred) == np.asarray(true), n_bins)
    return float((bins['count'] * (bins.accuracy - bins.confidence).abs()).sum() / len(conf))


@torch.no_grad()
def mc_sample_probs(model, data_x, S, indices=None):
    if S < 2:
        raise ValueError('UQ requires at least two posterior samples')
    was_training = model.training
    model.eval()  # BSNN Laplacian builders sample even in evaluation mode.
    try:
        samples = []
        for _ in range(S):
            log_probs, _ = model(data_x)
            if indices is not None:
                log_probs = log_probs[indices]
            samples.append(log_probs.exp().cpu().numpy())
        return np.stack(samples)
    finally:
        model.train(was_training)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', type=Path, default=ROOT/'saved_models')
    parser.add_argument('--output', type=Path, default=ROOT/'uq')
    parser.add_argument('--samples', type=int, default=3, help='Paper uses 3; increase for MC convergence studies')
    parser.add_argument('--seed', type=int, default=0, help='Posterior sampling RNG seed')
    options = parser.parse_args()
    paths = sorted(options.checkpoints.glob('*/*/*.pt'))
    if not paths:
        parser.error(f'No checkpoints in {options.checkpoints}; retrain with --save_model first')
    from utils.heterophilic import get_dataset, get_fixed_splits
    from models.positional_encodings import append_top_k_evectors
    from models.bayes_disc_models import BayesDiagSheafDiffusion, BayesBundleSheafDiffusion, BayesGeneralSheafDiffusion
    classes = dict(zip(['BayesDiagSheaf', 'BayesBundleSheaf', 'BayesGeneralSheaf'],
                       [BayesDiagSheafDiffusion, BayesBundleSheafDiffusion, BayesGeneralSheafDiffusion]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for path in paths:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict) or 'config' not in checkpoint:
            raise ValueError(f'{path}: checkpoint lacks training config; recover settings before evaluating')
        cfg = dict(checkpoint['config'])
        torch.manual_seed(options.seed + int(cfg['seed']))
        np.random.seed(options.seed + int(cfg['seed']))
        dataset = get_dataset(cfg['dataset'])
        if cfg.get('evectors', 0):
            dataset = append_top_k_evectors(dataset, cfg['evectors'])
        split = checkpoint['split']
        data = get_fixed_splits(dataset[0], cfg['dataset'], split, cfg['permute_masks']).to(device)
        cfg['device'] = device
        model = classes[cfg['model']](data.edge_index, cfg).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        idx = data.test_mask.nonzero(as_tuple=True)[0]
        stats = predictive_statistics(mc_sample_probs(model, data.x, options.samples, idx))
        true = data.y[idx].cpu().numpy()
        frame = pd.DataFrame(dict(node_idx=idx.cpu().numpy(), true=true, **stats))
        frame['correct'] = (frame.pred == frame.true).astype(int)
        frame['seed'], frame['split'] = cfg['seed'], split
        output = options.output/cfg['model']/cfg['dataset']/f"seed{cfg['seed']}_split{split}"
        output.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output/'uq_results.csv', index=False)
        calibration_bins(frame.confidence, frame.correct).to_csv(output/'calibration.csv', index=False)
        summary = dict(seed=cfg['seed'], split=split, samples=options.samples,
                       accuracy=float(frame.correct.mean()),
                       ece=ece_score(frame.confidence, frame.pred, frame.true),
                       checkpoint=str(path.resolve()), sampling_seed=options.seed + int(cfg['seed']))
        (output/'summary.json').write_text(json.dumps(summary, indent=2))
        print(output, summary['ece'])


if __name__ == '__main__':
    main()
