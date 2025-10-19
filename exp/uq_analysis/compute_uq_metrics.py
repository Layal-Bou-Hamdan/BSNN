import sys, os, yaml, torch, numpy as np, pandas as pd
from tqdm import tqdm
from scipy.stats import entropy
from sklearn.calibration import calibration_curve

# --- Project paths ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT); sys.path.append(os.path.join(PROJECT_ROOT, "lib"))

from utils.heterophilic import get_dataset, get_fixed_splits
from models.bayes_disc_models import (
    BayesDiagSheafDiffusion, BayesBundleSheafDiffusion, BayesGeneralSheafDiffusion
)

model_class_map = {
    'BayesDiagSheaf': BayesDiagSheafDiffusion,
    'BayesBundleSheaf': BayesBundleSheafDiffusion,
    'BayesGeneralSheaf': BayesGeneralSheafDiffusion
}

def ece_score(conf, pred, true, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & ((conf < hi) if i < n_bins - 1 else (conf <= hi))
        if mask.any():
            acc = np.mean(pred[mask] == true[mask])
            cal = np.mean(conf[mask])
            ece += mask.mean() * abs(cal - acc)
    return ece

@torch.no_grad()
def mc_sample_probs(model, data_x, S):
    """
    Returns probs of shape [S, N, C]. Tries true posterior sampling, else MC-dropout, else deterministic.
    """
    probs = []
    # Prefer explicit posterior sampling if the model provides it
    has_sampler = hasattr(model, "sample_posterior_")
    has_mcdo   = hasattr(model, "enable_mc_dropout") or hasattr(model, "mc_dropout")
    for _ in range(S):
        if has_sampler:
            model.sample_posterior_()
            model.eval()
        elif has_mcdo:
            # keep dropout active without changing BN stats (no BN in many GNNs; if present, consider eval-BN)
            if hasattr(model, "enable_mc_dropout"): model.enable_mc_dropout()
            model.train()
        else:
            model.eval()
        logits = model(data_x)[0]              
        p = torch.softmax(logits, dim=1).cpu().numpy()
        probs.append(p)
    return np.stack(probs, axis=0)              

all_datasets = ["cora", "pubmed", "texas", "film", "wisconsin", "cornell", "citeseer"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for dataset_name in all_datasets:
    yaml_path = f"config/bayesgen/{dataset_name}_bayesgen_30.yml"
    if not os.path.exists(yaml_path):
        print(f"YAML not found: {yaml_path}"); continue

    try:
        sweep_cfg = yaml.safe_load(open(yaml_path, "r"))
        params = sweep_cfg["parameters"]
        seeds = params["seed"]["values"]
        model_name = params["model"]["value"]
        dataset = get_dataset(dataset_name)
        data = dataset[0]
        # fixed split id 0; keep masks stable across seeds for eval
        data = get_fixed_splits(data, dataset_name, 0, params.get('permute_masks', False)).to(device)

        config_dict = {k: (v['value'] if isinstance(v, dict) and 'value' in v else v) for k, v in params.items()}
        config_dict.update({
            'input_dim': dataset.num_features,
            'output_dim': dataset.num_classes,
            'model': model_name,
            'dataset': dataset_name,
            'device': device,
            'graph_size': data.x.size(0),
            'deg_normalised': False,
            'linear': False,
            'second_linear': False,
            'sheaf_act': "tanh",
            'sparse_learner': False,
            'max_t': 1.0
        })
        S_per_model = int(config_dict.get('num_ensemble', 1))
        model_cls = model_class_map[model_name]

        test_idx = data.test_mask.cpu().numpy().nonzero()[0]
        y_true = data.y[test_idx].cpu().numpy()

        # --- collect posterior samples across all seeds/models ---
        samples_all = []  # list of [S, n_test, C]
        for seed in tqdm(seeds, desc=f"{dataset_name} seeds"):
            model_path = f"saved_models/{model_name}/{dataset_name}/{dataset_name}_{model_name}_seed{seed}.pt"
            model = model_cls(data.edge_index, config_dict).to(device)
            state = torch.load(model_path, map_location=device)
            model.load_state_dict(state); model.eval()

            probs = mc_sample_probs(model, data.x, S=S_per_model)         
            probs_test = probs[:, test_idx, :]                       
            samples_all.append(probs_test)

        probs_test = np.concatenate(samples_all, axis=0)                   
        assert np.allclose(probs_test.sum(axis=-1), 1.0, atol=1e-6)

        # --- predictive stats ---
        mean_probs = probs_test.mean(axis=0)                        
        ent_mean = entropy(mean_probs.T)                                  
        ent_each = np.array([entropy(p.T) for p in probs_test])           
        mi = ent_mean - ent_each.mean(axis=0)                            
        var_prob = probs_test.var(axis=0).mean(axis=1)                    

        y_pred = mean_probs.argmax(axis=1)
        conf = mean_probs.max(axis=1)
        correct = (y_pred == y_true).astype(int)

        # top-k coverage and sharpness from predictive mean
        k = 3
        topk = np.argsort(mean_probs, axis=1)[:, -k:]
        topk_cov = np.mean([y_true[i] in topk[i] for i in range(len(y_true))])
        sharpness = conf.mean()

        # --- calibration ---
        ece = ece_score(conf, y_pred, y_true, n_bins=15)
        # For plotting calibration curve if desired:
        correct_bin = (y_pred == y_true).astype(int)
        prob_true, prob_pred = calibration_curve(correct_bin, conf, n_bins=15, strategy="uniform")

        # --- save per-node table ---
        output_dir = f"uq/{model_name}/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame({
            "node_idx": test_idx,
            "entropy": ent_mean,
            "mutual_information": mi,
            "prob_variance_mean": var_prob,
            "confidence": conf,
            "pred": y_pred,
            "true": y_true,
            "correct": correct
        })
        df.to_csv(f"{output_dir}/uq_results2.csv", index=False)

        # --- summary ---
        print(f"[{dataset_name}] n_test={len(y_true)}  ECE={ece:.4f}  Top-{k}={topk_cov:.4f}  Sharpness={sharpness:.4f}")

    except Exception as e:
        print(f"Skipping {dataset_name}: {e}")
