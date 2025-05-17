import os
import torch
import numpy as np
import yaml
import pandas as pd
from tqdm import tqdm
from scipy.stats import entropy
from utils.heterophilic import get_dataset, get_fixed_splits
from models.bayes_disc_models import (
    BayesDiagSheafDiffusion,
    BayesBundleSheafDiffusion,
    BayesGeneralSheafDiffusion,
)

model_class_map = {
    'BayesDiagSheaf': BayesDiagSheafDiffusion,
    'BayesBundleSheaf': BayesBundleSheafDiffusion,
    'BayesGeneralSheaf': BayesGeneralSheafDiffusion,
}

def compute_ece(confidences, predictions, labels, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if np.sum(in_bin) > 0:
            bin_accuracy = np.mean(predictions[in_bin] == labels[in_bin])
            bin_confidence = np.mean(confidences[in_bin])
            ece += (np.sum(in_bin) / len(confidences)) * abs(bin_confidence - bin_accuracy)
    return ece

def main():
    all_datasets = ["cora", "pubmed", "texas", "film", "wisconsin", "cornell", "citeseer"]
    all_models = ["BayesDiagSheaf", "BayesGeneralSheaf", "BayesBundleSheaf"]

    for model_name in all_models:
        for dataset_name in all_datasets:
            yaml_path = f"config/{dataset_name}_{model_name}_30.yml"
            if not os.path.exists(yaml_path):
                print(f"Config not found: {yaml_path}, skipping.")
                continue

            try:
                with open(yaml_path, "r") as f:
                    sweep_cfg = yaml.safe_load(f)
                params = sweep_cfg["parameters"]
                seeds = params["seed"]["values"]
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                dataset = get_dataset(dataset_name)
                config_dict = {
                    key: val.get('value', val) if isinstance(val, dict) else val
                    for key, val in params.items()
                }
                config_dict.update({
                    'input_dim': dataset.num_features,
                    'output_dim': dataset.num_classes,
                    'model': model_name,
                    'dataset': dataset_name,
                    'device': device,
                    'graph_size': dataset[0].x.size(0),
                    'deg_normalised': False,
                    'linear': False,
                    'second_linear': False,
                    'sheaf_act': "tanh",
                    'sparse_learner': False,
                    'max_t': 1.0
                })

                model_cls = model_class_map[model_name]
                all_entropy, all_variance, all_mi, all_correct = [], [], [], []
                all_pred_labels, all_true_labels = [], []

                for seed in tqdm(seeds, desc=f"{model_name} - {dataset_name}"):
                    model_path = f"saved_models/{model_name}/{dataset_name}/{dataset_name}_{model_name}_seed{seed}.pt"
                    data = dataset[0]
                    data = get_fixed_splits(data, dataset_name, 0, config_dict['permute_masks']).to(device)

                    model = model_cls(data.edge_index, config_dict).to(device)
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.eval()

                    with torch.no_grad():
                        probs_list = [torch.softmax(model(data.x)[0], dim=1).cpu().numpy()
                                      for _ in range(config_dict['num_ensemble'])]

                    probs_array = np.stack(probs_list)
                    test_idx = data.test_mask.cpu().numpy().nonzero()[0]
                    probs_test = probs_array[:, test_idx, :]

                    mean_probs = probs_test.mean(axis=0)
                    entropy_vals = entropy(mean_probs.T)
                    variance_vals = np.var(probs_test, axis=0).mean(axis=1)

                    ens_mean_entropy = entropy(np.mean(probs_test, axis=0).T)
                    mean_ens_entropy = np.mean([entropy(p.T) for p in probs_test], axis=0)
                    mutual_info = ens_mean_entropy - mean_ens_entropy

                    pred_labels = mean_probs.argmax(axis=1)
                    true_labels = data.y[test_idx].cpu().numpy()
                    correct_flags = (true_labels == pred_labels).astype(int)

                    all_entropy.extend(entropy_vals)
                    all_variance.extend(variance_vals)
                    all_mi.extend(mutual_info)
                    all_correct.extend(correct_flags)
                    all_pred_labels.extend(pred_labels)
                    all_true_labels.extend(true_labels)

                output_dir = f"uq/{model_name}/{dataset_name}"
                os.makedirs(output_dir, exist_ok=True)
                pd.DataFrame({
                    "true_label": all_true_labels,
                    "predicted_label": all_pred_labels,
                    "correct": all_correct,
                    "entropy": all_entropy,
                    "variance": all_variance,
                    "mutual_information": all_mi
                }).to_csv(f"{output_dir}/uq_results.csv", index=False)

                pred_confidences = mean_probs.max(axis=1)
                ece = compute_ece(pred_confidences, pred_labels, true_labels)
                print(f"{model_name} | {dataset_name} — ECE: {ece:.4f}")

            except Exception as e:
                print(f"Skipping {model_name} - {dataset_name} due to error: {e}")
                continue

if __name__ == "__main__":
    main()
