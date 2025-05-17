#! /usr/bin/env python
# Copyright 2022 Twitter, Inc.
# Modifications 2024 Patrick Gillespie
# Modifications 2025 Layal Bou Hamdan
# SPDX-License-Identifier: Apache-2.0

import os
import random
import torch
import torch.nn.functional as F
import git
import numpy as np
import wandb
from tqdm import tqdm
from exp.parser import get_parser
from models.positional_encodings import append_top_k_evectors
from models.bayes_disc_models import BayesDiagSheafDiffusion, BayesBundleSheafDiffusion, BayesGeneralSheafDiffusion
from utils.heterophilic import get_dataset, get_fixed_splits

def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def train(model, optimizer, data, config, epoch):
    model.train()
    optimizer.zero_grad()
    beta = torch.sigmoid(torch.tensor((epoch % 40) / 2 - 10))

    if config.bayes_model:
        out, kl = model(data.x)
        out = out[data.train_mask]
        nll = F.nll_loss(out, data.y[data.train_mask])
        loss = nll + beta * kl if config.use_kl else nll
    else:
        out = model(data.x)[data.train_mask]
        loss = F.nll_loss(out, data.y[data.train_mask])

    loss.backward()
    optimizer.step()

def test(model, data, config):
    model.eval()
    with torch.no_grad():
        if config.bayes_model:
            probs_list = [torch.exp(model(data.x)[0]) for _ in range(config.num_ensemble)]
            probs = torch.mean(torch.stack(probs_list), 0, keepdim=True).squeeze(0)
            logits = torch.log(probs)
            kl = None
        else:
            logits = model(data.x)
            kl = None

        accs, losses = [], []
        for mask in (data.train_mask, data.val_mask, data.test_mask):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            loss = F.nll_loss(logits[mask], data.y[mask])
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, losses, kl

def run_exp(config, dataset, model_cls, fold, return_model=False):
    args = dict(config)
    data = dataset[0]
    fixed_seed = 0
    data = get_fixed_splits(data, config.dataset, fixed_seed, config.permute_masks).to(config.device)

    model = model_cls(data.edge_index, args).to(config.device)
    sheaf_params, other_params = model.grouped_parameters()
    optimizer = torch.optim.Adam([
        {'params': sheaf_params, 'weight_decay': config.sheaf_decay},
        {'params': other_params, 'weight_decay': config.weight_decay}
    ], lr=config.lr)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    best_epoch = bad_counter = 0

    for epoch in range(config.epochs):
        train(model, optimizer, data, config, epoch)
        (train_acc, val_acc, tmp_test_acc), (train_loss, val_loss, tmp_test_loss), kl = test(model, data, config)

        if fold == 0:
            wandb.log({
                f'fold{fold}_train_acc': train_acc,
                f'fold{fold}_train_loss': train_loss,
                f'fold{fold}_val_acc': val_acc,
                f'fold{fold}_val_loss': val_loss,
                f'fold{fold}_tmp_test_acc': tmp_test_acc,
                f'fold{fold}_tmp_test_loss': tmp_test_loss,
                f'fold{fold}_kl': kl,
            }, step=epoch)

        new_best = val_acc > best_val_acc if config.stop_strategy == 'acc' else val_loss < best_val_loss
        if new_best:
            best_val_acc, best_val_loss = val_acc, val_loss
            test_acc, best_epoch = tmp_test_acc, epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter >= config.early_stopping:
            break

    wandb.log({'best_test_acc': test_acc, 'best_val_acc': best_val_acc, 'best_epoch': best_epoch})
    keep_running = False if test_acc < args['min_acc'] else True

    if return_model and args.get("save_model", False):
        save_base = args.get("save_dir", "saved_models")
        save_direct = os.path.join(save_base, args['model'])  
        save_dir = os.path.join(save_direct, args['dataset']) 
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, f"{args['dataset']}_{args['model']}_seed{args['seed']}.pt")
        torch.save(model.state_dict(), save_path)

    return test_acc, best_val_acc, keep_running

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    try:
        repo = git.Repo(search_parent_directories=True)
        args.sha = repo.head.object.hexsha
    except Exception:
        args.sha = "unknown"

    dataset = get_dataset(args.dataset)
    if args.evectors > 0:
        dataset = append_top_k_evectors(dataset, args.evectors)

    args.bayes_model = 'Bayes' in args.model
    args.graph_size = dataset[0].x.size(0)
    args.input_dim = dataset.num_features
    args.output_dim = dataset.num_classes
    args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    if args.sheaf_decay is None:
        args.sheaf_decay = args.weight_decay

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    wandb.init(project="sheaf", config=vars(args), entity=args.entity)
    config = wandb.config

    model_cls_map = {
        'BayesDiagSheaf': BayesDiagSheafDiffusion,
        'BayesBundleSheaf': BayesBundleSheafDiffusion,
        'BayesGeneralSheaf': BayesGeneralSheafDiffusion,
    }

    model_cls = model_cls_map.get(config.model)
    if model_cls is None:
        raise ValueError(f'Unknown model {config.model}')

    results = []
    for fold in tqdm(range(config.folds)):
        test_acc, best_val_acc, continue_running = run_exp(config, dataset, model_cls, fold, return_model=True)
        results.append([test_acc, best_val_acc])
        if not continue_running:
            break

    test_acc_mean, val_acc_mean = np.mean(results, axis=0) * 100
    test_acc_std = np.std([r[0] for r in results]) * 100

    wandb.log({
        'test_acc': test_acc_mean,
        'val_acc': val_acc_mean,
        'test_acc_std': test_acc_std,
    })

    model_name = config.model if config.evectors == 0 else f"{config.model}+LP{config.evectors}"
    print(f"{model_name} on {config.dataset} | SHA: {config.sha}")
    print(f"Test acc: {test_acc_mean:.4f} +/- {test_acc_std:.4f} | Val acc: {val_acc_mean:.4f}")
    wandb.finish()

