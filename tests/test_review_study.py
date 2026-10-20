import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import torch
from torch import nn
from exp.parser import get_parser
from exp.inference import inference_mode, log_probabilities
from exp.uq_analysis.review_study import MapMoments, scores
from exp.uq_analysis.review_study import guarded_step, train_trial
from types import SimpleNamespace
from unittest.mock import patch
from models.registry import MODELS


class FixedGaussianLearner(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x, edge_index):
        shape = (edge_index.shape[1], self.size)
        return x.new_zeros(shape), x.new_full(shape, -.7)


class ReviewStudyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(9)
        self.edges = torch.tensor([[0,1,1,2,2,3,3,0],[1,0,2,1,3,2,0,3]])
        self.x = torch.randn(4,3)
        self.cfg = dict(vars(get_parser().parse_args([])), graph_size=4,
                        input_dim=3, output_dim=2, d=2, hidden_channels=3,
                        layers=2, device=torch.device('cpu'), dropout=.5, input_dropout=.5)

    def test_gaussian_kl_matches_sampled_distribution(self):
        for name, size in [('BayesDiagSheaf',2),('BayesGeneralSheaf',4)]:
            model = MODELS[name](self.edges,self.cfg).eval()
            model.sheaf_learner = FixedGaussianLearner(size)
            captured = []
            handle=model.laplacian_builder.register_forward_pre_hook(lambda m,args: captured.append(args))
            _, kl = model(self.x)
            handle.remove()
            mean, std = captured[0]
            torch.testing.assert_close(std,torch.full_like(std,np.exp(-.35)))
            posterior=torch.distributions.Normal(mean,std)
            prior=torch.distributions.Normal(torch.zeros_like(mean),torch.ones_like(std))
            expected=torch.distributions.kl_divergence(posterior,prior).sum(1).mean()
            torch.testing.assert_close(kl,expected)
            model.legacy_gaussian_scale=True
            handle=model.laplacian_builder.register_forward_pre_hook(lambda m,args: captured.append(args))
            model(self.x)
            handle.remove()
            torch.testing.assert_close(captured[-1][1],torch.full_like(std,np.exp(-.7)))

    def test_functional_mc_dropout_and_mode_restoration(self):
        for name in ['DiagSheaf','BundleSheaf','GeneralSheaf']:
            model=MODELS[name](self.edges,self.cfg).train()
            sentinel=nn.BatchNorm1d(3).eval()
            model.add_module('sentinel_batchnorm',sentinel)
            running=sentinel.running_mean.clone()
            with inference_mode(model,False):
                a=log_probabilities(model,self.x)
                b=log_probabilities(model,self.x)
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            with inference_mode(model,True):
                self.assertFalse(model.training)
                self.assertFalse(sentinel.training)
                draws=torch.stack([log_probabilities(model,self.x) for _ in range(10)])
                self.assertGreater(float(draws.var(0).sum()),0)
            self.assertTrue(model.training)
            self.assertFalse(sentinel.training)
            self.assertFalse(model.mc_dropout)
            torch.testing.assert_close(sentinel.running_mean,running)

    def test_proper_scores(self):
        perfect=np.array([[[1.,0.],[0.,1.]]])
        actual=scores(perfect,np.array([0,1]))
        self.assertEqual(actual['accuracy'],1)
        for metric in ['ece','nll','brier','variance','mutual_information']:
            self.assertEqual(actual[metric],0)
        uniform=scores(np.full((3,2,2),.5),np.array([0,1]))
        self.assertAlmostEqual(uniform['nll'],np.log(2))
        self.assertAlmostEqual(uniform['brier'],.5)

    def test_known_map_spread_and_node_aggregation(self):
        moments=MapMoments()
        moments.add(0,torch.tensor([[0.,0.],[0.,0.],[0.,0.]]))
        moments.add(0,torch.tensor([[2.,0.],[0.,4.],[0.,0.]]))
        with tempfile.TemporaryDirectory() as directory:
            moments.save(Path(directory),np.array([0,0,1]),np.array([0,1]),
                         dict(mutual_information=np.array([.1,0]),variance=np.array([.2,0])),2)
            nodes=pd.read_csv(Path(directory)/'sheaf_nodes.csv')
            np.testing.assert_allclose(nodes.sheaf_spread,[2.5,0])
            maps=np.load(Path(directory)/'sheaf_maps.npz')
            np.testing.assert_allclose(maps['layer0_variance'],[[1,0],[0,4],[0,0]])

    def test_bad_gradient_is_rejected_before_optimizer_update(self):
        model=nn.Linear(3,2)
        model.weight.register_hook(lambda grad: torch.full_like(grad,float('nan')))
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        data=SimpleNamespace(x=self.x,y=torch.tensor([0,1,0,1]),train_mask=torch.ones(4,dtype=torch.bool),edge_index=self.edges)
        optimizer=torch.optim.Adam(model.parameters())
        with self.assertRaisesRegex(FloatingPointError,'Nonfinite gradients'):
            guarded_step(model,optimizer,data,self.cfg,0,False)
        for k,v in model.state_dict().items():
            torch.testing.assert_close(v,before[k],rtol=0,atol=0)
        self.assertEqual(len(optimizer.state),0)

    def test_trial_preserves_best_finite_checkpoint_on_bad_gradient(self):
        class Model(nn.Linear):
            def grouped_parameters(self):
                return [],list(self.parameters())
        model=Model(3,2)
        calls=[0]
        def hook(grad):
            calls[0]+=1
            return grad if calls[0]==1 else torch.full_like(grad,float('nan'))
        model.weight.register_hook(hook)
        data=SimpleNamespace(x=self.x,y=torch.tensor([0,1,0,1]),train_mask=torch.ones(4,dtype=torch.bool),edge_index=self.edges)
        cfg=dict(self.cfg,model='DiagSheaf',dataset='fixture',seed=0,epochs=3,
                 sheaf_decay=0.,split_index=0,split_dir='unused')
        with tempfile.TemporaryDirectory() as directory, \
             patch('exp.uq_analysis.review_study.load_data',return_value=(data,cfg)), \
             patch.dict('exp.uq_analysis.review_study.MODELS',{'DiagSheaf':lambda *a:model}), \
             patch('exp.uq_analysis.review_study.validation',return_value=(.5,.7)), \
             patch('exp.uq_analysis.review_study.digest',return_value='fixture'):
            checkpoint=train_trial(cfg,Path(directory),torch.device('cpu'))
            self.assertEqual(checkpoint['termination'],'numerical_guard')
            self.assertEqual(checkpoint['best_epoch'],0)
            self.assertEqual(checkpoint['guard_event']['epoch'],1)
            self.assertEqual(checkpoint['guard_event']['stage'],'gradient_check_before_adam')
            self.assertTrue(all(torch.isfinite(x).all() for x in checkpoint['state_dict'].values()))
            self.assertTrue((Path(directory)/'numerical_guard.json').exists())


if __name__ == '__main__':
    unittest.main()
