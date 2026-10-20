import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch
from exp.uq_analysis.compute_uq_metrics import predictive_statistics, ece_score, mc_sample_probs
from exp.uq_analysis.create_uq_plots_tables import generate


class UQTests(unittest.TestCase):
    def test_identical_uniform_predictions(self):
        stats = predictive_statistics(np.full((3, 2, 2), .5))
        np.testing.assert_allclose(stats['entropy'], np.log(2))
        np.testing.assert_allclose(stats['mutual_information'], 0)
        np.testing.assert_allclose(stats['variance'], 0)

    def test_disagreeing_certain_predictions(self):
        stats = predictive_statistics(np.array([[[1., 0.]], [[0., 1.]]]))
        np.testing.assert_allclose(stats['mutual_information'], np.log(2))
        np.testing.assert_allclose(stats['variance'], .25)
        self.assertEqual(ece_score(np.array([1., 1.]), np.array([0,1]), np.array([0,0])), .5)

    def test_sampling_mode_and_shape(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.log_softmax(torch.randn(len(x), 2), -1), None
        model = Model().train()
        samples = mc_sample_probs(model, torch.ones(4, 1), 3)
        self.assertEqual(samples.shape, (3,4,2))
        self.assertTrue(model.training)
        with self.assertRaises(ValueError):
            mc_sample_probs(model, torch.ones(4,1), 1)

    def test_partial_plot_grid_and_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial = root/'uq'/'BayesDiagSheaf'/'texas'/'seed0_split0'
            trial.mkdir(parents=True)
            pd.DataFrame({'entropy':[.2,.4], 'variance':[0,.01],
                          'mutual_information':[0,.1], 'confidence':[.6,1.], 'correct':[1,0]}).to_csv(trial/'uq_results.csv', index=False)
            (trial/'summary.json').write_text(json.dumps({'seed':0, 'split':0, 'ece':.1}))
            generate(root/'uq',root/'plots')
            for file in ['entropy_variance.pdf','mutual_information.pdf','table2.csv','table3.csv','ece_comparison.pdf','reliability_diagrams.pdf','reliability_bins.csv']:
                self.assertTrue((root/'plots'/file).is_file())
            bins = pd.read_csv(root/'plots'/'reliability_bins.csv')
            self.assertEqual(bins['count'].sum(), 2)
            self.assertEqual(bins.iloc[-1]['accuracy'], 0)
            self.assertEqual(bins.iloc[-1]['confidence'], 1)

if __name__ == '__main__':
    unittest.main()
