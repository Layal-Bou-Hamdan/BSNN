"""Shared registry for matched Bayesian and deterministic sheaf experiments."""
from models.bayes_disc_models import (
    BayesDiagSheafDiffusion, BayesBundleSheafDiffusion, BayesGeneralSheafDiffusion,
)
from models.disc_models import (
    DiscreteDiagSheafDiffusion, DiscreteBundleSheafDiffusion, DiscreteGeneralSheafDiffusion,
)

MODELS = {
    'BayesDiagSheaf': BayesDiagSheafDiffusion,
    'BayesBundleSheaf': BayesBundleSheafDiffusion,
    'BayesGeneralSheaf': BayesGeneralSheafDiffusion,
    'DiagSheaf': DiscreteDiagSheafDiffusion,
    'BundleSheaf': DiscreteBundleSheafDiffusion,
    'GeneralSheaf': DiscreteGeneralSheafDiffusion,
}
