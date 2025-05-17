import torch
from scipy.stats import special_ortho_group


class KLDivergence(torch.nn.Module):
    def __init__(self, distribution, dim=None):
        super(KLDivergence, self).__init__()

        self.dim = dim
        self.dist = distribution

    def cayley_pdf(self, P, n, kappa):

        constant = (1 - kappa ** 2) ** (n * (n - 1) // 2)
        I = torch.eye(n, device=kappa.device).view(1, n, n).tile(P.size(0), 1, 1)
        det = torch.linalg.det(P - kappa.unsqueeze(-1) * I) ** (n - 1)

        return constant / det

    def cayley_transform(self, A):
        """If A is skew symmetric, return special orthogonal matrix Q.
        Since this version of the Cayley transform is an involution,
        can also take as input a special orthogonal matrix and return a skew symmetric"""
        Id = torch.eye(self.dim, dtype=A.dtype, device=A.device)
        Q = torch.linalg.solve(torch.add(Id, A, alpha=1.0), torch.add(Id, A, alpha=-1.0))
        return Q

    def forward(self, param1, param2):

        if self.dist == 'normal':

            mean, log_var = param1, param2
            inner = -(log_var - mean**2 - torch.exp(log_var) + 1) / 2
            kl = inner.sum(dim=1).mean(dim=0)

            return kl

        elif self.dist == 'cayley':
            kappas = (1 - param2) / (param2 + 1)

            if self.dim == 2:
                kl = (-torch.log(1 - kappas**2)).mean()
                return kl

            elif self.dim == 3:
                kl = (-torch.log(1 - kappas ** 2) - 2 * torch.log(1 - kappas) - 2 * kappas).mean()
                return kl

            else:
                gammas = param2.view(param2.size(0), 1, 1)
                so_uniform_samples = torch.tensor(
                    special_ortho_group.rvs(dim=self.dim, size=kappas.size(0)),
                    device=kappas.device, dtype=torch.float)
                cayley_samples = self.cayley_transform(gammas * self.cayley_transform(so_uniform_samples))
                densities = self.cayley_pdf(cayley_samples, self.dim, kappas)
                kl = torch.log(densities).mean()
                return kl

        else:
            raise ValueError(f"Unsupported distribution {self.dist}")


