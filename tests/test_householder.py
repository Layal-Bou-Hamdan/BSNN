import unittest
import torch
from models.householder import householder_product
from models.orthogonal_v2 import Orthogonal


class HouseholderTests(unittest.TestCase):
    def test_product_reference_and_special_orthogonality(self):
        torch.manual_seed(7)
        for d in [1,2,3,4,7]:
            a = torch.randn(3,d,d,dtype=torch.double)
            vectors = a.tril(-1)+torch.eye(d,dtype=torch.double)
            tau = 2/vectors.square().sum(-2)
            expected = torch.linalg.householder_product(vectors,tau)
            torch.testing.assert_close(householder_product(a),expected)
            x = torch.randn(3,d*(d-1)//2,dtype=torch.double)
            q = Orthogonal(d,'householder')(x)
            torch.testing.assert_close(q.transpose(-1,-2)@q,torch.eye(d,dtype=q.dtype).expand_as(q))
            torch.testing.assert_close(torch.linalg.det(q),torch.ones(3,dtype=q.dtype))

    def test_gradients(self):
        x = torch.randn(2,6,dtype=torch.double,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(Orthogonal(4,'householder'),(x,)))

if __name__ == '__main__':
    unittest.main()
