import math
import numpy as np
import torch

from scoring import *

torch.manual_seed(0)

dim = 10

b = 0.1
w = 0.01

b, w = torch.tensor([b]), torch.tensor([w])

# 
# X = torch.randn(N, dim)
# y = torch.randint(2, size=(N,))

I = torch.eye(dim)
B_iso = b * I
W_iso = w * I

N = 3
M = 4
X1 = torch.randn(N, dim)
X2 = torch.randn(M, dim)

# X1 = F.normalize(X1, dim=1)
# X2 = F.normalize(X2, dim=1)

sim_mat = plda_score(X1, X2, B_iso, W_iso)

sim_mat2 = plda_score_scalar_one2one_up(X1, X2, b, w)

centroids_enroll = (X1, torch.ones(N,))
centroids_test = (X2, torch.ones(M,))
sim_mat3 = plda_score_scalar_many2many(centroids_enroll, centroids_test, b, w)

sim_mat4_upper_left = plda_score_many2many_single(X1[0:1], X2[0:1], B_iso, W_iso)

print(sim_mat)
print(sim_mat2)
print(sim_mat3)
print(sim_mat4_upper_left)

##

print("-------------")

enroll_sets = [torch.randn(torch.randint(1, 10, (1,)), dim) for _ in range(4)]
test_sets = [torch.randn(torch.randint(1, 10, (1,)), dim) for _ in range(5)]

c_e = torch.cat([torch.mean(X, dim=0, keepdim=True) for X in enroll_sets])
c_t = torch.cat([torch.mean(X, dim=0, keepdim=True) for X in test_sets])

n_e = torch.tensor([X.shape[0] for X in enroll_sets])
n_t = torch.tensor([X.shape[0] for X in test_sets])

sim_mat = plda_score_scalar_many2many((c_e, n_e), (c_t, n_t), b, w)


sim_mat2 = torch.zeros_like(sim_mat)
for i, X_e in enumerate(enroll_sets):
    for j, X_t in enumerate(test_sets):
        sim_mat2[i, j] = plda_score_many2many_single(X_e, X_t, B_iso, W_iso)
        
        
print(sim_mat)
print(sim_mat2)