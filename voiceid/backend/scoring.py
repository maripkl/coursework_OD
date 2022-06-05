import numpy as np
import torch


def cosine_similarity(a, b, numpy=False):
    if numpy:
        a,b = torch.tensor(a),torch.tensor(b)
    a,b = a / torch.norm(a, dim=1, keepdim=True), b / torch.norm(b, dim=1, keepdim=True)
    if numpy:
        return torch.mm(a, b.t()).numpy()
    else:
        return torch.mm(a, b.t())#.numpy()
    
    
def plda_score(x1, x2, P, Q, c, k):
    quad_cross = x1 @ P @ x2.t() + x2 @ P @ x1.t()
    quad_same = x1 @ Q @ x1.t() + x2 @ Q @ x2.t()
    lin = c @ (x1 + x2).t()
    return quad_cross + quad_same + lin + k