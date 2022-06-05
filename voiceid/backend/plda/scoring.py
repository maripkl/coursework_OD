import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import inv


# FULL COVARIANCE MODEL

def get_plda_score_params(B, W, m=None):
    dim = B.shape[0]
    Sigma_ac = B
    Sigma_tot = B + W
    Q = torch.linalg.inv(Sigma_tot) - torch.linalg.inv(Sigma_tot - Sigma_ac @ torch.linalg.inv(Sigma_tot) @ Sigma_ac)
    P = torch.linalg.inv(Sigma_tot) @ Sigma_ac @ torch.linalg.inv(Sigma_tot - Sigma_ac @ torch.linalg.inv(Sigma_tot) @ Sigma_ac)
    P = 0.5 * P
    Q = 0.5 * Q
    if m is None:
        c = torch.zeros(1, dim).to(B.dtype)
    else:
        c = -2 * m @ (P + Q)
    k = 0.5 * torch.logdet(Sigma_tot) - 0.5 * torch.logdet(Sigma_tot - Sigma_ac @ torch.linalg.inv(Sigma_tot) @ Sigma_ac)
    if m is not None:
        k = k + 2 * m.t() @ (P + Q) @ m
    return P, Q, c, k


def plda_score_single(x1, x2, B, W):
    P, Q, c, k = get_plda_score_params(B, W)
    quad_cross = x1 @ P @ x2.t() + x2 @ P @ x1.t()
    quad_same = x1 @ Q @ x1.t() + x2 @ Q @ x2.t()
    lin = c @ (x1 + x2).t()
    return quad_cross + quad_same + lin + k


def plda_score(X1, X2, B, W):
    P, Q, c, k = get_plda_score_params(B, W)
    quad_cross = X1 @ (P + P.t()) @ X2.t()
    quad_same1 = torch.sum((X1 @ Q) * X1, dim=1, keepdim=True)
    quad_same2 = torch.sum((X2 @ Q) * X2, dim=1, keepdim=True)
    lin1 = X1 @ c.t()
    lin2 = X2 @ c.t()
    return quad_cross + quad_same1 + quad_same2.t() + lin1 + lin2.t() + k


def plda_score_many2many_single(X_e, X_t, B, W): 
    
    # computes a single score between two sets
    
    dim = X_e.shape[1]
    
    n_e = X_e.shape[0]
    n_t = X_t.shape[0]
    
    B_inv = inv(B)
    W_inv = inv(W)
    
    a_e = torch.sum(X_e, dim=0, keepdim=True) @ W_inv
    a_t = torch.sum(X_t, dim=0, keepdim=True) @ W_inv 
    
    Sigma_e_inv = B_inv + n_e * W_inv
    Sigma_t_inv = B_inv + n_t * W_inv
    Sigma_e = inv(Sigma_e_inv) 
    Sigma_t = inv(Sigma_t_inv) 
    Sigma_inv_sum = Sigma_e_inv + Sigma_t_inv - B_inv

    a = a_e  + a_t
    mu_quad_term = -0.5 * a_e @ Sigma_e @ a_e.t() - 0.5 * a_t @ Sigma_t @ a_t.t()
    return 0.5 * a @ inv(Sigma_inv_sum) @ a.t() + mu_quad_term + 0.5 * torch.logdet(B) + 0.5 * torch.logdet(Sigma_e_inv) + 0.5 * torch.logdet(Sigma_t_inv) - 0.5 * torch.logdet(Sigma_inv_sum)



# DIAGONAL MODEL

def plda_score_diag_many2many_single(X_e, X_t, w_diag): # UNTESTED
    
    # computes a single score between two sets

    # b_diag = ones

    dim = X_e.shape[1]
    
    n_e = X_e.shape[0]
    n_t = X_t.shape[0]
    
    w_diag_inv = 1/w_diag.view(1, -1)
    
    a_e = torch.sum(X_e, dim=0, keepdim=True) * w_diag_inv
    a_t = torch.sum(X_t, dim=0, keepdim=True) * w_diag_inv
    
    sigma_e_inv = 1 + n_e * w_diag_inv
    sigma_t_inv = 1 + n_t * w_diag_inv
    sigma_e = 1/sigma_e_inv
    sigma_t = 1/sigma_t_inv 
    sigma_inv_sum = sigma_e_inv + sigma_t_inv - 1
    
    const = 0.5 * torch.sum(torch.log(sigma_e_inv)) + 0.5 * torch.sum(torch.log(sigma_t_inv)) - 0.5 * torch.sum(torch.log(lmsigma_inv_sumbda))
    
    a = a_e + a_t
    mu_quad_term = -0.5 * torch.sum(a_e**2 * sigma_e) - 0.5 * torch.sum(a_t**2 * sigma_t)
    return 0.5 * torch.sum(a**2 / sigma_inv_sum) + mu_quad_term + const
   

def plda_score_diag_many2one(X_e, X_t, w_diag): # UNTESTED
    
    # computes a vector of scores: 1 envoll (consisting of multiple vectors) vs. N test vectors

    dim = X_e.shape[1]
    
    n_e = X_e.shape[0]
    n_t = 1

    w_diag_inv = 1 / w_diag.view(1, -1)
    
    c_e = torch.mean(X_e, dim=0, keepdim=True)
    c_t = X_t

    a_e = n_e * c_e * w_diag_inv
    a_t = n_t * c_t * w_diag_inv

    sigma_e_inv = 1 + n_e * w_diag_inv
    sigma_t_inv = 1 + n_t * w_diag_inv
    sigma_e = 1 / sigma_e_inv
    sigma_t = 1 / sigma_t_inv
    sigma_inv_sum = sigma_e_inv + sigma_t_inv - 1
    
    const = 0.5 * torch.sum(torch.log(sigma_e_inv)) + 0.5 * torch.sum(torch.log(sigma_t_inv)) - 0.5 * torch.sum(torch.log(sigma_inv_sum))

    a = a_e + a_t
    mu_quad_term = -0.5 * torch.sum(a_e ** 2 * sigma_e) - 0.5 * torch.sum(a_t ** 2 * sigma_t, dim=1)
    return 0.5 * torch.sum(a ** 2 / sigma_inv_sum, dim=1) + mu_quad_term + const


# ISOTROPIC MODEL
    
def plda_score_scalar_up(X1, X2, b, w, unc_var_e=0., unc_var_t=0.):
    
    # computes a matrix of all-vs-all scores

    dim = X1.shape[1]
    
    sigma_s = b + w + unc_var_e
    sigma_t = b + w + unc_var_t
    sigma_ac = b
    
    a_st = 1/sigma_s - 1/(sigma_s - sigma_ac**2/sigma_t)
    b_st = 1/sigma_s * sigma_ac * 1/(sigma_t - sigma_ac**2/sigma_s)
    c_st = 1/sigma_t - 1/(sigma_t - sigma_ac**2/sigma_s)
    d_st = -0.5 * torch.log(sigma_s * sigma_t - sigma_ac**2) + 0.5 * torch.log(sigma_s * sigma_t)
        
    #scale = b_st
    #shift = 0.5 * a_st + 0.5 * c_st + d_st * dim
    #cosine = cosine_similarity(x_e, x_t)
    #return scale * cosine + shift
    
    dot_e = torch.sum(X1**2, dim=1, keepdim=True)
    dot_t = torch.sum(X2**2, dim=1, keepdim=True).t()
    dot_et = torch.mm(X1, X2.t())
    return b_st * dot_et + 0.5 * a_st * dot_e + 0.5 * c_st * dot_t +  d_st * dim


def plda_score_scalar_many2many(centroids_enroll, centroids_test, b, w):
    
    # inputs are pairs (centroids, counts)
    # computes a matrix of all-vs-all scores

    b_inv = 1 / b
    w_inv = 1 / w

    centroids_e, n_e = centroids_enroll
    centroids_t, n_t = centroids_test

    dim = centroids_e.shape[1]

    n_e = n_e.view(-1, 1)
    n_t = n_t.view(-1, 1)   

    a_e = n_e * centroids_e * w_inv
    a_t = n_t * centroids_t * w_inv 

    n_t = n_t.t()

    sigma_e_inv = b_inv + n_e * w_inv
    sigma_t_inv = b_inv + n_t * w_inv
    sigma_e = 1 / sigma_e_inv
    sigma_t = 1 / sigma_t_inv
    sigma_inv_sum = sigma_e_inv + sigma_t_inv - b_inv

    const = dim * (0.5 * torch.log(b) + 0.5 * torch.log(sigma_e_inv) + 0.5 * torch.log(sigma_t_inv) - 0.5 * torch.log(sigma_inv_sum))

#     mu_quad_term = -0.5 * sigma_e * a_e @ a_e.t() - 0.5 * sigma_t * a_t @ a_t.t()
#     a = a_e  + a_t
#     return 0.5 / sigma_inv_sum * a @ a.t() + mu_quad_term + const

    a_e_sqr = torch.sum(a_e**2, dim=1, keepdim=True)
    a_t_sqr = torch.sum(a_t**2, dim=1, keepdim=True).t()
    mu_quad_term = -0.5 * (a_e_sqr * sigma_e + a_t_sqr * sigma_t)
    a_sqr = a_e_sqr + a_t_sqr + 2 * a_e @ a_t.t()
    return 0.5 / sigma_inv_sum * a_sqr + mu_quad_term + const
    
    
def logmvnpdf(X, mu, Sigma):
    dim = X.shape[1]
    var = Sigma.view(1, -1)
    return -0.5 * dim * math.log(2 * math.pi) - 0.5 * dim * torch.log(var) - 0.5 * (torch.sum(X**2, dim=1, keepdim=True) + torch.sum(mu**2, dim=1, keepdim=True).t() - 2 * torch.mm(X, mu.t())) / var


def init_classes(centroids, counts, b, w):
    b_inv, w_inv = 1/b, 1/w
    counts = counts.view(-1, 1)
    Sigma = 1 / (b_inv + counts * w_inv * 1) # (K,)
    mu = Sigma * (w_inv * centroids * counts) # (K, dim)
    return mu, Sigma


def log_predictive(X, centroids, counts, b, w):
    mu, Sigma = init_classes(centroids, counts, b, w)
    return logmvnpdf(X, mu, Sigma + w)