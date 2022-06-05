import os
import numpy as np
import torch
from data_io.kaldi_io import read_plda
from scipy.linalg import eigh
import h5py


l2_norm = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)


def transform_embeddings(X, embeddings_name, model_dir=""):
    
    if embeddings_name in ["clova", "speechbrain"]:
        X = l2_norm(X)
        
    elif embeddings_name == "brno":
        xvec_transform = os.path.join(model_dir, "ResNet101_16kHz/transform.h5")
        
        with h5py.File(xvec_transform, "r") as f:
            mean1 = np.array(f["mean1"])
            mean2 = np.array(f["mean2"])
            lda = np.array(f["lda"])
            X = l2_norm(np.dot(l2_norm(X - mean1), lda) - mean2)
    else:
        raise NotImplementedError
        
    return X
    
    
def prepare_plda(embeddings_name, model_dir=""):
    
    if embeddings_name == "brno":
    
        plda_file = os.path.join(model_dir, "ResNet101_16kHz/plda")

        kaldi_plda = read_plda(plda_file)
        plda_mu, plda_tr, plda_psi = [kaldi_plda[key] for key in ["mean", "transform", "psi"]]

        W = np.linalg.inv(plda_tr.T.dot(plda_tr))
        B = np.linalg.inv((plda_tr.T / plda_psi).dot(plda_tr))
        acvar, wccn = eigh(B, W)
        plda_mu = plda_mu.ravel()
        plda_psi = acvar[::-1]
        plda_tr = wccn.T[::-1]
        
    elif embeddings_name == "clova":
        dim = 512
        b = 0.00095
        w = 0.0010
        plda_mu = np.zeros(dim)
        plda_tr = np.eye(dim) / np.sqrt(w)
        plda_psi = np.ones(dim) * b / w
        
    elif embeddings_name == "speechbrain":
        dim = 192
        b = 0.0030
        w = 0.0020
        plda_mu = np.zeros(dim)
        plda_tr = np.eye(dim) / np.sqrt(w)
        plda_psi = np.ones(dim) * b / w
    
    return plda_mu, plda_tr, plda_psi