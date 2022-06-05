import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from embeddings import prepare_model_brno, prepare_model_clova, prepare_model_speechbrain
from embeddings.extraction import extract_embeddings_path
from backend import transform_embeddings, prepare_plda
import backend
import evaluation.metrics as metrics


data_root_vox1_test = "/home/alexey/Documents/data/vox1_test/wav"

# https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt
trials_file = "meta/voxceleb/veri_test2.txt"


DEVICE = "cuda:0"

model_path_brno = "pretrained/brno/ResNet101_16kHz/nnet/raw_81.pth"
#model_path_brno = "pretrained/brno/ResNet101_16kHz/nnet/final.onnx"
model_path_speechbrain = "pretrained/speechbrain/embedding_model.ckpt"
model_path = "pretrained/clova/baseline_v2_ap.model"


for EMBEDDINGS_NAME in ["brno", "speechbrain", "clova"]:
    
    print(f"Embeddings: {EMBEDDINGS_NAME}")

    if EMBEDDINGS_NAME == "brno":
        emb_model = prepare_model_brno(model_path_brno, DEVICE, "onnx" if model_path_brno.endswith("onnx") else "pytorch")

    elif EMBEDDINGS_NAME == "speechbrain":
        emb_model = prepare_model_speechbrain(model_path_speechbrain, DEVICE)

    elif EMBEDDINGS_NAME == "clova":
        emb_model = prepare_model_clova(model_path, DEVICE)
    else:
        print(f"Embeddings extractor '{EMBEDDINGS_NAME}' is not implemented yet")
        exit()


    # Extract embeddings from speech

    cache_emb_file = f"emb_vox1_test_{EMBEDDINGS_NAME}.npz"

    batch_size = 1

    crop_size = None
    crop_center = True # if True crop_size must be a number

    path2utt_vox_fn = lambda wav_path: wav_path.split("/wav/")[-1]
    utt2spk_vox_fn = lambda utt: utt.split("/")[0]
    file2utt_vox_fn = lambda wav: f"ID{wav[2:-4].replace('/', '-')}"

    if not os.path.exists(cache_emb_file):

        embeddings, utt_ids = extract_embeddings_path(emb_model, 
                                                      data_root_vox1_test, 
                                                      DEVICE,
                                                      path2utt_vox_fn, 
                                                      batch_size=batch_size, 
                                                      crop_size=crop_size, 
                                                      crop_center=crop_center)

        #embeddings = torch.cat(embeddings).cpu().numpy()

        utt_ids_new = []
        for utt in utt_ids:
            utt_new = file2utt_vox_fn(utt)
            utt_ids_new += [utt_new]

        np.savez(cache_emb_file, X=embeddings, ids=utt_ids_new)


    # Load the VoxCeleb1 protocol

    get_spk = lambda utt_id: utt_id.split("-")[0]
    data = pd.read_csv(trials_file, sep=" ", header=None)
    labels_trials = data[0].to_numpy()

    trials = []
    for utt1, utt2 in zip(data[1], data[2]):
        pair = (file2utt_vox_fn(utt1), file2utt_vox_fn(utt2))
        trials += [pair]


    # Load and preprocess embeddings

    data = np.load(cache_emb_file)
    X_test = data["X"]
    ids_test = data["ids"]


    # Load pre-trained backend

    if EMBEDDINGS_NAME == "brno":
        plda_mu, plda_tr, plda_psi = prepare_plda("brno", "pretrained/brno")

        lda_dim = 128
        
        X_test = transform_embeddings(X_test, "brno", "pretrained/brno")
        X_test = (X_test - plda_mu).dot(plda_tr.T)[:, :lda_dim]

        mean = np.zeros(lda_dim)
        invW = np.eye(lda_dim)
        V = np.diag(np.sqrt(plda_psi[:lda_dim]))

        B = np.diag(plda_psi[:lda_dim])
        W = np.eye(lda_dim)

        P, Q, c, k = backend.plda.get_plda_score_params(torch.tensor(B), torch.tensor(W))

        plda_similarity = lambda x1, x2: backend.plda_score(x1, x2, P, Q, c, k)

        similarity_score = plda_similarity

    else:
        similarity_score = backend.cosine_similarity

    utt2emb_test = {utt: emb for (utt, emb) in zip(ids_test, X_test)}    
    
    
    # Compute scores

    scores = []
    mask = []
    for u1, u2 in tqdm(trials):
        e1 = utt2emb_test[u1].reshape(1, -1)# + 1e-6
        e2 = utt2emb_test[u2].reshape(1, -1)# + 1e-6
        e1 = torch.tensor(e1)
        e2 = torch.tensor(e2)
        score = similarity_score(e1, e2).numpy()
        if np.isnan(score):
            print("NaN score (will be ignored):", u1, u2)
            mask += [0]
        else:
            mask += [1]
        scores += [score]
    scores = np.array(scores).ravel()
    mask = np.array(mask) > 0.5

    # Compute equal error rate (EER)

    eer, _ = metrics.calculate_eer(scores[mask], labels_trials[mask], pos_label=1)
    print(f"EER: {eer*100:.2f}%", ) 
    
    # "brno": 0.82%
    # "speechbrain": 0.90%
    # "clova": 1.18%
    
