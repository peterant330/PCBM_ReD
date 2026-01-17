import numpy as np
import torch
import pickle
from tqdm import tqdm
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="ham")
    args = parser.parse_args()
    data = args.data
    concept_emb = torch.from_numpy(np.load(f"file/{data}/concept_emb.npy")).cuda()
    train_emb = torch.from_numpy(np.load(f"file/{data}/image_emb_train.npy")).cuda().float()
    with open(f"file/{data}/concept.pkl", "rb") as f:
        d = pickle.load(f)
    with open(f"file/{data}/concept_rating.pkl", "rb") as f:
        rating = pickle.load(f)
    concept_list = []
    for i in list(d.keys()):
        concept_list.extend(d[i])
    concept_rating = []
    for i in rating:
        concept_rating.extend(rating[i])
    concept_rating = [int(i) for i in concept_rating]
    concept_filter = np.array([i>6 for i in concept_rating])
    concept_emb = concept_emb[concept_filter]
    concept_list = [i for i, j in zip(concept_list, concept_rating) if j >6]

    loss_min = torch.tensor(10000.)
    concept_candidate = []
    for i in range(len(concept_emb)):
        x = concept_emb[[i]].reshape(-1, 1)
        L = x @ x.T / (x.T @ x)
        loss = torch.mean(torch.square((torch.eye(768).cuda() - L) @ train_emb.T))
        if loss < loss_min:
            loss_min = loss
            selected_idx = i
    concept_candidate.append(concept_list.pop(selected_idx))
    X = concept_emb[[selected_idx]].T
    if selected_idx == len(concept_emb) - 1:
        concept_emb = concept_emb[:selected_idx]
    elif selected_idx == 0:
        concept_emb = concept_emb[1:]
    else:
        concept_emb = torch.concat([concept_emb[:selected_idx], concept_emb[selected_idx + 1:]], dim=0).contiguous()

    for step in tqdm(range(768)):
        A = X.T @ X
        R = X @ torch.linalg.inv(A) @ X.T
        loss_min = torch.tensor(10000.)
        selected_idx = None
        try:
            for i in range(len(concept_emb)):
                x = concept_emb[[i]].reshape(-1, 1)
                denominator = (x.T @ (torch.eye(768).cuda() - R) @ x)
                if torch.abs(denominator) < 0.01:
                    continue
                Q = x @ x.T / denominator
                L = R + R @ Q @ R - Q @ R - R @ Q + Q
                loss = torch.mean(torch.square((torch.eye(768).cuda() - L) @ train_emb.T))
                if loss < loss_min:
                    loss_min = loss
                    selected_idx = i
            if selected_idx is None:
                break
        except:
            break
        concept_candidate.append(concept_list.pop(selected_idx))
        X = torch.concat([X, concept_emb[[selected_idx]].T], dim=1)
        if selected_idx == len(concept_emb) - 1:
            concept_emb = concept_emb[:selected_idx]
        elif selected_idx == 0:
            concept_emb = concept_emb[1:]
        else:
            concept_emb = torch.concat([concept_emb[:selected_idx], concept_emb[selected_idx + 1:]], dim=0).contiguous()

    with open(f"file/{data}/selected_concept.pkl", "wb") as f:
        pickle.dump((X.detach().cpu().numpy(), concept_candidate), f)