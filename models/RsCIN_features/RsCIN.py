import numpy as np
import torch
import pickle

def MMO(W, score, k_list=[1, 2, 3]):
    S_list = []
    if 1 not in k_list:
        k_list.append(1)
    for k in k_list:
        _, topk_matrix = torch.topk(W.float(), W.shape[0]-k, largest=False, sorted=True)
        W_mask = W.clone()
        for i in range(W.shape[0]):
            W_mask[i, topk_matrix[i]] = 0
        n = W.shape[-1]
        D_ = torch.zeros_like(W).float()
        for i in range(n):
            D_[i, i] = 1 / (W_mask[i,:].sum())
        P = D_ @ W_mask
        S = score.clone().unsqueeze(-1)
        S = P @ S
        S_list.append(S)
    S = torch.concat(S_list, -1).mean(-1)
    return S

def RsCIN(scores_old, cls_tokens=None, k_list=[0]):
    if cls_tokens is None or 0 in k_list:
        return scores_old
    cls_tokens = np.array(cls_tokens)
    scores = (scores_old - scores_old.min()) / (scores_old.max() - scores_old.min())
    similarity_matrix = cls_tokens @ cls_tokens.T
    similarity_matrix = torch.tensor(similarity_matrix)
    scores_new = MMO(similarity_matrix.clone().float(), score=torch.tensor(scores).clone().float(), k_list=k_list)
    scores_new = scores_new.numpy()
    return scores_new

def Mobile_RsCIN(classification_results, k_list=[2, 3], dataset_name='mvtec', class_name='bottle', cls_tokens=None):
    if 'mvtec' in dataset_name.lower():
        cls_features_path = 'mvtec_ad_cls.dat'
        k_list = [2, 3]
    elif 'visa' in dataset_name.lower():
        cls_features_path = 'visa_cls.dat'
        k_list = [8, 9]
    if cls_tokens is None:
        with open(cls_features_path, 'rb') as f:
            cls_tokens = pickle.load(f)
        cls_tokens = cls_tokens[class_name]
    optimized_classification_results = RsCIN(classification_results, cls_tokens, k_list)
    return optimized_classification_results

if __name__ == "__main__":
    classification_results = np.random.rand(83)
    optimized_classification_results = Mobile_RsCIN(classification_results, dataset_name='mvtec_ad', class_name='bottle')
    # optimized_classification_results = Mobile_RsCIN(classification_results, k_list=k_list, cls_tokens=cls_tokens)