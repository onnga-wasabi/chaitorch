import numpy as np
from scipy.spatial import distance


def get_ranking(data):
    dist_M = distance.cdist(data, data)
    return np.argsort(dist_M, axis=1)[:, 1:]


def recall_at_rank_k(embeddings, labels, K=1):
    ranking = get_ranking(embeddings)[:, :K]
    acc = []
    for rank, label in zip(ranking, labels):
        acc.append(len([neighbour for neighbour in labels[rank] if neighbour == label]) / K)
    return sum(acc) / len(labels)
