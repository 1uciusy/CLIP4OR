from itertools import product

import torch


def sim(seq1: torch.Tensor, seq2: torch.Tensor):
    # seq1 = torch.randn((100, 35))
    # seq2 = torch.randn((100, 35))
    # seq2 = seq1
    length = min(seq1.shape[0], seq2.shape[0])
    seq1 = seq1[:length, :]
    seq2 = seq2[:length, :]
    s1 = torch.sum(torch.abs(seq1), dim=0)
    s2 = torch.sum(torch.abs(seq2), dim=0)
    s = torch.sum(torch.abs(seq1 - seq2), dim=0)
    u = (s + s1 + s2 + 1e-9) / 2
    i = (s1 + s2 - s) / 2
    return torch.mean(i / u)


def batch_sim(seq1: torch.Tensor, seq2: torch.Tensor):
    # seq1 = torch.randn((2, 100, 35))
    # seq2 = torch.randn((2, 100, 35))
    # seq2 = seq1
    res = list()
    for i, v in enumerate(product(seq1, seq2)):
        res.append(torch.exp(-torch.mean(torch.abs(v[0] - v[1])) * 10))
    res = torch.tensor(res)
    b = seq1.shape[0]
    res = res.reshape(b, b)  # res[a][b] is the similarity of seq1[a] and seq2[b]
    return res


if __name__ == "__main__":
    a = torch.randn(2, 10, 3)
    sim(a[0], a[1])
    batch_sim(a, a)
