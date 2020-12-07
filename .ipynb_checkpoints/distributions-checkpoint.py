import torch


def discretized_logistic_logp(mean, logscale, x, binsize=1 / 256.0):
    scale = torch.exp(logscale)
    normalized_x = (torch.floor(x / binsize) * binsize - mean) / scale
    logp = torch.log(torch.sigmoid(normalized_x + binsize / scale) - torch.sigmoid(normalized_x) + 1e-7)
    return torch.sum(logp,-1)
