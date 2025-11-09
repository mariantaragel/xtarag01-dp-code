import torch

try:
    from .PyTorchEMD.emd import earth_mover_distance
except:
    raise


def emd_loss(p1, p2, transpose=False):
    return earth_mover_distance(p1, p2, transpose)


if __name__ == "__main__":
    pca = torch.rand(10, 2048, 3).cuda()
    pcb = torch.rand(10, 4096, 3).cuda()
    d = emd_loss(pca, pcb, transpose=False)
    print(d.shape)
    print(d)
