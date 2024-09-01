import torch
import torch.nn.functional as F

def soft_histogram(x, bins, min=0.0, max=1.0, sigma=0.01):
    # Create bin centers
    bin_centers = torch.linspace(min, max, steps=bins, device=x.device)

    # Adjust shapes for broadcasting
    x = x.unsqueeze(2)  # Now shape [N, 2000, 1]
    bin_centers = bin_centers.unsqueeze(0).unsqueeze(0)  # Now shape [1, 1, 100]

    # Calculate differences and apply Gaussian kernel
    differences = x - bin_centers  # Broadcasting to shape [N, 2000, 100]
    gauss_kernel = torch.exp(-0.5 * (differences / sigma) ** 2)

    # Sum contributions to each bin for each sample
    hist = gauss_kernel.sum(dim=1)  # Now shape [N, 100]

    # Normalize the histogram
    hist /= hist.sum(dim=1, keepdim=True)

    return hist



#######################################################
#       STATISTICAL DISTANCES(LOSSES) IN PYTORCH      #
#######################################################

## Statistial Distances for 1D weight distributions
## Inspired by Scipy.Stats Statistial Distances for 1D
## Pytorch Version, supporting Autograd to make a valid Loss
## Supposing Inputs are Groups of Same-Length Weight Vectors
## Instead of (Points, Weight), full-length Weight Vectors are taken as Inputs
## Code Written by E.Bao, CASIA



def torch_wasserstein_loss(tensor_a,tensor_b):
    histogram_a = soft_histogram(tensor_a, bins=10, sigma=0.05)
    histogram_b = soft_histogram(tensor_b, bins=10, sigma=0.05)
    return(torch_cdf_loss(histogram_a,histogram_b,p=1))
    # Compute the first Wasserstein distance between two 1D distributions.
    # return(torch_cdf_loss(tensor_a,tensor_b,p=1))

def torch_energy_loss(tensor_a,tensor_b):
    # Compute the energy distance between two 1D distributions.
    return((2**0.5)*torch_cdf_loss(tensor_a,tensor_b,p=2))

def torch_cdf_loss(tensor_a,tensor_b,p=1):
    # last-dimension is weight distribution
    # p is the norm of the distance, p=1 --> First Wasserstein Distance
    # to get a positive weight with our normalized distribution
    # we recommend combining this loss with other difference-based losses like L1

    # normalize distribution, add 1e-14 to divisor to avoid 0/0
    tensor_a = tensor_a / (torch.sum(tensor_a, dim=-1, keepdim=True) + 1e-14)
    tensor_b = tensor_b / (torch.sum(tensor_b, dim=-1, keepdim=True) + 1e-14)
    # make cdf with cumsum
    cdf_tensor_a = torch.cumsum(tensor_a,dim=-1)
    cdf_tensor_b = torch.cumsum(tensor_b,dim=-1)

    # choose different formulas for different norm situations
    if p == 1:
        cdf_distance = torch.sum(torch.abs((cdf_tensor_a-cdf_tensor_b)),dim=-1)
    elif p == 2:
        cdf_distance = torch.sqrt(torch.sum(torch.pow((cdf_tensor_a-cdf_tensor_b),2),dim=-1))
    else:
        cdf_distance = torch.pow(torch.sum(torch.pow(torch.abs(cdf_tensor_a-cdf_tensor_b),p),dim=-1),1/p)

    cdf_loss = cdf_distance.mean()
    return cdf_loss

def torch_validate_distibution(tensor_a,tensor_b):
    # Zero sized dimension is not supported by pytorch, we suppose there is no empty inputs
    # Weights should be non-negetive, and with a positive and finite sum
    # We suppose all conditions will be corrected by network training
    # We only check the match of the size here
    if tensor_a.size() != tensor_b.size():
        raise ValueError("Input weight tensors must be of the same size")
    