import torch
import torch.nn as nn
def randconv(image, K, mix, p, dim) -> torch.Tensor:
    """
    Outputs the image or the random convolution applied on the image.

    Args:
        image (torch.Tensor): input image
        K (int): maximum kernel size of the random convolution
    """

    p0 = torch.rand(1).item()
    if p0 < p:
        return image
    else:
        k = torch.randint(0, K+1, (1, )).item()
        if dim == '2D':
           random_convolution = nn.Conv2d(3, 3, 2*k + 1, padding=k)
        else:
           random_convolution = nn.Conv3d(3, 3, 2*k + 1, padding=k)
        torch.nn.init.uniform_(random_convolution.weight,
                              0, 1. / (3 * k * k))
        image_rc = random_convolution(image).to(image.device)

        if mix:
            alpha = torch.rand(1,)
            return alpha * image + (1 - alpha) * image_rc
        else:
            return image_rc