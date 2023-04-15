import torch
import torchvision
from typing import List, Tuple
from dataclasses import dataclass
from nn import NeuralNet
import numpy as np
from PIL import Image
from datetime import datetime


@dataclass
class BatchData:
    coordinates: torch.tensor
    rgb_values: torch.tensor


def set_seed():
    torch.manual_seed(0)


def get_device(verbose: bool=True) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print("--"*30)
        if device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('\tAllocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('\tReserved: ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
        else:
            print('Using CPU')
        print("--"*30)

    return device


def reconstruct_image(size: int, net: NeuralNet, device: torch.device, mode: str):
    # Build coordinates matrix
    coordinates = get_mgrid(size, 2, device)

    # Get predictions
    y = net(coordinates)
    y = y.view(-1, size, size)

    # torch.tensor to PIL image
    img = torchvision.transforms.functional.to_pil_image(y, mode)

    # Save image
    now = datetime.now() # current date and time
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    img.save("reconstructions/" + date_time + '.png')



def read_image(path: str, mode: str) -> torch.Tensor:
    if mode == 'RGB':
        image = torchvision.io.read_image(path, mode=torchvision.io.ImageReadMode.RGB)
    else:
        image = torchvision.io.read_image(path, mode=torchvision.io.ImageReadMode.GRAY)
    return image


def get_mgrid(sidelen, dim, device):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    mgrid = mgrid.to(device)
    return mgrid


def build_dataset_from_image(image: torch.Tensor, device: torch.device)-> Tuple[torch.Tensor, torch.Tensor]:
    # Get pixel values
    C, W, H = image.shape
    image = torch.reshape(image, (W, H, C))
    image = image / 255
    pixel_values = image.view(-1, C)

    # Get coordinates
    assert W == H, "Width and height of the input image must be the same."
    coordinates = get_mgrid(W, 2, device)

    pixel_values = pixel_values.to(device)
    
    return coordinates, pixel_values


def create_batches(coordinates: torch.Tensor, pixel_values: torch.Tensor, batch_size: int)-> List[BatchData]:
    no_of_pixels = coordinates.shape[0]
    no_of_batches = int(no_of_pixels / batch_size)

    # Generate random permutation
    indices = torch.randperm(no_of_pixels)
    
    # Shuffle the coordinates and the RGB values
    coordinates = coordinates[indices]
    pixel_values = pixel_values[indices]

    # Create batches
    coordinates = coordinates.view(-1, batch_size, 2)
    C = pixel_values.shape[-1]
    pixel_values = pixel_values.view(-1, batch_size, C)

    batches = []
    for i in range(no_of_batches):
        batch = BatchData(coordinates[i], pixel_values[i])
        batches.append(batch)

    return batches
