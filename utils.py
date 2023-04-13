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


def reconstruct_image(width: int, height: int, net: NeuralNet, device: torch.device):
    # Build coordinates matrix
    coordinates = []
    for i in range(width):
        for j in range(height):
            X = torch.tensor([i/width, j/height], dtype=torch.float, device=device)
            coordinates.append(X)

    # Get predicted pixel values
    coordinates = torch.cat(coordinates).view(-1, 2).to(device)
    y = net(coordinates)
    y = y * 256

    # torch.tensor to PIL image
    image_numpy = np.array(y.view(width, height).detach().cpu())
    img = Image.fromarray(image_numpy.astype(np.uint8), 'L')

    # Save image
    now = datetime.now() # current date and time
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    img.save("reconstructions/" + date_time + '.png')



def read_image(path: str) -> torch.Tensor:
    image = torchvision.io.read_image(path, mode=torchvision.io.ImageReadMode.GRAY)
    return image


def build_dataset_from_image(image: torch.Tensor, device: torch.device)-> Tuple[torch.Tensor, torch.Tensor]:
    C, W, H = image.shape
    image = torch.reshape(image, (W, H, C))

    coordinates = []
    rgb_values = []

    for i in range(W):
        for j in range(H):
            X = torch.tensor([i/W, j/H], dtype=torch.float)
            y = image[i, j] / 256.0
            coordinates.append(X)
            rgb_values.append(y)

    coordinates = torch.cat(coordinates).view(-1, 2).to(device)
    rgb_values = torch.cat(rgb_values).view(-1, C).to(device)
    return coordinates, rgb_values


def create_batches(coordinates: torch.Tensor, rgb_values: torch.Tensor, batch_size: int)-> List[BatchData]:
    no_of_pixels = coordinates.shape[0]
    no_of_batches = int(no_of_pixels / batch_size)

    # Generate random permutation
    indices = torch.randperm(no_of_pixels)
    
    # Shuffle the coordinates and the RGB values
    coordinates = coordinates[indices]
    rgb_values = rgb_values[indices]

    # Create batches
    coordinates = coordinates.view(-1, batch_size, 2)
    C = rgb_values.shape[-1]
    rgb_values = rgb_values.view(-1, batch_size, C)

    batches = []
    for i in range(no_of_batches):
        batch = BatchData(coordinates[i], rgb_values[i])
        batches.append(batch)

    return batches
