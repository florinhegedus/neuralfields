import sys
from utils import get_device, read_image, build_dataset_from_image, create_batches, reconstruct_image, set_seed
from nn import NeuralNet
import torch
import torch.nn as nn


def main_loop(image_path):
    set_seed()
    device = get_device()
    image = read_image(image_path)
    coordinates, rgb_values = build_dataset_from_image(image, device)

    net = NeuralNet(hidden_layer_size=256, num_hidden_layers=3)
    net = net.to(device)

    # Optimizers specified in the torch.optim package
    optimizer = torch.optim.SGD(net.parameters(), lr=0.005, momentum=0.9) 

    # Loss function
    l1_loss = nn.L1Loss()

    batch_size, epochs = 1024, 100
    for epoch in range(epochs):
        batches = create_batches(coordinates, rgb_values, batch_size)
        epoch_loss = 0.0
        for batch in batches:
            inputs = batch.coordinates
            targets = batch.rgb_values

            preds = net(inputs)

            optimizer.zero_grad()
            loss = l1_loss(preds, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch}: loss={epoch_loss}")

    reconstruct_image(512, 512, net, device)
                    
            

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError("Please provide the path to the image")
    image_path = sys.argv[1]
    main_loop(image_path)
