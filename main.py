import sys
from utils import get_device, read_image, \
                    build_dataset_from_image, \
                    create_batches, \
                    reconstruct_image, \
                    set_seed
from nn import NeuralNet
import torch
import torch.nn as nn
import time


def main_loop(image_path, mode):
    set_seed()
    device = get_device()
    image = read_image(image_path, mode)
    coordinates, pixel_values = build_dataset_from_image(image, device)

    output_channels = 3 if mode == "RGB" else 1
    net = NeuralNet(hidden_layer_size=256, 
                    num_hidden_layers=3, 
                    output_channels=output_channels)
    net = net.to(device)

    # Optimizers specified in the torch.optim package
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9) 

    # Loss function
    l1_loss = nn.L1Loss()

    start = time.time()
    batch_size, epochs = 1024, 100
    for epoch in range(epochs):
        batches = create_batches(coordinates, pixel_values, batch_size)
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
    
    end = time.time()
    print(f"Training took: {end - start}s.")

    reconstruct_image(512, net, device, mode)
                    

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError("Please provide the path to the image and the mode")
    image_path = sys.argv[1]
    mode = sys.argv[2]
    main_loop(image_path, mode)
