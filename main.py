import sys
from utils import get_device, read_image, \
                    build_dataset_from_image, \
                    train, \
                    reconstruct_image, \
                    set_seed, \
                    positional_encode
from nn import NeuralNet
import yaml


def main_loop(config):
    # Read data from config
    image_path = config['image']
    mode = config['mode']
    batch_size = config['batch_size']
    epochs = config['num_epochs']
    num_hidden_layers = config['network']['num_hidden_layers']
    hidden_layer_size = config['network']['hidden_layer_size']
    reconstruction_resolution = config['reconstruction_resolution']
    positional_encoding = config['positional_encoding']
    num_frequencies = config['num_frequencies']

    # Build dataset
    set_seed()
    device = get_device()
    image = read_image(image_path, mode)
    coordinates, pixel_values = build_dataset_from_image(image, device)

    if positional_encoding:
        coordinates = positional_encode(coordinates, num_frequencies)

    # Neural net corresponding to image format
    input_channels = coordinates.shape[-1]
    output_channels = 3 if mode == "RGB" else 1
    net = NeuralNet(input_channels=input_channels,
                    hidden_layer_size=hidden_layer_size, 
                    num_hidden_layers=num_hidden_layers, 
                    output_channels=output_channels)
    net = net.to(device)

    train(net, coordinates, pixel_values, epochs, batch_size)

    reconstruct_image(reconstruction_resolution, net, mode, positional_encoding, num_frequencies)
                    

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError("Please provide the path to the config file")
    
    config = yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)

    main_loop(config)
