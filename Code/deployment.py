# Import the GAN model
import SimpleGAN as gan
import torch
# Genera una imagen
import matplotlib.pyplot as plt
import numpy as np

def RandomImage(generator, device):
    # Generate a random noise tensor
    noise = torch.randn(1, 100, 1, 1, device=device)

    # Generate an image
    with torch.no_grad():
        fake = generator(noise).detach().cpu()

    # Display the image
    plt.imshow(np.transpose(fake[0], (1, 2, 0)))
    plt.axis('off')
    plt.show()


# Call the function

if __name__ == '__main__':
    
    root='Models/generator128_4999.pth'

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        
    # Load the model
    generator = gan.MidGenerator(100, 64, 3, 1)
    generator.load_state_dict(torch.load(root, map_location=device))
    generator.to(device)
    
    # Generate a random image
    RandomImage(generator, device)
