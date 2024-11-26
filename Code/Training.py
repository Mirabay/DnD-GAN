import torch
import torch.nn as nn
import SimpleGAN as gan
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import matplotlib.animation as animation
from IPython.display import HTML
import os

def GenerateDataSet(root, image_size, batch_size):
    # Create dataset
    dataset = dset.ImageFolder(root=root,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=True, 
                                         num_workers=4, 
                                         pin_memory=True,
                                         persistent_workers=True)
    return dataloader

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.2)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.2)
        nn.init.constant_(m.bias.data, 0)
        
        
def trainGAN(gen, dis, lrG, lrD, epochs, dataloader, device, nz):
    # Set device for training
    gen.to(device)
    dis.to(device)
    # Set optimizers
    optimizerG = optim.Adam(gen.parameters(), lr=lrG, betas=(0.5, 0.999))
    optimizerD = optim.Adam(dis.parameters(), lr=lrD, betas=(0.5, 0.999))
    
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.
    # Set loss function
    criterion = nn.BCELoss()
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    D_xs = []
    iters = 0

    print("Starting Training Loop...")
    # Initialize tqdm progress bar for the entire training process
    total_steps = epochs * len(dataloader)
    progress_bar = tqdm(total=total_steps, desc="Training Progress")

    plt.style.use('bmh')
    
    # Initialize real-time plot
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    ax1.set_title("Generator and Discriminator Loss During Training")
    ax1.set_xlabel("iterations")
    ax1.set_ylabel("Loss")
    ax2.set_title("Discriminator Output During Training")
    ax2.set_xlabel("iterations")
    ax2.set_ylabel("Output")
    line1, = ax1.plot(G_losses, label="G")
    line2, = ax1.plot(D_losses, label="D")
    line3, = ax2.plot(D_xs, label="D(x)")
    ax1.legend()
    ax2.legend()

    # For each epoch
    for epoch in range(epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            # Update the discriminator
            dis.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device, dtype=torch.float)
            output = dis(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = gen(noise)
            label.fill_(fake_label)
            output = dis(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # Update the generator
            gen.zero_grad()
            label.fill_(real_label)
            output = dis(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            D_xs.append(D_x)

            # Update tqdm progress bar
            progress_bar.set_postfix({
                'Loss_D': f'{errD.item():.3f}',
                'Loss_G': f'{errG.item():.3f}',
                'D(x)': f'{D_x:.3f}',
                'D(G(z))': f'{D_G_z1:.3f} / {D_G_z2:.3f}'
            })
            progress_bar.update(1)

            # Update real-time plot
            line1.set_ydata(G_losses)
            line1.set_xdata(range(len(G_losses)))
            line2.set_ydata(D_losses)
            line2.set_xdata(range(len(D_losses)))
            line3.set_ydata(D_xs)
            line3.set_xdata(range(len(D_xs)))
            ax1.relim()
            ax1.autoscale_view()
            ax2.relim()
            ax2.autoscale_view()
            plt.draw()
            plt.pause(0.01)

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = gen(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                torch.save(gen.state_dict(), f'Models/generator128_{iters}.pth')

            iters += 1

    # Close the progress bar
    progress_bar.close()
    plt.ioff()
    
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('Reporte/Historial de Entrenamiento/LossHistory.png')

    
    plt.figure(figsize=(10,5))
    plt.title("Discriminator Output During Training")
    plt.plot(D_xs,label="D(x)")
    plt.xlabel("iterations")
    plt.ylabel("Output")
    plt.legend()
    plt.savefig('Reporte/Historial de Entrenamiento/DiscriminatorOutput.png')
    
    plt.rcParams['animation.embed_limit'] = 2**128

    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    HTML(ani.to_jshtml())
    # Save the animation
    ani.save('Reporte/Historial de Entrenamiento/dcgan.gif', writer='pillow', fps=14)
    
    return gen, dis
    
    
if __name__ == '__main__':
    # Set device for training
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(device)
    
    # Load the dataset into a DataLoader
    root= 'Data'    
    dndDataLoader = GenerateDataSet(root=root, image_size=128, batch_size=128)

    # Parameters
    lrD = 2e-6#8e-5
    lrG = 2e-5#6e-4
    epochs = 250
    nz = 1000
    filtros = 128
    
    # Create the generator and discriminator
    generator = gan.Generator(nz, filtros, 3, 1).to(device)
    discriminator = gan.Discriminator(filtros, 3, 1).to(device)
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # Train the GAN
    generator, discriminator = trainGAN(generator, discriminator, lrG, lrD, epochs, dndDataLoader, device, nz)
   