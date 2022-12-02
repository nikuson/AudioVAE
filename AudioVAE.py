import os
import numpy as np
from scipy.io import wavfile
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set the input and output directories
input_dir = 'input'
output_dir = 'output'

# Create a list of all the WAV files in the input directory
wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]

class VAE(nn.Module):
  def __init__(self, data_shape):
    super(VAE, self).__init__()
    self.data_shape = data_shape

    # Define the encoder part of the VAE
    self.fc1 = nn.Linear(data_shape[1], 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 32)

    # Define the latent distribution parameters
    self.fc4_mu = nn.Linear(32, 16)
    self.fc4_var = nn.Linear(32, 16)

    # Define the decoder part of the VAE
    self.fc5 = nn.Linear(16, 32)
    self.fc6 = nn.Linear(32, 64)
    self.fc7 = nn.Linear(64, 128)
    self.fc8 = nn.Linear(128, data_shape[1])

  def encode(self, x):
    # Apply the encoder layers and return the latent parameters
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    mu = self.fc4_mu(x)
    var = self.fc4_var(x)
    return mu, var

  def reparameterize(self, mu, var):
    # Sample from the latent distribution
    std = torch.exp(0.5*var)
    eps = torch.randn_like(std)
    z = eps.mul(std).add_(mu)
    return z

  def decode(self, z):
    # Apply the decoder layers and return the reconstructed data
    z = F.relu(self.fc5(z))
    z = F.relu(self.fc6(z))
    z = F.relu(self.fc7(z))
    z = torch.sigmoid(self.fc8(z))
    return z

  def forward(self, x):
    # Encode the data and sample from the latent distribution
    mu, var = self.encode(x)
    z = self.reparameterize(mu, var)

    # Decode the latent sample and return the reconstructed data
    recon_x = self.decode(z)
    return recon_x, mu, var

  def loss(self, data, recon_x, mu, var):
    # Compute the reconstruction loss
    recon_loss = F.binary_cross_entropy(recon_x, data, reduction='sum')

    # Compute the KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())

    # Return the total loss
    return recon_loss + kl_loss

# Loop over each WAV file
for wav_file in wav_files:
  # Read the WAV file and convert it to a PyTorch tensor
  rate, data = wavfile.read(os.path.join(input_dir, wav_file))
  data = data.astype('float32') / np.iinfo(data.dtype).max
  data = torch.from_numpy(data).float()

  # Create the VAE model and define the optimizer
  vae = VAE(data.shape)
  optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

  # Train the VAE
  for epoch in range(30):
    # Compute the loss and update the model
    recon_x, mu, var = vae(data)
    loss = vae.loss(data, recon_x, mu, var)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch+1}/30: Loss = {loss.item():.4f}')

  # Save the encoded and decoded audio to the output directory

  encoded_audio = vae.encode(data)[0]
  encoded_audio = encoded_audio.detach().numpy()
  wavfile.write(os.path.join(output_dir, wav_file), rate, encoded_audio)
