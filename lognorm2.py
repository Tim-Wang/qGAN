# best seed so far 1765891414

import time
import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import multivariate_normal, entropy
from scipy.stats import lognorm
from qiskit_machine_learning.utils import algorithm_globals
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal, EfficientSU2
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_finance.circuit.library import NormalDistribution
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from IPython.display import clear_output
from tqdm import tqdm

EPS = 1e-8

num_dim = 1
num_discrete_values = 8
# use log(n) qubits to express n discrete values, log(num_discrete_values^num_dim)=num_dim*log(num_discrete_values)
num_qubits = num_dim * int(np.log2(num_discrete_values))
num_qnn_outputs = 2 ** num_qubits
batch_size = 2000

# real_seed = int(time.time())
real_seed = 1765891414
print(f"Real random seed: {real_seed}")

algorithm_globals.random_seed = real_seed
torch.manual_seed(real_seed)
np.random.seed(real_seed)

generator_lr = 5e-4
discriminator_lr = 1e-3
b1 = 0.7
b2 = 0.99
# wd = 0.005
num_epoches = 200
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
print(f"Using device: {device}")

log_dir = "./logs/lognorm_neo/lognorm_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir=log_dir)

mu = 1
sigma = 1
target_count = 20000
current_count = 0
collected_samples = []

rv = lognorm(s=sigma, scale=np.exp(mu))
while current_count < target_count:
    sample = rv.rvs()
    if sample <= 7:
        collected_samples.append(sample)
        current_count += 1

collected_samples = np.round(collected_samples).astype(int)
counts = np.bincount(collected_samples)
possible_values = np.arange(num_discrete_values)
prob_data = counts / np.sum(counts)

qc = QuantumCircuit(num_qubits)
qc.h(qc.qubits)
# qc = NormalDistribution(num_qubits, mu=mu, sigma=sigma, bounds=(0, 7))
ansatz = TwoLocal(num_qubits, "ry", "cz", entanglement="circular", reps=1)
# ansatz = EfficientSU2(num_qubits, reps=2, entanglement="circular")
qc.compose(ansatz, inplace=True)

fig_circuit = qc.decompose().draw("mpl", scale=0.7, fold=100)
fig_circuit.tight_layout()
plt.show()

# print(qc.decompose().draw('latex_source'))

sampler = StatevectorSampler()
def create_generator() -> TorchConnector:
    qnn = SamplerQNN(
        circuit=qc,
        sampler=sampler,
        input_params=[],
        weight_params=qc.parameters,
        sparse=False
    )
    initial_weights = np.random.uniform(-np.pi, np.pi, qc.num_parameters)
    return TorchConnector(qnn, initial_weights)

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 8),
            nn.LeakyReLU(0.2),
            nn.Linear(8, 8),
            nn.LeakyReLU(0.2),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)
    
generator = create_generator()
discriminator = Discriminator(num_dim).to(device)

real_data_tensor = torch.tensor(collected_samples, dtype=torch.float32).reshape(-1, 1)
dataset = TensorDataset(real_data_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

generator_optimizer = Adam(generator.parameters(), lr=generator_lr, betas=(b1, b2), amsgrad=True)
discriminator_optimizer = Adam(discriminator.parameters(), lr=discriminator_lr, betas=(b1, b2), amsgrad=True)

def plot_output():
    with torch.no_grad():
        generated_probabilities = generator().numpy()
        plt.figure(figsize=(10, 6))
        plt.bar(possible_values, generated_probabilities)
        plt.plot(possible_values, prob_data, color="red", marker="o", linestyle="-")
        plt.grid()
        plt.show()

grid_elements = torch.arange(num_discrete_values, dtype=torch.float32).reshape(-1, 1).to(device)
prob_data_tensor = torch.tensor(prob_data, dtype=torch.float).reshape(-1, 1).to(device)
samples = torch.tensor(grid_elements, dtype=torch.float).to(device)

start = time.time()
for epoch in tqdm(range(num_epoches)):
    epoch_g_loss = 0
    epoch_d_loss = 0
    for (i, (real_batch, )) in enumerate(dataloader):
        real_batch = real_batch.to(device)
        current_batch_size = real_batch.size(0)

        disc_on_grid = discriminator(samples)
        gen_probs_real = generator(torch.tensor([]))
        gen_probs = gen_probs_real.reshape(-1, 1)

        generator_optimizer.zero_grad()
        g_loss = -torch.sum(gen_probs * torch.log(disc_on_grid + EPS))
        g_loss.backward(retain_graph=True)
        generator_optimizer.step()

        discriminator_optimizer.zero_grad()
        real_validity = discriminator(real_batch)
        real_loss = -torch.mean(torch.log(real_validity + EPS))
        fake_samples = torch.multinomial(gen_probs_real, current_batch_size, replacement=True).reshape(-1, 1).float().to(device)
        fake_validity = discriminator(fake_samples)
        fake_loss = -torch.mean(torch.log(1 - fake_validity + EPS))
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        discriminator_optimizer.step()

        epoch_g_loss += g_loss.item()
        epoch_d_loss += d_loss.item()
    
    with torch.no_grad():
        cur_probs = generator(torch.tensor([])).detach().cpu().numpy().flatten()
        ent = entropy(cur_probs, prob_data + EPS)
    
    avg_g_loss = epoch_g_loss / len(dataloader)
    avg_d_loss = epoch_d_loss / len(dataloader)
    writer.add_scalar("Loss/Generator", avg_g_loss, epoch)
    writer.add_scalar("Loss/Discriminator", avg_d_loss, epoch)
    writer.add_scalar("Metrics/Relative Entropy", ent, epoch)

    if epoch % 100 == 0:
        plot_output()

elapsed = time.time() - start
print(f"Training finished in {elapsed:.2f} seconds.")
writer.close()

with torch.no_grad():
    plot_output()
    for param in generator.parameters():
        print(param.data.numpy())