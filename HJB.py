import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
from tqdm import tqdm


dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

T = 5.0 # the time horizom
M = 20000 # the number of training samples

torch.manual_seed(0)

x_data = torch.randn(M, 2).to(dev)*2
t_data = torch.rand(M, 1).to(dev)*T

# The initial value
def g_in(x):
    return x.square().sum(axis=1, keepdims=True)

# The corresponding solution
def u_sol(xt):
    x_in = xt[:,:2]
    t_in = xt[0,2].item()
    op = 1 + 4 * (T - t_in)
    return torch.add(torch.div(x_in.square().sum(axis=1, keepdims=True), op), np.log(op))

# We use a network with 4 hidden layers of 50 neurons each and the
# Swish activation function ( called SiLU in PyTorch )
N = torch.nn.Sequential (
    torch.nn.Linear(3, 50), torch.nn.SiLU(),
    torch.nn.Linear(50, 50), torch.nn.SiLU(),
    torch.nn.Linear(50, 50), torch.nn.SiLU(),
    torch.nn.Linear(50, 50), torch.nn.SiLU(),
    torch.nn.Linear(50, 1),
).to(dev)

optimizer = torch.optim.Adam(N.parameters(), lr=3e-4)

J = 256 # the batch size
iterations = 2000
lambda_0 = 1

for i in tqdm(range(iterations)):
    # Choose a random batch of training samples
    indices = torch.randint(0, M, (J,))
    x = x_data[indices, :]
    t = t_data[indices, :]

    x1, x2 = x[:, 0:1], x[:, 1:2]

    x1.requires_grad_()
    x2.requires_grad_()
    t.requires_grad_()

    optimizer.zero_grad ()

    # Denoting by u the realization function of the ANN, compute
    # u(0, x) for each x in the batch
    u0 = N(torch.hstack((torch.zeros_like(t), x)))
    # Compute the loss for the initial condition
    initial_loss = (u0 - g_in(x)).square().mean()

    # Compute the partial derivatives using automatic differentiation
    u = N(torch.hstack((t, x1, x2)))
    ones = torch.ones_like(u)
    u_t = grad(u, t, ones, create_graph = True)[0]
    u_x1 = grad (u, x1, ones, create_graph = True)[0]
    u_x2 = grad (u, x2, ones, create_graph = True)[0]
    ones = torch.ones_like(u_x1)
    u_x1x1 = grad(u_x1, x1, ones, create_graph = True)[0]
    u_x2x2 = grad(u_x2, x2, ones, create_graph = True)[0]

    # Compute the loss for the PDE
    Laplace = u_x1x1 + u_x2x2
    pde_loss = (-1 * u_t - Laplace + u_x1**2 + u_x2**2).square().mean()

    # Compute the total loss and perform a gradient step
    loss = pde_loss + lambda_0*initial_loss
    loss.backward()
    optimizer.step()


### Plot the solution at different times

mesh = 128
a, b = -2, 2

gs = GridSpec(2, 7, width_ratios=[1, 1, 1, 1, 1, 1, 0.05])
fig = plt.figure(figsize=(16, 10), dpi=300)

x, y = torch.meshgrid(
    torch.linspace(a, b, mesh),
    torch.linspace(a, b, mesh),
    indexing = "xy"
)

x = x.reshape((mesh * mesh, 1)).to(dev)
y = y.reshape((mesh * mesh, 1)).to(dev)

for i in range(6):
    t = torch.full((mesh * mesh, 1), i * T / 5).to(dev)
    z = N(torch.cat((t, x, y), 1))
    z = z.detach().cpu().numpy().reshape((mesh, mesh))

    ax = fig.add_subplot(gs[0, i])
    ax.set_title(f"PINN: t = {i * T / 5}")
    ax.imshow(
        z, cmap ="viridis", extent=[0, 2*b, 0, 2*b]#, vmin=-1.2, vmax=1.2
    )

for i in range(6):
    t = torch.full((mesh * mesh, 1), i * T / 5).to(dev)
    xt = torch.cat((x, y, t), 1)
    z = u_sol(xt)
    z = z.detach().cpu().numpy().reshape((mesh,mesh))

    ax = fig.add_subplot(gs[1, i])
    ax.set_title(f"Sol: t = {i * T / 5}")
    ax.imshow(
        z, cmap ="viridis", extent=[0, 2*b, 0, 2*b]#, vmin=-1.2, vmax=1.2
    )

# Add the colorbar to the figure
norm = plt.Normalize(vmin=0, vmax=2*b)
sm = ScalarMappable(cmap="viridis", norm=norm)
cax = fig.add_subplot(gs[:, 6])
fig.colorbar(sm, cax=cax, orientation ='vertical')

fig.savefig("HJBpinn.pdf", bbox_inches="tight")
