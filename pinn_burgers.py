"""
PINNs 入门示例：Burgers 方程求解
基于 Raissi et al. (2019) Physics-informed neural networks

方程: u_t + u*u_x = ν*u_xx
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = layers
        self.linears = nn.ModuleList()
        self.activation = nn.Tanh()
        
        for i in range(len(layers) - 1):
            self.linears.append(nn.Linear(layers[i], layers[i+1]))
    
    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        for i in range(len(self.linears) - 1):
            xt = self.activation(self.linears[i](xt))
        u = self.linears[-1](xt)
        return u

def get_pde_residual(model, x, t, nu):
    x.requires_grad = True
    t.requires_grad = True
    
    u = model(x, t)
    
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), 
                              create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                              create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0]
    
    residual = u_t + u * u_x - nu * u_xx
    return residual

def train_pinn(model, nu, epochs=10000, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    x_domain = torch.rand(1000, 1) * 2 - 1
    t_domain = torch.rand(1000, 1) * 1
    
    x_bc = torch.tensor([[-1.0], [1.0]]).repeat(20, 1)
    t_bc = torch.rand(40, 1)
    u_bc = torch.zeros(40, 1)
    
    x_ic = torch.rand(100, 1) * 2 - 1
    t_ic = torch.zeros(100, 1)
    u_ic = -torch.sin(np.pi * x_ic)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        residual = get_pde_residual(model, x_domain, t_domain, nu)
        loss_pde = torch.mean(residual ** 2)
        
        u_pred_bc = model(x_bc, t_bc)
        loss_bc = torch.mean((u_pred_bc - u_bc) ** 2)
        
        u_pred_ic = model(x_ic, t_ic)
        loss_ic = torch.mean((u_pred_ic - u_ic) ** 2)
        
        loss = loss_pde + 10 * loss_bc + 10 * loss_ic
        loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

def visualize_results(model):
    x = np.linspace(-1, 1, 50)
    t_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    
    for idx, t_val in enumerate(t_vals):
        x_grid, t_grid = np.meshgrid(x, [t_val])
        x_flat = torch.tensor(x_grid.flatten().reshape(-1, 1))
        t_flat = torch.tensor(t_grid.flatten().reshape(-1, 1))
        
        with torch.no_grad():
            u_pred = model(x_flat, t_flat).numpy().reshape(1, -1)
        
        axes[idx].plot(x, u_pred.flatten())
        axes[idx].set_title(f't = {t_val}')
        axes[idx].set_xlabel('x')
        axes[idx].set_ylabel('u')
        axes[idx].set_ylim([-1.5, 1.5])
    
    plt.tight_layout()
    plt.savefig('burgers_pinn_result.png', dpi=150)
    plt.show()
    print("结果已保存到 burgers_pinn_result.png")

if __name__ == "__main__":
    nu = 0.01
    
    model = PINN([2, 32, 32, 1])
    
    print("开始训练 PINNs...")
    train_pinn(model, nu, epochs=5000, lr=1e-3)
    
    print("\n可视化结果...")
    visualize_results(model)
