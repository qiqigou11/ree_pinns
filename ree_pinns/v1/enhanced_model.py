"""
增强版稀土元素（REE）PINNs模型

变量说明：
    t: 时间
    z: 深度
    C_LREE: 轻稀土浓度
    C_HREE: 重稀土浓度
    D: 扩散系数
    u: 对流速度
    k_ads: 吸附系数

环境变量：pH, 黏土, 坡度, 温度, 降水, 径流
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d


class ParameterModulator(nn.Module):
    """环境变量 → 物理参数调制因子"""

    def __init__(self, n_features=7, hidden_dim=32):
        super().__init__()
        # 输入: n_features个环境变量 → 隐藏层 → 4个调制因子
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 4),  # D, u, k_ads, S 的调制因子
            nn.Sigmoid()  # 输出 [0, 1]
        )

    def forward(self, env_features):
        factors = self.net(env_features)
        # 4个调制因子
        D_factor = factors[:, 0:1]      # 扩散系数调制
        u_factor = factors[:, 1:2]      # 对流速度调制
        k_factor = factors[:, 2:3]      # 吸附系数调制
        S_factor = factors[:, 3:4]       # 源汇项调制
        return D_factor, u_factor, k_factor, S_factor


class EnhancedREE_PINN(nn.Module):
    """PINN神经网络：输入(t,z,环境变量) → 输出(C_LREE, C_HREE)"""

    def __init__(self, input_dim=9, hidden_dims=[64, 64, 64, 64], output_dim=2,
                 use_modulator=True, n_env_features=7):
        super().__init__()
        self.use_modulator = use_modulator

        # 构建隐藏层
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.Tanh())
            prev_dim = dim
        self.hidden_layers = nn.Sequential(*layers)

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.Tanh(),
            nn.Linear(32, output_dim)
        )

        # 参数调制器
        if use_modulator:
            self.modulator = ParameterModulator(n_features=n_env_features)

    def forward(self, t, z, env_features=None):
        # 合并输入
        x = torch.cat([t, z], dim=1)  # [时间, 深度]
        if env_features is not None:
            x = torch.cat([x, env_features], dim=1)

        x = self.hidden_layers(x)
        output = self.output_layer(x)

        # 输出浓度（非负）
        C_LREE = torch.nn.functional.softplus(output[:, 0:1])  # 轻稀土
        C_HREE = torch.nn.functional.softplus(output[:, 1:2])  # 重稀土

        if self.use_modulator and env_features is not None:
            factors = self.modulator(env_features)
            return C_LREE, C_HREE, *factors

        return C_LREE, C_HREE

    def predict(self, t, z, env_features=None):
        """推理接口"""
        self.eval()
        with torch.no_grad():
            t_tensor = torch.tensor(t, dtype=torch.float32) if isinstance(t, np.ndarray) else t
            z_tensor = torch.tensor(z, dtype=torch.float32) if isinstance(z, np.ndarray) else z

            if t_tensor.dim() == 1:
                t_tensor = t_tensor.reshape(-1, 1)
            if z_tensor.dim() == 1:
                z_tensor = z_tensor.reshape(-1, 1)

            if env_features is not None:
                env_tensor = torch.tensor(env_features, dtype=torch.float32) if isinstance(env_features, np.ndarray) else env_features
            else:
                env_tensor = None

            result = self.forward(t_tensor, z_tensor, env_tensor)
            C_LREE = result[0]
            C_HREE = result[1]

        return C_LREE.numpy(), C_HREE.numpy()


class EnhancedREEPhysics:
    """物理方程：对流-扩散-吸附方程"""

    def __init__(self, base_params):
        self.D = base_params.get('D', 0.01)         # 扩散系数
        self.u = base_params.get('u', 1e-4)          # 对流速度
        self.k_ads_LREE = base_params.get('k_ads_LREE', 0.1)  # LREE吸附系数
        self.k_ads_HREE = base_params.get('k_ads_HREE', 0.5)  # HREE吸附系数
        self.S_LREE = base_params.get('S_LREE', 0.0)   # LREE源汇项
        self.S_HREE = base_params.get('S_HREE', 0.0)   # HREE源汇项
        self.C_atm_LREE = base_params.get('C_atm_LREE', 0.1)  # 顶部边界浓度
        self.C_atm_HREE = base_params.get('C_atm_HREE', 0.05)

    def compute_physics_residuals(self, pinn, t, z, env_features=None):
        """计算PDE残差"""
        t_input = t.clone().detach().requires_grad_(True)
        z_input = z.clone().detach().requires_grad_(True)
        env_input = env_features.clone().detach().requires_grad_(True) if env_features is not None else None

        # 网络预测
        result = pinn(t_input, z_input, env_input)
        if len(result) > 2:
            C_LREE, C_HREE, D_f, u_f, k_f, S_f = result
        else:
            C_LREE, C_HREE = result
            D_f = u_f = k_f = S_f = torch.ones_like(C_LREE)

        # 计算导数
        dC_LREE_dt = torch.autograd.grad(C_LREE, t_input, torch.ones_like(C_LREE), create_graph=True)[0]
        dC_LREE_dz = torch.autograd.grad(C_LREE, z_input, torch.ones_like(C_LREE), create_graph=True)[0]
        dC_HREE_dt = torch.autograd.grad(C_HREE, t_input, torch.ones_like(C_HREE), create_graph=True)[0]
        dC_HREE_dz = torch.autograd.grad(C_HREE, z_input, torch.ones_like(C_HREE), create_graph=True)[0]

        d2C_LREE_dz2 = torch.autograd.grad(dC_LREE_dz, z_input, torch.ones_like(dC_LREE_dz), create_graph=True)[0]
        d2C_HREE_dz2 = torch.autograd.grad(dC_HREE_dz, z_input, torch.ones_like(dC_HREE_dz), create_graph=True)[0]

        # 实际物理参数 = 基础值 × 调制因子
        D_actual = self.D * D_f
        u_actual = self.u * u_f
        k_LREE = self.k_ads_LREE * k_f
        k_HREE = self.k_ads_HREE * k_f
        S_L = self.S_LREE * S_f
        S_H = self.S_HREE * S_f

        # PDE: ∂C/∂t = D·∂²C/∂z² - u·∂C/∂z - k·C + S
        residual_LREE = dC_LREE_dt - D_actual * d2C_LREE_dz2 + u_actual * dC_LREE_dz + k_LREE * C_LREE - S_L
        residual_HREE = dC_HREE_dt - D_actual * d2C_HREE_dz2 + u_actual * dC_HREE_dz + k_HREE * C_HREE - S_H

        return residual_LREE, residual_HREE


class EnvDataInterpolator:
    """环境变量插值器：根据深度z获取环境变量"""

    def __init__(self, env_data):
        self.interpolators = {}
        for key, data in env_data.items():
            self.interpolators[key] = interp1d(
                data['z'], data['values'],
                kind='linear', bounds_error=False, fill_value='extrapolate'
            )
        self.feature_names = list(env_data.keys())

    def interpolate(self, z_values):
        """根据深度插值环境变量"""
        features = []
        for name in self.feature_names:
            features.append(self.interpolators[name](z_values))
        return np.stack(features, axis=1)

    def sample(self, z_values, noise_std=0.01):
        """采样 + 噪声"""
        return self.interpolate(z_values) + np.random.randn(*self.interpolate(z_values).shape) * noise_std


class EnhancedPINNTrainer:
    """PINNs训练器"""

    def __init__(self, pinn, physics, lambda_dict, obs_data=None, env_interpolator=None):
        self.pinn = pinn
        self.physics = physics
        self.lambda_dict = lambda_dict
        self.obs_data = obs_data
        self.env_interpolator = env_interpolator

        self.optimizer = optim.Adam(pinn.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=2000, factor=0.5)
        self.history = {'total': [], 'physics': [], 'obs': [], 'bc': []}

    def generate_points(self, n_points, T, h, env_sampler=None, is_boundary=False, is_bottom=False):
        """生成采样点"""
        t = torch.rand(n_points, 1) * T
        if is_boundary:
            z = torch.zeros(n_points, 1) if is_bottom else torch.ones(n_points, 1) * h
        else:
            z = torch.rand(n_points, 1) * h

        env_features = None
        if env_sampler is not None and self.env_interpolator is not None:
            env_features = torch.tensor(env_sampler.sample(z.numpy().flatten()), dtype=torch.float32)

        return t, z, env_features

    def train_step(self, n_physics, n_bc, T, h, env_sampler=None):
        """单步训练"""
        self.optimizer.zero_grad()

        # 1. 物理方程采样点
        t_pc, z_pc, env_pc = self.generate_points(n_physics, T, h, env_sampler)

        # 2. 边界采样点
        t_top, z_top, env_top = self.generate_points(n_bc, T, h, env_sampler, is_boundary=True)
        t_bottom, z_bottom, env_bottom = self.generate_points(n_bc, T, h, env_sampler, is_boundary=True, is_bottom=True)

        # 3. 物理损失
        residual_L, residual_H = self.physics.compute_physics_residuals(self.pinn, t_pc, z_pc, env_pc)
        loss_physics = torch.mean(residual_L**2) + torch.mean(residual_H**2)

        # 4. 实测数据损失
        if self.obs_data is not None:
            t_obs = self.obs_data['t']
            z_obs = self.obs_data['z']
            env_obs = None
            if self.env_interpolator is not None:
                env_obs = torch.tensor(self.env_interpolator.interpolate(z_obs.numpy().flatten()), dtype=torch.float32)

            result = self.pinn(t_obs, z_obs, env_obs)
            C_L_pred, C_H_pred = result[0], result[1]
            loss_obs = torch.mean((C_L_pred - self.obs_data['C_LREE'])**2) + \
                       torch.mean((C_H_pred - self.obs_data['C_HREE'])**2)
        else:
            loss_obs = torch.tensor(0.0, requires_grad=True)

        # 5. 边界损失
        result_top = self.pinn(t_top, z_top, env_top)
        C_L_top, C_H_top = result_top[0], result_top[1]
        loss_bc_top = torch.mean((C_L_top - self.physics.C_atm_LREE)**2) + \
                     torch.mean((C_H_top - self.physics.C_atm_HREE)**2)

        z_bottom.requires_grad_(True)
        result_bottom = self.pinn(t_bottom, z_bottom, env_bottom)
        C_L_bottom, C_H_bottom = result_bottom[0], result_bottom[1]
        dC_L_dz = torch.autograd.grad(C_L_bottom, z_bottom, torch.ones_like(C_L_bottom), create_graph=True)[0]
        dC_H_dz = torch.autograd.grad(C_H_bottom, z_bottom, torch.ones_like(C_H_bottom), create_graph=True)[0]
        loss_bc_bottom = torch.mean(dC_L_dz**2) + torch.mean(dC_H_dz**2)
        loss_bc = loss_bc_top + loss_bc_bottom

        # 6. 总损失
        loss_total = (self.lambda_dict['physics'] * loss_physics +
                      self.lambda_dict['obs'] * loss_obs +
                      self.lambda_dict['bc'] * loss_bc)

        loss_total.backward()
        self.optimizer.step()
        self.scheduler.step(loss_total)

        self.history['total'].append(loss_total.item())
        self.history['physics'].append(loss_physics.item())
        self.history['obs'].append(loss_obs.item() if isinstance(loss_obs, torch.Tensor) else 0)
        self.history['bc'].append(loss_bc.item())

        return {'total': loss_total.item(), 'physics': loss_physics.item(), 'obs': loss_obs.item(), 'bc': loss_bc.item()}

    def train(self, n_iterations=50000, n_physics=1000, n_bc=100, T=10000, h=1.0, env_sampler=None, print_every=1000):
        """训练循环"""
        self.pinn.train()
        for i in range(n_iterations):
            losses = self.train_step(n_physics, n_bc, T, h, env_sampler)
            if (i + 1) % print_every == 0:
                print(f"Iter {i+1}/{n_iterations} - Total: {losses['total']:.6f} | "
                      f"Physics: {losses['physics']:.6f} | Obs: {losses['obs']:.6f}")


class EnhancedREESimulator:
    """整合所有组件的模拟器"""

    def __init__(self, params=None, env_data=None, obs_data=None):
        self.params = {
            'D': 0.01, 'u': 1e-4, 'k_ads_LREE': 0.1, 'k_ads_HREE': 0.5,
            'S_LREE': 0.0, 'S_HREE': 0.0, 'C_atm_LREE': 0.1, 'C_atm_HREE': 0.05,
            'T': 10000, 'h': 1.0,
        }
        if params:
            self.params.update(params)

        self.env_data = env_data
        self.obs_data = obs_data

        # 创建组件
        physics_params = {k: v for k, v in self.params.items() if k not in ['T', 'h']}
        self.physics = EnhancedREEPhysics(physics_params)

        input_dim = 2 + len(env_data) if env_data else 2
        self.pinn = EnhancedREE_PINN(
            input_dim=input_dim, use_modulator=(env_data is not None),
            n_env_features=len(env_data) if env_data else 0
        )

        self.env_interpolator = EnvDataInterpolator(env_data) if env_data else None

        # 处理实测数据
        if obs_data:
            obs_tensor = {
                't': torch.tensor(obs_data['t'], dtype=torch.float32).reshape(-1, 1),
                'z': torch.tensor(obs_data['z'], dtype=torch.float32).reshape(-1, 1),
                'C_LREE': torch.tensor(obs_data['C_LREE'], dtype=torch.float32).reshape(-1, 1),
                'C_HREE': torch.tensor(obs_data['C_HREE'], dtype=torch.float32).reshape(-1, 1),
            }
        else:
            obs_tensor = None

        # 损失权重
        self.lambda_dict = {
            'physics': 1.0,
            'obs': 10.0 if obs_data else 0.0,
            'bc': 1.0
        }

        self.trainer = EnhancedPINNTrainer(
            self.pinn, self.physics, self.lambda_dict, obs_tensor, self.env_interpolator
        )

    def train(self, n_iterations=50000, n_physics=1000, n_bc=100, print_every=1000):
        self.trainer.train(n_iterations, n_physics, n_bc, self.params['T'], self.params['h'],
                          self.env_interpolator, print_every)

    def predict(self, t, z):
        self.pinn.eval()
        if isinstance(t, (int, float)):
            t = np.array([t])
        if isinstance(z, (int, float)):
            z = np.array([z])

        env_features = None
        if self.env_interpolator is not None:
            env_features = self.env_interpolator.interpolate(z)

        return self.pinn.predict(
            torch.tensor(t, dtype=torch.float32).reshape(-1, 1),
            torch.tensor(z, dtype=torch.float32).reshape(-1, 1),
            torch.tensor(env_features, dtype=torch.float32) if env_features is not None else None
        )

    def plot_results(self, save_path='/home/gousuheng/pinns/demo/ree_pinns/ree_pinns_enhanced/v1/results.png'):
        """
        绘制完整的REE分馏结果图（参照原始模型）
        """
        self.pinn.eval()

        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(3, 3, figure=fig)

        time_points = [0, 2500, 5000, 7500, 10000]
        colors = plt.cm.viridis(np.linspace(0, 1, len(time_points)))

        z = np.linspace(0, self.params['h'], 100)

        ax1 = fig.add_subplot(gs[0, 0])
        for i, t in enumerate(time_points):
            C_LREE, _ = self.predict(np.full(100, t), z)
            ax1.plot(C_LREE, z, color=colors[i],
                    label=f't={t/1000:.1f} kyr', linewidth=2)
        ax1.set_xlabel('LREE Concentration [mg/kg]')
        ax1.set_ylabel('Depth [m]')
        ax1.set_title('LREE Profile Evolution')
        ax1.legend(loc='best')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[0, 1])
        for i, t in enumerate(time_points):
            _, C_HREE = self.predict(np.full(100, t), z)
            ax2.plot(C_HREE, z, color=colors[i],
                    label=f't={t/1000:.1f} kyr', linewidth=2)
        ax2.set_xlabel('HREE Concentration [mg/kg]')
        ax2.set_ylabel('Depth [m]')
        ax2.set_title('HREE Profile Evolution')
        ax2.legend(loc='best')
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[0, 2])
        for i, t in enumerate(time_points):
            C_LREE, C_HREE = self.predict(np.full(100, t), z)
            ratio = C_LREE / (C_HREE + 1e-8)
            ax3.plot(ratio, z, color=colors[i],
                    label=f't={t/1000:.1f} kyr', linewidth=2)
        ax3.set_xlabel('LREE/HREE Ratio')
        ax3.set_ylabel('Depth [m]')
        ax3.set_title('LREE/HREE Fractionation')
        ax3.legend(loc='best')
        ax3.invert_yaxis()
        ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(gs[1, 0])
        t_final = self.params['T']
        C_LREE_final, C_HREE_final = self.predict(np.full(100, t_final), z)
        ax4.plot(C_LREE_final, z, 'b-', linewidth=2, label='LREE')
        ax4.plot(C_HREE_final, z, 'r-', linewidth=2, label='HREE')
        ax4.set_xlabel('Concentration [mg/kg]')
        ax4.set_ylabel('Depth [m]')
        ax4.set_title('Final REE Profile (t = 10 kyr)')
        ax4.legend(loc='best')
        ax4.invert_yaxis()
        ax4.grid(True, alpha=0.3)

        ax5 = fig.add_subplot(gs[1, 1])
        history = self.trainer.history
        iters = range(len(history['total']))
        ax5.semilogy(iters, history['total'], 'k-', linewidth=1, label='Total')
        ax5.semilogy(iters, history['physics'], 'b-', linewidth=1, label='Physics')
        ax5.semilogy(iters, history['obs'], 'r-', linewidth=1, label='Observation')
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Loss')
        ax5.set_title('Training Loss History')
        ax5.legend(loc='best')
        ax5.grid(True, alpha=0.3)

        ax6 = fig.add_subplot(gs[1, 2])
        selectivity = self.params['k_ads_HREE'] / self.params['k_ads_LREE']
        ratio_final = C_LREE_final / (C_HREE_final + 1e-8)
        ax6.axhline(y=selectivity, color='k', linestyle='--',
                   label=f'Theoretical: {selectivity:.1f}')
        ax6.plot(ratio_final, z, 'b-', linewidth=2, label='Observed')
        ax6.set_xlabel('LREE/HREE Ratio')
        ax6.set_ylabel('Depth [m]')
        ax6.set_title('Fractionation Factor')
        ax6.legend(loc='best')
        ax6.invert_yaxis()
        ax6.grid(True, alpha=0.3)

        ax7 = fig.add_subplot(gs[2, 0])
        ax8 = fig.add_subplot(gs[2, 1])

        n_t, n_z = 50, 50
        t_range = np.linspace(0, self.params['T'], n_t)
        z_range = np.linspace(0, self.params['h'], n_z)
        T_grid, Z_grid = np.meshgrid(t_range, z_range)

        C_LREE_flat, _ = self.predict(T_grid.flatten(), Z_grid.flatten())
        C_LREE_grid = C_LREE_flat.reshape(n_z, n_t)

        _, C_HREE_flat = self.predict(T_grid.flatten(), Z_grid.flatten())
        C_HREE_grid = C_HREE_flat.reshape(n_z, n_t)

        im = ax7.contourf(T_grid/1000, Z_grid, C_LREE_grid, levels=20, cmap='viridis')
        plt.colorbar(im, ax=ax7, label='LREE [mg/kg]')
        ax7.set_xlabel('Time [kyr]')
        ax7.set_ylabel('Depth [m]')
        ax7.set_title('LREE Spatiotemporal Distribution')

        im = ax8.contourf(T_grid/1000, Z_grid, C_HREE_grid, levels=20, cmap='plasma')
        plt.colorbar(im, ax=ax8, label='HREE [mg/kg]')
        ax8.set_xlabel('Time [kyr]')
        ax8.set_ylabel('Depth [m]')
        ax8.set_title('HREE Spatiotemporal Distribution')

        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')

        param_text = (
            f"Model Parameters:\n"
            f"{'─' * 20}\n"
            f"D = {self.params['D']:.4f} m²/year\n"
            f"u = {self.params['u']:.2e} m/year\n"
            f"k_ads(LREE) = {self.params['k_ads_LREE']:.2f} 1/year\n"
            f"k_ads(HREE) = {self.params['k_ads_HREE']:.2f} 1/year\n"
            f"{'─' * 20}\n"
            f"Selectivity = {selectivity:.2f}\n"
            f"{'─' * 20}\n"
            f"T_final = {self.params['T']/1000:.0f} kyr\n"
            f"h = {self.params['h']:.1f} m"
        )

        ax9.text(0.1, 0.9, param_text, fontsize=10,
                family='monospace', verticalalignment='top')

        fig.suptitle('Enhanced REE Fractionation PINNs Simulation Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"图片已保存到: {save_path}")


if __name__ == '__main__':
    print("增强版 PINNs 测试\n")

    env_data = {
        'pH': {'z': np.array([0.0, 0.5, 1.0]), 'values': np.array([5.5, 6.0, 6.2])},
        'clay': {'z': np.array([0.0, 0.5, 1.0]), 'values': np.array([0.2, 0.4, 0.3])},
    }

    np.random.seed(42)
    obs_data = {
        't': np.random.uniform(0, 10000, 10),
        'z': np.random.uniform(0, 1, 10),
        'C_LREE': np.random.uniform(0.05, 0.2, 10),
        'C_HREE': np.random.uniform(0.02, 0.1, 10)
    }

    sim = EnhancedREESimulator(env_data=env_data, obs_data=obs_data)
    sim.train(n_iterations=100, n_physics=200, n_bc=50, print_every=50)

    z_test = np.linspace(0, 1, 10)
    C_L, C_H = sim.predict(np.full(10, 5000.0), z_test)
    print(f"\n预测结果: LREE={C_L.flatten()[:3]}..., HREE={C_H.flatten()[:3]}...")

    print("\n正在生成可视化图表...")
    sim.plot_results()
    print("\n✅ 完成!")
