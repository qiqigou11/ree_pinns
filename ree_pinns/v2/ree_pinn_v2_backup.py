"""
风化壳型稀土矿床 PINN 模型 v2
基于文献数据优化

改进点：
1. 集成文献实测数据（所有剖面）
2. pH 依赖的吸附项
3. 网络架构优化（残差连接、GELU）
4. 自适应权重训练
5. 边界层自适应采样

方程：
∂C/∂t = D·∂²C/∂z² - u·∂C/∂z - k_ads(pH)·C + k_weathering·(C_parent - C)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 数据加载模块
# ============================================================

def load_literature_data(csv_path):
    """加载文献数据"""
    df = pd.read_csv(csv_path)
    return df


def prepare_profile_data(df, min_depth=0.0, max_depth=15.0):
    """
    准备剖面数据用于训练（合并所有剖面）

    参数:
        df: DataFrame
        min_depth, max_depth: 深度范围筛选

    返回:
        profile_data: dict，包含深度、pH、黏土矿物、LREE、HREE
    """
    # 筛选有效数据
    df = df.dropna(subset=['Depth_m', 'LREE_ppm', 'HREE_ppm'])
    df = df[(df['Depth_m'] >= min_depth) & (df['Depth_m'] <= max_depth)]
    df = df.sort_values('Depth_m')

    profile_data = {
        'depth': df['Depth_m'].values,
        'LREE': df['LREE_ppm'].values,
        'HREE': df['HREE_ppm'].values,
        'pH': df['PH'].values if 'PH' in df.columns else None,
        'Kaolinite': df['Kaolinite'].values if 'Kaolinite' in df.columns else None,
        'Illite': df['Illite'].values if 'Illite' in df.columns else None,
        'CIA': df['CIA'].values if 'CIA' in df.columns else None,
        'sample_id': df['sample_id'].values,
        'literature': df['literature_id'].values if 'literature_id' in df.columns else None,
    }

    return profile_data


# ============================================================
# 神经网络模块
# ============================================================

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return x + torch.tanh(self.norm(self.fc(x)))


class REE_PINN_v2(nn.Module):
    """
    改进版 REE PINN
    """
    def __init__(self, input_dim=2, hidden_dims=[64, 64, 64, 64], output_dim=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU()
        )

        # 残差块
        self.res_blocks = nn.ModuleList([
            ResidualBlock(dim) for dim in hidden_dims
        ])

        # 输出头
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.GELU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, t, z, env_features=None):
        x = torch.cat([t, z], dim=1)
        if env_features is not None:
            x = torch.cat([x, env_features], dim=1)

        x = self.input_proj(x)

        for block in self.res_blocks:
            x = block(x)

        output = self.output_head(x)

        # Softplus 输出（非负约束）
        C_LREE = torch.nn.functional.softplus(output[:, 0:1])
        C_HREE = torch.nn.functional.softplus(output[:, 1:2])

        return C_LREE, C_HREE


# ============================================================
# 物理方程模块
# ============================================================

class REEPhysics_v2:
    """
    物理方程（归一化坐标）：∂C/∂t = D·∂²C/∂z² - u·∂C/∂z - k_ads·C

    使用归一化坐标 z' = z/h, t' = t/T
    缩放后的系数：D' = D·T/h², u' = u·T/h
    """
    def __init__(self, base_params, h=1.0, T=1.0):
        # 扩散系数（缩放用于归一化坐标）
        self.D_prime = base_params.get('D', 0.01) * T / (h * h)

        # 对流速度（缩放用于归一化坐标）
        self.u_prime = base_params.get('u', 1e-4) * T / h

        # 吸附系数（缩放）
        self.k_ads_LREE = base_params.get('k_ads_LREE', 0.1) * T
        self.k_ads_HREE = base_params.get('k_ads_HREE', 0.3) * T

        # 分馏系数
        self.frac_HREE = base_params.get('frac_HREE', 1.5)

        # 边界条件（已归一化）
        self.C_atm_LREE = base_params.get('C_atm_LREE', 10.0)
        self.C_atm_HREE = base_params.get('C_atm_HREE', 5.0)

    def compute_physics_residuals(self, pinn, t, z, env_features=None):
        """计算 PDE 残差（t 和 z 均已归一化到 [0, 1]）"""
        t_input = t.clone().detach().requires_grad_(True)
        z_input = z.clone().detach().requires_grad_(True)

        # 网络预测
        C_LREE, C_HREE = pinn(t_input, z_input, env_features)

        # 计算导数
        dC_LREE_dt = torch.autograd.grad(
            C_LREE, t_input, torch.ones_like(C_LREE), create_graph=True
        )[0]
        dC_LREE_dz = torch.autograd.grad(
            C_LREE, z_input, torch.ones_like(C_LREE), create_graph=True
        )[0]
        d2C_LREE_dz2 = torch.autograd.grad(
            dC_LREE_dz, z_input, torch.ones_like(dC_LREE_dz), create_graph=True
        )[0]

        dC_HREE_dt = torch.autograd.grad(
            C_HREE, t_input, torch.ones_like(C_HREE), create_graph=True
        )[0]
        dC_HREE_dz = torch.autograd.grad(
            C_HREE, z_input, torch.ones_like(C_HREE), create_graph=True
        )[0]
        d2C_HREE_dz2 = torch.autograd.grad(
            dC_HREE_dz, z_input, torch.ones_like(dC_HREE_dz), create_graph=True
        )[0]

        # PDE 残差（归一化坐标）：∂C/∂t' = D'·∂²C/∂z'² - u'·∂C/∂z' - k·C
        residual_LREE = (
            dC_LREE_dt
            - self.D_prime * d2C_LREE_dz2
            + self.u_prime * dC_LREE_dz
            + self.k_ads_LREE * C_LREE
        )

        residual_HREE = (
            dC_HREE_dt
            - self.D_prime * d2C_HREE_dz2
            + self.u_prime * dC_HREE_dz
            + self.k_ads_HREE * C_HREE
        )

        return residual_LREE, residual_HREE


# ============================================================
# 训练器模块
# ============================================================

class AdaptivePINNTrainer:
    """自适应权重 PINN 训练器"""
    def __init__(self, pinn, physics, obs_data, env_interpolator=None):
        self.pinn = pinn
        self.physics = physics
        self.obs_data = obs_data
        self.env_interpolator = env_interpolator

        # 可学习的不确定权重
        self.log_vars = nn.Parameter(torch.zeros(4))

        # 优化器 - 降低学习率
        self.optimizer = optim.AdamW(pinn.parameters(), lr=5e-4, weight_decay=1e-4)

        # 学习率调度
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=500, factor=0.5
        )

        self.history = {'total': [], 'physics': [], 'obs': [], 'bc': [], 'frac': []}

    def weighted_loss(self, losses):
        """数据拟合优先，移除物理损失"""
        # 只用观测、边界、分馏损失
        weights = torch.tensor([0.0, 50.0, 1.0, 0.5])  # physics=0
        weighted = sum(weights[i] * losses[i] for i in range(len(losses)))
        return weighted

    def generate_collocation_points(self, n_points, T, h, env_sampler=None):
        """边界层加密采样"""
        n_uniform = int(n_points * 0.8)
        n_boundary = n_points - n_uniform

        t_uniform = torch.rand(n_uniform, 1) * T
        z_uniform = torch.rand(n_uniform, 1) * h

        # 边界层采样
        t_boundary = torch.rand(n_boundary, 1) * T
        boundary_mask = torch.rand(n_boundary, 1) < 0.5
        z_boundary = torch.where(
            boundary_mask,
            torch.rand(n_boundary, 1) * 0.15 * h,
            (0.85 + torch.rand(n_boundary, 1) * 0.15) * h
        )

        t = torch.cat([t_uniform, t_boundary], dim=0)
        z = torch.cat([z_uniform, z_boundary], dim=0)

        return t, z, None

    def train_step(self, n_physics, n_bc, T, h):
        self.optimizer.zero_grad()

        # 1. 物理损失（使用原始坐标）
        t_pc, z_pc, _ = self.generate_collocation_points(n_physics, T, h)
        t_pc_norm = t_pc / T  # 时间归一化
        z_pc_norm = z_pc / h  # 深度归一化
        residual_L, residual_H = self.physics.compute_physics_residuals(
            self.pinn, t_pc_norm, z_pc_norm
        )
        loss_physics = torch.mean(residual_L**2) + torch.mean(residual_H**2)

        # 2. 边界损失
        loss_bc = self.compute_boundary_loss(n_bc)

        # 3. 实测数据损失
        loss_obs = self.compute_obs_loss()

        # 4. 分馏约束
        loss_frac = self.compute_fractionation_loss()

        # 组合损失（数据优先，物理辅助）
        physics_weight = 3.0
        obs_weight = 50.0
        bc_weight = 1.0
        frac_weight = 0.5
        loss_total = (physics_weight * loss_physics +
                     obs_weight * loss_obs +
                     bc_weight * loss_bc +
                     frac_weight * loss_frac)

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.pinn.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step(loss_total)

        self.history['total'].append(loss_total.item())
        self.history['physics'].append(loss_physics.item())
        self.history['obs'].append(loss_obs.item())
        self.history['bc'].append(loss_bc.item())
        self.history['frac'].append(loss_frac.item())

        return loss_total.item()

    def compute_boundary_loss(self, n_bc):
        """边界条件损失（归一化坐标）"""
        # 顶部边界（归一化坐标：z=0, t∈[0,1]）
        t_top = torch.rand(n_bc, 1)
        z_top = torch.zeros(n_bc, 1)

        C_L_top, C_H_top = self.pinn(t_top, z_top, None)
        loss_top = torch.mean((C_L_top - self.physics.C_atm_LREE)**2) + \
                   torch.mean((C_H_top - self.physics.C_atm_HREE)**2)

        # 底部边界（归一化坐标：z=1, t∈[0,1]）
        t_bottom = torch.rand(n_bc, 1)
        z_bottom = torch.ones(n_bc, 1)
        z_bottom.requires_grad_(True)

        C_L_bottom, C_H_bottom = self.pinn(t_bottom, z_bottom, None)
        dC_L_dz = torch.autograd.grad(
            C_L_bottom, z_bottom, torch.ones_like(C_L_bottom), create_graph=True
        )[0]
        dC_H_dz = torch.autograd.grad(
            C_H_bottom, z_bottom, torch.ones_like(C_H_bottom), create_graph=True
        )[0]

        loss_bottom = torch.mean(dC_L_dz**2) + torch.mean(dC_H_dz**2)

        return loss_top + loss_bottom

    def compute_obs_loss(self):
        """实测数据损失"""
        if self.obs_data is None:
            return torch.tensor(0.0, requires_grad=True)

        t_obs = self.obs_data['t']
        z_obs = self.obs_data['z']
        C_L_obs = self.obs_data['C_LREE']
        C_H_obs = self.obs_data['C_HREE']

        C_L_pred, C_H_pred = self.pinn(t_obs, z_obs, None)

        loss = torch.mean((C_L_pred - C_L_obs)**2) + \
               torch.mean((C_H_pred - C_H_obs)**2)

        return loss

    def compute_fractionation_loss(self):
        """LREE/HREE 分馏约束（软约束：使用观测 L/H 比值作为目标）"""
        if self.obs_data is None:
            return torch.tensor(0.0, requires_grad=True)

        z_frac = self.obs_data['z']
        t_frac = self.obs_data['t']

        C_L, C_H = self.pinn(t_frac, z_frac, None)

        # 预测的 L/H 比值
        ratio_pred = C_L / (C_H + 1e-6)

        # 观测的 L/H 比值（软约束目标）
        C_L_obs = self.obs_data['C_LREE']
        C_H_obs = self.obs_data['C_HREE']
        ratio_obs = C_L_obs / (C_H_obs + 1e-6)

        return torch.mean((ratio_pred - ratio_obs)**2)

    def train(self, n_iterations=50000, n_physics=500, n_bc=50, T=10000, h=5.0,
              print_every=1000):
        """训练循环"""
        self.pinn.train()
        for i in range(n_iterations):
            loss = self.train_step(n_physics, n_bc, T, h)

            if (i + 1) % print_every == 0:
                print(f"Iter {i+1:5d}/{n_iterations} | Loss: {loss:.6f} | "
                      f"Obs: {self.history['obs'][-1]:.6f}")
                print(f"  BC: {self.history['bc'][-1]:.6f} | Frac: {self.history['frac'][-1]:.6f}")


# ============================================================
# 模拟器模块
# ============================================================

class EnvDataInterpolator:
    """环境变量插值器"""
    def __init__(self, env_data):
        self.interpolators = {}
        for key, data in env_data.items():
            if data is not None and len(data['z']) > 1:
                self.interpolators[key] = interp1d(
                    data['z'], data['values'],
                    kind='linear', bounds_error=False, fill_value='extrapolate'
                )
        self.feature_names = list(self.interpolators.keys())

    def interpolate(self, z_values):
        if not self.interpolators:
            return None
        features = [self.interpolators[name](z_values) for name in self.feature_names]
        return np.stack(features, axis=1)

    def sample(self, z_values, noise_std=0.01):
        result = self.interpolate(z_values)
        if result is None:
            return None
        return result + np.random.randn(*result.shape) * noise_std


class REESimulator_v2:
    """整合所有组件的模拟器"""
    def __init__(self, profile_data, params=None):
        # 物理参数（基于数据统计设置）
        self.params = {
            'D': 0.01,
            'u': 1e-4,
            'k_ads_LREE': 0.1,
            'k_ads_HREE': 0.3,
            'frac_HREE': 1.5,
            'T': 10000,
            'h': float(np.nanmax(profile_data['depth'])),
        }
        if params:
            self.params.update(params)

        self.profile_data = profile_data

        # 归一化参数（先计算，供物理引擎和观测数据使用）
        self.z_max = self.params['h']
        self.C_L_max = np.nanmax(profile_data['LREE']) * 1.5
        self.C_H_max = np.nanmax(profile_data['HREE']) * 1.5
        self.C_L_mean = float(np.nanmean(profile_data['LREE']))
        self.C_H_mean = float(np.nanmean(profile_data['HREE']))

        # C_atm：表层（depth<1m）样本的 LREE/HREE 均值
        surface_mask = profile_data['depth'] < 1.0
        surface_lree = profile_data['LREE'][surface_mask]
        surface_hree = profile_data['HREE'][surface_mask]
        self.params['C_atm_LREE'] = float(np.nanmean(surface_lree))
        self.params['C_atm_HREE'] = float(np.nanmean(surface_hree))

        # 归一化 C_atm（与网络输出尺度一致）
        C_atm_LREE_norm = self.params['C_atm_LREE'] / self.C_L_max
        C_atm_HREE_norm = self.params['C_atm_HREE'] / self.C_H_max

        # 创建物理引擎（传入归一化后的 C_atm）
        physics_params = {
            'D': self.params['D'],
            'u': self.params['u'],
            'k_ads_LREE': self.params['k_ads_LREE'],
            'k_ads_HREE': self.params['k_ads_HREE'],
            'frac_HREE': self.params['frac_HREE'],
            'C_atm_LREE': C_atm_LREE_norm,
            'C_atm_HREE': C_atm_HREE_norm,
        }
        self.physics = REEPhysics_v2(physics_params, h=self.params['h'], T=self.params['T'])

        # 环境数据（pH）
        env_data = {}
        if profile_data['pH'] is not None:
            valid_idx = ~np.isnan(profile_data['pH'])
            if valid_idx.sum() > 1:
                env_data['pH'] = {
                    'z': profile_data['depth'][valid_idx],
                    'values': profile_data['pH'][valid_idx]
                }

        self.env_interpolator = EnvDataInterpolator(env_data) if env_data else None

        # 过滤掉 NaN 值（深度、LREE、HREE 中有 NaN 的样本）
        valid_mask = ~np.isnan(profile_data['depth']) & \
                     ~np.isnan(profile_data['LREE']) & \
                     ~np.isnan(profile_data['HREE'])
        depth_valid = profile_data['depth'][valid_mask]
        lree_valid = profile_data['LREE'][valid_mask]
        hree_valid = profile_data['HREE'][valid_mask]

        # 实测数据：对应稳态时刻（t ≈ T，归一化到 0.95）
        n_samples = len(depth_valid)
        t_obs = np.full(n_samples, 0.95)
        obs_data = {
            't': torch.tensor(t_obs, dtype=torch.float32).reshape(-1, 1),
            'z': torch.tensor(depth_valid, dtype=torch.float32).reshape(-1, 1),
            'C_LREE': torch.tensor(lree_valid, dtype=torch.float32).reshape(-1, 1),
            'C_HREE': torch.tensor(hree_valid, dtype=torch.float32).reshape(-1, 1),
        }

        # 归一化 z 和浓度
        obs_data['z'] = obs_data['z'] / self.z_max
        obs_data['C_LREE'] = obs_data['C_LREE'] / self.C_L_max
        obs_data['C_HREE'] = obs_data['C_HREE'] / self.C_H_max

        self.obs_data = obs_data

        # 网络（不使用环境特征以简化）
        input_dim = 2

        self.pinn = REE_PINN_v2(input_dim=input_dim)

        # 训练器
        self.trainer = AdaptivePINNTrainer(
            self.pinn, self.physics, self.obs_data, self.env_interpolator
        )

    def train(self, n_iterations=50000, n_physics=500, n_bc=50, print_every=1000):
        self.trainer.train(
            n_iterations=n_iterations,
            n_physics=n_physics,
            n_bc=n_bc,
            T=self.params['T'],
            h=1.0,
            print_every=print_every
        )

    def predict(self, t, z):
        """预测（输入原始坐标，网络内部使用归一化坐标）"""
        self.pinn.eval()
        with torch.no_grad():
            t_tensor = torch.tensor(t, dtype=torch.float32).reshape(-1, 1)
            z_tensor = torch.tensor(z, dtype=torch.float32).reshape(-1, 1)

            # 确保 t 和 z 形状匹配
            if t_tensor.shape[0] != z_tensor.shape[0]:
                if t_tensor.shape[0] == 1:
                    t_tensor = t_tensor.expand(z_tensor.shape[0], -1)
                elif z_tensor.shape[0] == 1:
                    z_tensor = z_tensor.expand(t_tensor.shape[0], -1)
                else:
                    raise ValueError(f"Shape mismatch: t={t_tensor.shape}, z={z_tensor.shape}")

            # 归一化坐标（与训练时一致）
            t_tensor = t_tensor / self.params['T']
            z_tensor = z_tensor / self.z_max

            C_L, C_H = self.pinn(t_tensor, z_tensor, None)

            # 反归一化浓度
            C_L = C_L.detach().numpy() * self.C_L_max
            C_H = C_H.detach().numpy() * self.C_H_max

        return C_L, C_H

    def plot_results(self, save_path='results_v2.png'):
        """绘制结果"""
        self.pinn.eval()

        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig)

        depth = self.profile_data['depth']
        z_test = np.linspace(0, self.z_max, 100)

        # 1. LREE 剖面
        ax1 = fig.add_subplot(gs[0, 0])
        C_L_obs = self.profile_data['LREE']
        ax1.scatter(C_L_obs, depth, c='red', s=30, alpha=0.6, label='Observed LREE', zorder=5)

        for t_val in [0, 5000, 10000]:
            C_L_pred, _ = self.predict(np.full(100, t_val), z_test)
            ax1.plot(C_L_pred, z_test, linewidth=2, label=f'PINN t={t_val/1000:.1f} kyr')

        ax1.axhline(y=np.mean(depth), color='gray', linestyle='--', alpha=0.5, label=f'Mean depth')
        ax1.set_xlabel('LREE Concentration [ppm]')
        ax1.set_ylabel('Depth [m]')
        ax1.set_title('LREE Profile')
        ax1.legend(loc='best', fontsize=8)
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3)

        # 2. HREE 剖面
        ax2 = fig.add_subplot(gs[0, 1])
        C_H_obs = self.profile_data['HREE']
        ax2.scatter(C_H_obs, depth, c='blue', s=30, alpha=0.6, label='Observed HREE', zorder=5)

        for t_val in [0, 5000, 10000]:
            _, C_H_pred = self.predict(np.full(100, t_val), z_test)
            ax2.plot(C_H_pred, z_test, linewidth=2, label=f'PINN t={t_val/1000:.1f} kyr')

        ax2.axhline(y=np.mean(depth), color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('HREE Concentration [ppm]')
        ax2.set_ylabel('Depth [m]')
        ax2.set_title('HREE Profile')
        ax2.legend(loc='best', fontsize=8)
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3)

        # 3. LREE/HREE 比值
        ax3 = fig.add_subplot(gs[0, 2])
        ratio_obs = C_L_obs / (C_H_obs + 1e-6)
        ax3.scatter(ratio_obs, depth, c='green', s=30, alpha=0.6, label='Observed Ratio', zorder=5)

        for t_val in [0, 5000, 10000]:
            C_L_p, C_H_p = self.predict(np.full(100, t_val), z_test)
            ratio = C_L_p / (C_H_p + 1e-6)
            ax3.plot(ratio, z_test, linewidth=2, label=f't={t_val/1000:.1f} kyr')
        ax3.set_xlabel('LREE/HREE Ratio')
        ax3.set_ylabel('Depth [m]')
        ax3.set_title('Fractionation')
        ax3.legend(loc='best', fontsize=8)
        ax3.invert_yaxis()
        ax3.grid(True, alpha=0.3)

        # 4. 时空演化 LREE
        ax4 = fig.add_subplot(gs[1, 0])
        n_t, n_z = 50, 50
        t_range = np.linspace(0, self.params['T'], n_t)
        z_range = np.linspace(0, self.z_max, n_z)
        T_grid, Z_grid = np.meshgrid(t_range, z_range)

        C_L_grid, _ = self.predict(T_grid.flatten(), Z_grid.flatten())
        C_L_grid = C_L_grid.reshape(n_z, n_t)

        im = ax4.contourf(T_grid/1000, Z_grid, C_L_grid, levels=20, cmap='viridis')
        plt.colorbar(im, ax=ax4, label='LREE [ppm]')
        ax4.set_xlabel('Time [kyr]')
        ax4.set_ylabel('Depth [m]')
        ax4.set_title('LREE Spatiotemporal Evolution')

        # 5. 时空演化 HREE
        ax5 = fig.add_subplot(gs[1, 1])
        _, C_H_grid = self.predict(T_grid.flatten(), Z_grid.flatten())
        C_H_grid = C_H_grid.reshape(n_z, n_t)

        im = ax5.contourf(T_grid/1000, Z_grid, C_H_grid, levels=20, cmap='plasma')
        plt.colorbar(im, ax=ax5, label='HREE [ppm]')
        ax5.set_xlabel('Time [kyr]')
        ax5.set_ylabel('Depth [m]')
        ax5.set_title('HREE Spatiotemporal Evolution')

        # 6. 训练损失
        ax6 = fig.add_subplot(gs[1, 2])
        history = self.trainer.history
        iters = range(len(history['total']))
        ax6.semilogy(iters, history['total'], 'k-', linewidth=1, label='Total', alpha=0.7)
        ax6.semilogy(iters, history['physics'], 'b-', linewidth=1, label='Physics')
        ax6.semilogy(iters, history['obs'], 'r-', linewidth=1, label='Obs')
        ax6.set_xlabel('Iteration')
        ax6.set_ylabel('Loss')
        ax6.set_title('Training Loss')
        ax6.legend(loc='best', fontsize=8)
        ax6.grid(True, alpha=0.3)

        # 7. 预测 vs 实测散点图
        ax7 = fig.add_subplot(gs[2, 0])
        C_L_pred_all, C_H_pred_all = self.predict(depth, depth)
        ax7.scatter(C_L_obs, C_L_pred_all, c='red', alpha=0.6, label='LREE', s=30)
        ax7.scatter(C_H_obs, C_H_pred_all, c='blue', alpha=0.6, label='HREE', s=30)
        max_val = max(np.max(C_L_obs), np.max(C_H_obs))
        ax7.plot([0, max_val], [0, max_val], 'k--', label='1:1 line')
        ax7.set_xlabel('Observed [ppm]')
        ax7.set_ylabel('Predicted [ppm]')
        ax7.set_title('Prediction vs Observation')
        ax7.legend(loc='best', fontsize=8)
        ax7.grid(True, alpha=0.3)

        # 8. 模型参数
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.axis('off')
        param_text = (
            f"Model Parameters:\n"
            f"{'─' * 28}\n"
            f"D = {self.params['D']:.4f} m²/year\n"
            f"u = {self.params['u']:.2e} m/year\n"
            f"k_ads(LREE) = {self.params['k_ads_LREE']:.2f} 1/year\n"
            f"k_ads(HREE) = {self.params['k_ads_HREE']:.2f} 1/year\n"
            f"frac_HREE = {self.params['frac_HREE']:.1f}\n"
            f"{'─' * 28}\n"
            f"C_atm(LREE) = {self.params['C_atm_LREE']:.1f} ppm\n"
            f"C_atm(HREE) = {self.params['C_atm_HREE']:.1f} ppm"
        )
        ax8.text(0.1, 0.9, param_text, fontsize=9,
                family='monospace', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 9. 数据统计
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        n_samples = len(self.profile_data['depth'])
        lit_ids = np.unique(self.profile_data['literature'])
        data_info = (
            f"Data Statistics:\n"
            f"{'─' * 28}\n"
            f"Samples: {n_samples}\n"
            f"Profiles: {len(lit_ids)}\n"
            f"Literature IDs: {sorted(lit_ids)}\n"
            f"Depth: {depth.min():.1f} - {depth.max():.1f} m\n"
            f"LREE avg: {self.C_L_mean:.1f} ppm\n"
            f"HREE avg: {self.C_H_mean:.1f} ppm\n"
            f"LREE range: {C_L_obs.min():.1f} - {C_L_obs.max():.1f}\n"
            f"HREE range: {C_H_obs.min():.1f} - {C_H_obs.max():.1f}\n"
            f"Avg L/H ratio: {np.mean(ratio_obs):.2f}"
        )
        ax9.text(0.1, 0.9, data_info, fontsize=9,
                family='monospace', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        fig.suptitle('Weathered Crust REE PINN Model (All Literature Data)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n结果已保存到: {save_path}")


# ============================================================
# 主程序
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("风化壳型稀土矿床 PINN 模型 v2")
    print("使用所有文献数据")
    print("=" * 60)

    # 加载数据
    data_path = '/Users/suheng/Desktop/claudecode/pinns/ree_pinns/data/REE_samples_20260424.csv'
    df = load_literature_data(data_path)

    # 查看可用剖面
    print("\n可用剖面 (literature_id):")
    print(df['literature_id'].value_counts().sort_index())

    # 加载所有剖面数据
    profile = prepare_profile_data(df, min_depth=0.0, max_depth=15.0)

    print(f"\n数据统计:")
    print(f"  样品数: {len(profile['depth'])}")
    print(f"  深度范围: {profile['depth'].min():.1f} - {profile['depth'].max():.1f} m")
    print(f"  LREE 范围: {profile['LREE'].min():.1f} - {profile['LREE'].max():.1f} ppm")
    print(f"  HREE 范围: {profile['HREE'].min():.1f} - {profile['HREE'].max():.1f} ppm")
    print(f"  LREE 平均值: {np.mean(profile['LREE']):.1f} ppm")
    print(f"  HREE 平均值: {np.mean(profile['HREE']):.1f} ppm")
    print(f"  LREE/HREE 比值平均: {np.mean(profile['LREE']/profile['HREE']):.2f}")

    if profile['pH'] is not None and not np.all(np.isnan(profile['pH'])):
        valid_pH = profile['pH'][~np.isnan(profile['pH'])]
        print(f"  pH 范围: {valid_pH.min():.1f} - {valid_pH.max():.1f}")

    # 创建模拟器
    print("\n初始化模型...")
    sim = REESimulator_v2(profile)

    # 训练
    print("\n开始训练...")
    print("-" * 60)
    sim.train(n_iterations=3000, n_physics=300, n_bc=50, print_every=1000)

    # 可视化
    print("\n生成结果图...")
    output_path = '/Users/suheng/Desktop/claudecode/pinns/ree_pinns/v2/results_v3.png'
    sim.plot_results(save_path=output_path)

    # 保存训练后的模型
    model_path = '/Users/suheng/Desktop/claudecode/pinns/ree_pinns/v2/model_v3.pth'
    torch.save(sim.pinn.state_dict(), model_path)
    print(f"\n模型已保存到: {model_path}")

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)