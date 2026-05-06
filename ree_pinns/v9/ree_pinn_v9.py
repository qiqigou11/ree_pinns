"""
REE PINN v9 - Climate-Aware Multi-Profile Integration
=======================================================
把所有剖面数据整合在一起训练，用气候条件作为额外输入。

核心改动（vs v8）：
- v8: 每个剖面独立训练，独立归一化，无法跨剖面迁移
- v9: 所有样品一起训练，输入 = [z_norm, T_norm, P_norm, R_norm]
       网络自己学习"深度+气候 → REE" 的隐式物理关系

输入:
  - z_norm: 归一化深度，z / 33.0
  - T_norm: 归一化年平均温度，(T_K - T_mean) / T_std
  - P_norm: 归一化年降水量，(P_mm - P_mean) / P_std
  - R_norm: 归一化径流量，  (R_m - R_mean) / R_std

输出:
  - C_total: TotalREE (ppm)

网络结构: [4D SIREN] → 隐藏层 × 5 → [Linear + ReLU6] → C_scale × ppm

损失函数:
  L_data = MSE(C_pred, C_obs)
  L_bc   = MSE(C(z=0), C_atm_global)，C_atm = 0~1m 表层样本均值
  L_total = 100 * L_data + 10 * L_bc
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

DATA_PATH = '/Users/suheng/Desktop/claudecode/pinns/ree_pinns/data/REE_samples_20260424_with_climate.csv'
LIT_PATH = '/Users/suheng/Desktop/claudecode/pinns/ree_pinns/data/REE_literature_20260424.csv'

# =====================================================================
# 全局归一化参数（从数据统计得出）
# =====================================================================
Z_MAX = 33.0        # 全局最大深度(m)
T_MEAN = 290.62     # K
T_STD = 3.78
P_MEAN = 1604.84    # mm/yr
P_STD = 710.27
R_MEAN = 0.5210     # m/yr
R_STD = 0.2310

C_SCALE = 4000.0    # 全局浓度上限（ppm）


# =====================================================================
# 1. 网络架构（4D SIREN）
# =====================================================================
class SineLayer(nn.Module):
    def __init__(self, in_f, out_f, is_first=False, omega=30):
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(in_f, out_f)
        nn.init.constant_(self.linear.bias, 0.0)
        if is_first:
            nn.init.uniform_(self.linear.weight, -1.0 / in_f, 1.0 / in_f)
        else:
            nn.init.uniform_(self.linear.weight, -np.sqrt(6 / in_f) / omega,
                                   np.sqrt(6 / in_f) / omega)

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


class REE_PINN(nn.Module):
    """
    气候感知的 REE 预测网络

    输入: [z_norm, T_norm, P_norm, R_norm] — 4D
    输出: TotalREE (ppm)

    架构: 4D SIREN (4-64-64-64-64-64-1)
    输出层用 ReLU6 限制在 [0, C_SCALE] 范围
    """
    def __init__(self, hidden_dim=64, n_layers=5):
        super().__init__()
        layers = [SineLayer(4, hidden_dim, is_first=True)]
        for _ in range(n_layers - 1):
            layers.append(SineLayer(hidden_dim, hidden_dim))
        self.net = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.ReLU6()  # 限制 >= 0，× C_SCALE 后限制在 [0, C_SCALE]
        )

    def forward(self, x):
        """
        x: (N, 4) — [z_norm, T_norm, P_norm, R_norm]
        返回: (N, 1) — TotalREE (ppm)
        """
        if x.dim() == 1:
            x = x.reshape(1, -1)
        h = self.net(x)
        C = self.head(h) * C_SCALE
        return C


# =====================================================================
# 2. 训练器
# =====================================================================
class Trainer:
    def __init__(self, pinn, atm_conc, z_obs, climate_obs, C_obs):
        """
        pinn: REE_PINN 模型
        atm_conc: 边界条件，表层(0~1m)样本的 TotalREE 均值
        z_obs: 深度数据 (N,)
        climate_obs: 气候数据 (N, 3) — [T_norm, P_norm, R_norm]
        C_obs: 实测 TotalREE (N,)
        """
        self.pinn = pinn
        self.atm_conc = float(atm_conc)
        self.device = next(pinn.parameters()).device

        self.z_obs = z_obs.to(self.device)
        self.climate_obs = climate_obs.to(self.device)
        self.C_obs = C_obs.to(self.device)

        self.opt = torch.optim.Adam(pinn.parameters(), lr=1e-3)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, patience=500, factor=0.5, min_lr=1e-6)

    def step(self, N_bc=300):
        self.opt.zero_grad()

        # ---- (1) 边界条件损失: C(z=0, T, P, R) = C_atm ----
        # z=0 对所有气候条件成立，重复气候数据至 N_bc 行
        z_bc = torch.zeros(N_bc, 1, device=self.device)
        n = len(self.climate_obs)
        repeats = (N_bc + n - 1) // n
        climate_bc = self.climate_obs.repeat(repeats, 1)[:N_bc].clone()
        x_bc = torch.cat([z_bc, climate_bc], dim=1)
        C_bc = self.pinn(x_bc)
        loss_bc = torch.mean((C_bc - self.atm_conc) ** 2)

        # ---- (2) 数据损失 ----
        x_obs = torch.cat([self.z_obs, self.climate_obs], dim=1)
        C_pred = self.pinn(x_obs)
        loss_data = torch.mean((C_pred - self.C_obs) ** 2)

        # ---- 总损失 ----
        loss = 100.0 * loss_data + 10.0 * loss_bc
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.pinn.parameters(), 1.0)
        self.opt.step()
        self.sched.step(loss_data + loss_bc)

        return {
            'total': loss.item(),
            'data': loss_data.item(),
            'bc': loss_bc.item(),
        }


# =====================================================================
# 3. 主程序
# =====================================================================
def main():
    print('=' * 60)
    print('REE PINN v9 — Climate-Aware Multi-Profile Integration')
    print('=' * 60)

    # ---- 加载数据 ----
    df = pd.read_csv(DATA_PATH)
    df_lit = pd.read_csv(LIT_PATH)

    # 合并母岩信息
    df = df.merge(df_lit[['id', 'parent_rock']], left_on='literature_id',
                  right_on='id', how='left', suffixes=('', '_lit'))
    df['parent_rock'] = df['parent_rock'].fillna(df['Bedrock'])

    # 去掉无深度的样品（Wang2024 的 6 个层位均值）
    df = df.dropna(subset=['Depth_m']).reset_index(drop=True)
    print(f'\nData: {len(df)} samples from {df["literature_id"].nunique()} profiles')

    # 归一化
    df['z_norm'] = df['Depth_m'] / Z_MAX
    df['T_norm'] = (df['T_annual_mean_K'] - T_MEAN) / T_STD
    df['P_norm'] = (df['P_annual_mean_mm_yr'] - P_MEAN) / P_STD
    df['R_norm'] = (df['runoff_m_yr'] - R_MEAN) / R_STD

    # 边界条件：所有 0~1m 的表层样本均值
    surf = df[df['Depth_m'] < 1.0]
    atm_conc = float(surf['TotalREE_ppm'].mean())
    print(f'Boundary condition C_atm = {atm_conc:.1f} ppm (surface mean, depth<1m, n={len(surf)})')

    # ---- 准备张量 ----
    z_obs = torch.tensor(df['z_norm'].values, dtype=torch.float32).reshape(-1, 1)
    climate_obs = torch.tensor(
        df[['T_norm', 'P_norm', 'R_norm']].values, dtype=torch.float32)
    C_obs = torch.tensor(df['TotalREE_ppm'].values, dtype=torch.float32).reshape(-1, 1)

    # ---- 建网络 ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    pinn = REE_PINN(hidden_dim=64, n_layers=5).to(device)
    trainer = Trainer(pinn, atm_conc, z_obs, climate_obs, C_obs)

    print(f'Network: REE_PINN(4-64-64-64-64-64-1)')
    n_params = sum(p.numel() for p in pinn.parameters())
    print(f'Parameters: {n_params:,}')

    # ---- 训练循环 ----
    N_ITER = 5000
    log_every = 500
    print(f'\nTraining for {N_ITER} iterations...')
    print(f'{"Iter":>6}  {"Loss":>10}  {"Data":>10}  {"BC":>10}')

    history = []
    for i in range(N_ITER + 1):
        metrics = trainer.step()
        history.append(metrics)

        if i % log_every == 0 or i < 10:
            print(f'{i:>6}  {metrics["total"]:>10.4f}  '
                  f'{metrics["data"]:>10.4f}  {metrics["bc"]:>10.4f}')

    # ---- 评估 ----
    pinn.eval()
    with torch.no_grad():
        x_obs = torch.cat([z_obs, climate_obs], dim=1).to(device)
        C_pred = pinn(x_obs).cpu().reshape(-1).numpy()

    C_obs_np = df['TotalREE_ppm'].values
    r2 = np.corrcoef(C_obs_np, C_pred)[0, 1] ** 2
    rmse = np.sqrt(np.mean((C_pred - C_obs_np) ** 2))
    mae = np.mean(np.abs(C_pred - C_obs_np))

    print(f'\n=== Global Metrics ===')
    print(f'R²  = {r2:.4f}')
    print(f'RMSE = {rmse:.1f} ppm')
    print(f'MAE  = {mae:.1f} ppm')

    # ---- 按文献分组评估 ----
    df['_C_pred'] = C_pred
    df['_resid'] = C_pred - C_obs_np

    print(f'\n=== Per-Profile R² ===')
    for lid, grp in df.groupby('literature_id'):
        r2_p = np.corrcoef(grp['TotalREE_ppm'], grp['_C_pred'])[0, 1] ** 2
        rmse_p = np.sqrt(np.mean(grp['_resid'] ** 2))
        t = grp['T_annual_mean_K'].iloc[0] - 273.15
        p = grp['P_annual_mean_mm_yr'].iloc[0]
        r = grp['runoff_m_yr'].iloc[0]
        print(f'  [{int(lid)}] T={t:.1f}C P={p:.0f}mm R={r:.2f}m/yr  '
              f'R²={r2_p:.3f}  RMSE={rmse_p:.0f} ppm  '
              f'(n={len(grp)}, pred={grp["_C_pred"].min():.0f}~{grp["_C_pred"].max():.0f})')

    # ---- 保存 ----
    save_dir = '/Users/suheng/Desktop/claudecode/pinns/ree_pinns/v9'
    torch.save(pinn.state_dict(), f'{save_dir}/model_v9.pt')
    print(f'\nModel saved: {save_dir}/model_v9.pt')

    # ---- 绘图 ----
    plot_results(df, history, r2, rmse, pinn)

    return pinn, df, history


# =====================================================================
# 4. 绘图
# =====================================================================
def plot_results(df, history, r2_global, rmse_global, pinn):
    save_dir = '/Users/suheng/Desktop/claudecode/pinns/ree_pinns/v9'
    pinn.eval()

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ---- 图1: 全局 1:1 散点 ----
    ax = fig.add_subplot(gs[0, 0])
    colors = plt.cm.tab10(np.linspace(0, 1, df['literature_id'].nunique()))
    for (lid, grp), c in zip(df.groupby('literature_id'), colors):
        ax.scatter(grp['TotalREE_ppm'], grp['_C_pred'],
                   c=[c], s=30, alpha=0.7, label=f'[{int(lid)}] {grp["parent_rock"].iloc[0][:20]}')
    m = max(df['TotalREE_ppm'].max(), df['_C_pred'].max())
    ax.plot([0, m], [0, m], 'k--', lw=1.5)
    ax.set_xlabel('Observed TotalREE (ppm)')
    ax.set_ylabel('Predicted TotalREE (ppm)')
    ax.set_title(f'Global 1:1 Plot\nR²={r2_global:.3f}  RMSE={rmse_global:.0f} ppm')
    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.legend(fontsize=6, loc='upper left')

    # ---- 图2: 残差 vs 实测值 ----
    ax = fig.add_subplot(gs[0, 1])
    for (lid, grp), c in zip(df.groupby('literature_id'), colors):
        ax.scatter(grp['TotalREE_ppm'], grp['_resid'], c=[c], s=30, alpha=0.7)
    ax.axhline(0, color='k', ls='--', lw=1.5)
    ax.set_xlabel('Observed TotalREE (ppm)')
    ax.set_ylabel('Residual (pred - obs)')
    ax.set_title('Residual Analysis')

    # ---- 图3: 训练曲线 ----
    ax = fig.add_subplot(gs[0, 2])
    losses = np.array([h['total'] for h in history])
    ax.semilogy(losses, lw=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Total Loss')
    ax.set_title('Training Curve')
    ax.grid(True, alpha=0.3)

    # ---- 图4-9: 各剖面深度剖面图 ----
    lit_names = {
        1: 'Nagasawa (granite, T=13C)',
        2: 'Li2019 (A-type granite, T=18C)',
        3: 'Li&Zhou (A-type granite, T=18C)',
        4: 'Fu2019 (rhyolite, T=20C)',
        5: 'Yaraghi (two-mica granite, T=24C)',
        6: 'Luo (diorite, T=22C)',
    }

    profiles = df['literature_id'].unique()
    for idx, lid in enumerate(sorted(profiles)):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row + 1, col])
        grp = df[df['literature_id'] == lid].sort_values('Depth_m')

        z_max = grp['Depth_m'].max()
        z_smooth = np.linspace(0, z_max, 200)

        # 用平均气候条件生成预测曲线
        T_avg = grp['T_norm'].mean()
        P_avg = grp['P_norm'].mean()
        R_avg = grp['R_norm'].mean()
        z_n = torch.tensor(z_smooth / Z_MAX, dtype=torch.float32).reshape(-1, 1)
        clim = torch.tensor([[T_avg, P_avg, R_avg]] * len(z_smooth), dtype=torch.float32)
        x_in = torch.cat([z_n, clim], dim=1)

        with torch.no_grad():
            C_smooth = pinn(x_in).cpu().numpy().ravel()

        ax.scatter(grp['TotalREE_ppm'], grp['Depth_m'],
                   c='red', s=30, alpha=0.8, label='Obs', zorder=5)
        ax.plot(C_smooth, z_smooth, 'b-', lw=2, label='PINN v9')
        ax.set_xlabel('TotalREE (ppm)')
        ax.set_ylabel('Depth (m)')
        ax.invert_yaxis()
        name = lit_names.get(int(lid), f'[{int(lid)}]')
        r2_p = np.corrcoef(grp['TotalREE_ppm'], grp['_C_pred'])[0, 1] ** 2
        ax.set_title(f'{name}\nR²={r2_p:.3f}')
        ax.legend(fontsize=7)
        ax.set_xlim(0)

    plt.suptitle(f'REE PINN v9 — Climate-Aware Multi-Profile (n={len(df)})\n'
                 f'Inputs: z, T, P, R | R²={r2_global:.3f} | RMSE={rmse_global:.0f} ppm',
                 fontsize=14)
    plt.tight_layout()
    out = f'{save_dir}/results_v9.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Figure: {out}')


def model_device():
    """返回模型所在设备（从保存的权重推断）"""
    sd = torch.load(
        '/Users/suheng/Desktop/claudecode/pinns/ree_pinns/v9/model_v9.pt',
        map_location='cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    main()
