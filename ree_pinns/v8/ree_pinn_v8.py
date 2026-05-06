"""
REE PINN v8 - Per-Profile Modeling
====================================
按文献分组建模，每个剖面独立训练。

根因分析：
- v2~v7 用全局归一化 C_scale，但6篇文献的 REE 浓度范围差异巨大
  (57-305 ppm vs 373-1360 ppm)，导致网络无法同时拟合所有数据
- 各剖面的深度-Relationship 不同，不应该用统一模型
- IC loss 对稳态问题无物理意义

v8 方案：
1. 按 literature_id 分组，每个剖面独立 C_scale 归一化
2. 只用数据损失 + 边界条件损失，删除无意义的 IC loss
3. 1D SIREN 网络，输入 z' = z / z_max，输出 ppm
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

DATA_PATH = '/Users/suheng/Desktop/claudecode/pinns/ree_pinns/data/REE_samples_20260424.csv'


# =====================================================================
# 1. 网络架构（SIREN）
# =====================================================================
class SineLayer(nn.Module):
    """SIREN 的核心：sin(omega * Linear(x))"""
    def __init__(self, in_features, out_features, is_first=False, omega=30):
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(in_features, out_features)
        # SIREN 初始化：first layer 用小方差，后续层用 0.5/omega
        nn.init.constant_(self.linear.bias, 0.0)
        nn.init.normal_(self.linear.weight, 0.0, 0.5 / self.omega)

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


class REE_PINN(nn.Module):
    """
    1D SIREN，稳态 REE 浓度预测

    输入: z' = z / z_max (归一化深度，范围 [0, 1])
    输出: LREE (ppm), HREE (ppm)

    网络结构: z' → [Sine(Linear)] × 4 → [Linear + Sigmoid] → C_scale × ppm
    Sigmoid 限制输出在 [0, C_scale] 范围内，自动适配不同剖面的浓度量级
    """
    def __init__(self, C_scale_L=500.0, C_scale_H=200.0,
                 hidden_dim=64, n_layers=4):
        super().__init__()
        self.C_scale_L = float(C_scale_L)
        self.C_scale_H = float(C_scale_H)

        layers = [SineLayer(1, hidden_dim, is_first=True)]
        for _ in range(n_layers - 1):
            layers.append(SineLayer(hidden_dim, hidden_dim))
        self.net = nn.Sequential(*layers)

        self.head_L = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.head_H = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, z_prime):
        """输入归一化深度，输出 ppm 浓度"""
        if z_prime.dim() == 1:
            z_prime = z_prime.reshape(-1, 1)
        h = self.net(z_prime)
        C_L = self.head_L(h) * self.C_scale_L
        C_H = self.head_H(h) * self.C_scale_H
        return C_L, C_H


# =====================================================================
# 2. 训练器
# =====================================================================
class Trainer:
    """
    训练器，只包含两个有物理意义的损失项：

    (1) 数据损失: MSE(预测浓度, 实测浓度)
    (2) 边界条件损失: MSE(C(z=0), C_atm)
       C_atm = 表层样本(< 1m)的实测均值

    注意：删除了 IC loss，因为：
    - IC loss 要求 C(z, t=0) = 0
    - 但对稳态问题(t→∞)，这个条件没有物理意义
    - IC loss 会和 BC / Data 冲突，导致训练不稳定
    """
    def __init__(self, pinn, atm_L, atm_H, obs_z, obs_CL, obs_CH):
        self.pinn = pinn
        self.atm_L = float(atm_L)   # 表层 LREE 均值（边界条件）
        self.atm_H = float(atm_H)
        self.device = next(pinn.parameters()).device

        self.obs_z = obs_z.to(self.device)
        self.obs_CL = obs_CL.to(self.device)
        self.obs_CH = obs_CH.to(self.device)

        self.opt = torch.optim.Adam(pinn.parameters(), lr=1e-3)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, patience=500, factor=0.5, min_lr=1e-6)

    def step(self, N_bc=300):
        self.opt.zero_grad()

        # ---- (1) 边界条件损失: C(z=0) = C_atm ----
        z_top = torch.zeros(N_bc, 1, device=self.device)
        CL_top, CH_top = self.pinn(z_top)
        loss_bc = (torch.mean((CL_top - self.atm_L) ** 2)
                 + torch.mean((CH_top - self.atm_H) ** 2))

        # ---- (2) 实测数据损失 ----
        CL_pred, CH_pred = self.pinn(self.obs_z)
        loss_data = (torch.mean((CL_pred - self.obs_CL) ** 2)
                   + torch.mean((CH_pred - self.obs_CH) ** 2))

        # ---- 总损失 ----
        # data=100 为主，确保拟合观测值
        # bc=10 约束边界行为
        loss = 100.0 * loss_data + 10.0 * loss_bc
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.pinn.parameters(), 1.0)
        self.opt.step()
        self.sched.step(loss_data + loss_bc)

        return loss.item(), loss_data.item(), loss_bc.item()


# =====================================================================
# 3. 单个剖面训练
# =====================================================================
def train_profile(depth, LREE, HREE, n_iter=3000):
    """
    训练单个剖面的 PINN

    参数:
        depth: 深度数组 (m)
        LREE, HREE: 浓度数组 (ppm)
        n_iter: 训练迭代次数

    步骤:
        1. 计算归一化坐标 z' = z / z_max
        2. 计算剖面专属的 C_scale = 1.1 × max(REE)
        3. 计算表层边界条件 C_atm = mean(REE[depth < 1m])
        4. 训练 SIREN 网络
    """
    z_max = float(np.nanmax(depth))
    z_norm = depth / z_max

    # 浓度尺度（剖面专属）
    C_scale_L = float(np.nanmax(LREE)) * 1.1
    C_scale_H = float(np.nanmax(HREE)) * 1.1

    # 边界条件（表层均值）
    surf_mask = depth < 1.0
    atm_L = float(np.nanmean(LREE[surf_mask])) if surf_mask.sum() > 0 else float(np.nanmean(LREE[:3]))
    atm_H = float(np.nanmean(HREE[surf_mask])) if surf_mask.sum() > 0 else float(np.nanmean(HREE[:3]))

    # 网络
    pinn = REE_PINN(C_scale_L=C_scale_L, C_scale_H=C_scale_H,
                   hidden_dim=64, n_layers=4)

    trainer = Trainer(
        pinn, atm_L, atm_H,
        torch.tensor(z_norm).float().reshape(-1, 1),
        torch.tensor(LREE).float().reshape(-1, 1),
        torch.tensor(HREE).float().reshape(-1, 1))

    # 训练循环
    for i in range(n_iter + 1):
        trainer.step()

    # 评估
    pinn.eval()
    with torch.no_grad():
        CL_pred, CH_pred = pinn(trainer.obs_z)
    CL_pred = CL_pred.reshape(-1).numpy()
    CH_pred = CH_pred.reshape(-1).numpy()

    r2_L = np.corrcoef(LREE.ravel(), CL_pred)[0, 1] ** 2
    r2_H = np.corrcoef(HREE.ravel(), CH_pred)[0, 1] ** 2
    rmse_L = np.sqrt(np.mean((CL_pred - LREE.ravel()) ** 2))
    rmse_H = np.sqrt(np.mean((CH_pred - HREE.ravel()) ** 2))

    return {
        'pinn': pinn,
        'pinn_state': pinn.state_dict(),
        'z_norm': z_norm,
        'z_max': z_max,
        'LREE_obs': LREE,
        'HREE_obs': HREE,
        'LREE_pred': CL_pred,
        'HREE_pred': CH_pred,
        'C_scale_L': C_scale_L,
        'C_scale_H': C_scale_H,
        'atm_L': atm_L,
        'atm_H': atm_H,
        'r2_L': r2_L,
        'r2_H': r2_H,
        'rmse_L': rmse_L,
        'rmse_H': rmse_H,
    }


# =====================================================================
# 4. 绘图
# =====================================================================
def plot_results(results, path):
    """绘制所有剖面的训练结果"""
    n = len(results)
    fig = plt.figure(figsize=(6 * n, 12))
    gs = GridSpec(2, n, figure=fig, wspace=0.35)

    for col, (lid, r) in enumerate(results.items()):
        z_max = r['z_max']
        z_test = np.linspace(0, z_max, 200)
        z_n = torch.tensor(z_test / z_max).float().reshape(-1, 1)

        pinn = REE_PINN(C_scale_L=r['C_scale_L'], C_scale_H=r['C_scale_H'])
        with torch.no_grad():
            CLt, CHt = pinn(z_n)
        CLt = CLt.reshape(-1).numpy()
        CHt = CHt.reshape(-1).numpy()

        # ---- 左图: LREE 剖面 ----
        ax = fig.add_subplot(gs[0, col])
        ax.scatter(r['LREE_obs'].ravel(), r['z_norm'] * z_max,
                   c='red', s=40, alpha=0.7, label='Observed', zorder=5)
        ax.plot(CLt, z_test, 'r-', lw=2, label='PINN v8')
        ax.set_xlabel('LREE (ppm)')
        ax.set_ylabel('Depth (m)')
        ax.set_title(f'{lid}\n'
                     f'R² = {r["r2_L"]:.3f}   RMSE = {r["rmse_L"]:.0f} ppm\n'
                     f'pred: {r["LREE_pred"].min():.0f}~{r["LREE_pred"].max():.0f} ppm\n'
                     f'obs:  {r["LREE_obs"].min():.0f}~{r["LREE_obs"].max():.0f} ppm')
        ax.invert_yaxis()
        ax.legend(fontsize=8)
        ax.set_xlim(0)

        # ---- 右图: 1:1 散点图 ----
        ax = fig.add_subplot(gs[1, col])
        ax.scatter(r['LREE_obs'].ravel(), r['LREE_pred'],
                   c='red', s=30, alpha=0.5, label='LREE', zorder=5)
        ax.scatter(r['HREE_obs'].ravel(), r['HREE_pred'],
                   c='blue', s=30, alpha=0.5, label='HREE', zorder=5)
        m = max(r['LREE_obs'].ravel().max(), r['LREE_pred'].max(),
                r['HREE_obs'].ravel().max(), r['HREE_pred'].max())
        ax.plot([0, m], [0, m], 'k--', lw=1, label='1:1 line')
        ax.set_xlabel('Observed (ppm)')
        ax.set_ylabel('Predicted (ppm)')
        ax.set_title(f'R²_L = {r["r2_L"]:.3f}   R²_H = {r["r2_H"]:.3f}')
        ax.set_xlim(0)
        ax.set_ylim(0)
        ax.legend(fontsize=8)

    plt.suptitle('REE PINN v8 - Per-Profile Modeling Results', fontsize=16)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Figure: {path}')


# =====================================================================
# 5. 主程序
# =====================================================================
def main():
    """
    流程:
    1. 加载 REE_samples CSV
    2. 按 literature_id 分组建模
    3. 逐剖面训练
    4. 绘图保存
    """
    print('=' * 60)
    print('REE PINN v8 - Per-Profile Modeling')
    print('=' * 60)

    # 加载数据
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=['Depth_m', 'LREE_ppm', 'HREE_ppm', 'literature_id'])
    print(f'\nData loaded: {len(df)} samples')

    # 训练每个剖面
    results = {}
    for lid, grp in df.groupby('literature_id', sort=True):
        depth = grp['Depth_m'].values.astype(np.float32)
        LREE = grp['LREE_ppm'].values.astype(np.float32)
        HREE = grp['HREE_ppm'].values.astype(np.float32)

        print(f'\n[{lid}] {len(depth)} samples, depth: {depth.min():.1f}~{depth.max():.1f}m')
        print(f'       LREE: {LREE.min():.0f}~{LREE.max():.0f} ppm')

        r = train_profile(depth, LREE, HREE, n_iter=3000)
        results[lid] = r

        print(f'       R²_L={r["r2_L"]:.3f}  R²_H={r["r2_H"]:.3f}  '
              f'pred: {r["LREE_pred"].min():.0f}~{r["LREE_pred"].max():.0f} ppm')

    # 汇总统计
    print('\n' + '=' * 60)
    print('Summary:')
    print('-' * 60)
    for lid, r in results.items():
        status = 'OK' if r['r2_L'] > 0.9 else 'WARN' if r['r2_L'] > 0.5 else 'FAIL'
        print(f'  [{status}] {lid}: R²_L={r["r2_L"]:.3f}  '
              f'R²_H={r["r2_H"]:.3f}  RMSE={r["rmse_L"]:.0f} ppm')
    avg_r2 = np.mean([r['r2_L'] for r in results.values()])
    print(f'\n  Average R²_L: {avg_r2:.3f}')
    print('=' * 60)

    # 绘图
    save_dir = '/Users/suheng/Desktop/claudecode/pinns/ree_pinns/v8'
    plot_results(results, f'{save_dir}/results_v8.png')

    # 保存模型
    model_dict = {lid: r['pinn_state'] for lid, r in results.items()}
    torch.save(model_dict, f'{save_dir}/model_v8.pt')
    print(f'Models saved: {save_dir}/model_v8.pt')

    return results


if __name__ == '__main__':
    main()
