"""
REE PINN v10 - Per-Profile Equal-Weight Training
================================================
核心改动：每个剖面同等权重，而非每条样品同等权重。

v9 问题：
- Nagasawa 有 42 条样品，其他合计 58 条
- 样品级平均损失让样品多的剖面主导训练

v10 方案：
- 损失按剖面计算，然后取平均
- 每个剖面不管有多少样品，权重完全相等

输入: [z_norm, T_norm, P_norm, R_norm] (4D)
输出: TotalREE (ppm)
损失: mean(L_profile) across all profiles
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

DATA = '/Users/suheng/Desktop/claudecode/pinns/ree_pinns/data/REE_samples_20260424_with_climate.csv'
LIT   = '/Users/suheng/Desktop/claudecode/pinns/ree_pinns/data/REE_literature_20260424.csv'

Z_MAX  = 33.0
T_MEAN, T_STD = 290.62, 3.78
P_MEAN, P_STD = 1604.84, 710.27
R_MEAN, R_STD = 0.5210, 0.2310
C_SCALE = 4000.0


# =====================================================================
# 网络（和 v9 相同）
# =====================================================================
class SineLayer(nn.Module):
    def __init__(self, inf, outf, first=False, omega=30):
        super().__init__()
        self.omega = omega
        self.l = nn.Linear(inf, outf)
        nn.init.constant_(self.l.bias, 0.0)
        if first:
            nn.init.uniform_(self.l.weight, -1.0 / inf, 1.0 / inf)
        else:
            nn.init.uniform_(self.l.weight,
                -np.sqrt(6/inf)/omega, np.sqrt(6/inf)/omega)

    def forward(self, x):
        return torch.sin(self.omega * self.l(x))


class REE_PINN(nn.Module):
    def __init__(self, h=64, n=5):
        super().__init__()
        self.net = nn.Sequential(
            SineLayer(4, h, True),
            *[SineLayer(h, h) for _ in range(n-1)]
        )
        self.head = nn.Sequential(nn.Linear(h, 1), nn.ReLU6())

    def forward(self, x):
        if x.dim() == 1:
            x = x.reshape(1, -1)
        return self.head(self.net(x)) * C_SCALE


# =====================================================================
# Trainer（按剖面平均损失）
# =====================================================================
class Trainer:
    def __init__(self, pinn, profile_data, atm_global):
        """
        profile_data: dict {literature_id: {
            'z': tensor(N,1),
            'clim': tensor(N,3),
            'C': tensor(N,1),
            'atm': float   # 该剖面表层均值
        }}
        atm_global: 全局 C_atm（所有表层样本均值）
        """
        self.pinn = pinn
        self.profile_data = profile_data
        self.atm_global = float(atm_global)
        self.device = next(pinn.parameters()).device

        self.opt = torch.optim.Adam(pinn.parameters(), lr=1e-3)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, patience=500, factor=0.5, min_lr=1e-6)

    def step(self, N_bc=200):
        self.opt.zero_grad()

        # ---- (1) 边界条件损失 ----
        # 对每个剖面用其专属 C_atm
        z_bc = torch.zeros(N_bc, 1, device=self.device)
        lbc_list = []
        for pid, d in self.profile_data.items():
            n = len(d['z'])
            reps = (N_bc + n - 1) // n
            clim_bc = d['clim'].to(self.device).repeat(reps, 1)[:N_bc]
            x_bc = torch.cat([z_bc, clim_bc], dim=1)
            C_bc = self.pinn(x_bc)
            lbc_list.append(torch.mean((C_bc - d['atm']) ** 2))
        loss_bc = torch.stack(lbc_list).mean()

        # ---- (2) 数据损失（按剖面平均） ----
        loss_data_list = []
        for pid, d in self.profile_data.items():
            x = torch.cat([d['z'].to(self.device), d['clim'].to(self.device)], dim=1)
            C_pred = self.pinn(x)
            loss_data_list.append(torch.mean((C_pred - d['C'].to(self.device)) ** 2))
        loss_data = torch.stack(loss_data_list).mean()

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
# 主程序
# =====================================================================
def main():
    print('=' * 60)
    print('REE PINN v10 — Per-Profile Equal-Weight Training')
    print('=' * 60)

    df = pd.read_csv(DATA)
    dfl = pd.read_csv(LIT)
    df = df.merge(dfl[['id', 'parent_rock']], left_on='literature_id',
                  right_on='id', how='left', suffixes=('', '_l'))
    df.parent_rock = df.parent_rock.fillna(df.Bedrock)
    df = df.dropna(subset=['Depth_m']).reset_index(drop=True)

    df['zn'] = df.Depth_m / Z_MAX
    df['Tn'] = (df.T_annual_mean_K - T_MEAN) / T_STD
    df['Pn'] = (df.P_annual_mean_mm_yr - P_MEAN) / P_STD
    df['Rn'] = (df.runoff_m_yr - R_MEAN) / R_STD

    # 全局 C_atm
    surf = df[df.Depth_m < 1.0]
    atm_global = float(surf.TotalREE_ppm.mean())

    # 按剖面组织数据
    profile_data = {}
    for lid, grp in df.groupby('literature_id'):
        z = torch.tensor(grp.zn.values, dtype=torch.float32).reshape(-1, 1)
        c = torch.tensor(grp[['Tn', 'Pn', 'Rn']].values, dtype=torch.float32)
        C = torch.tensor(grp.TotalREE_ppm.values, dtype=torch.float32).reshape(-1, 1)

        # 该剖面的表层均值
        s = grp[grp.Depth_m < 1.0]
        atm_p = float(s.TotalREE_ppm.mean()) if len(s) > 0 else float(grp.TotalREE_ppm.iloc[:2].mean())

        profile_data[int(lid)] = {
            'z': z, 'clim': c, 'C': C, 'atm': atm_p,
            'name': f"[{int(lid)}] {grp.parent_rock.iloc[0][:15]}",
            'T_C': grp.T_annual_mean_K.iloc[0] - 273.15,
            'P_mm': grp.P_annual_mean_mm_yr.iloc[0],
            'R_m': grp.runoff_m_yr.iloc[0],
            'Tn_mean': float(grp['Tn'].mean()),
            'Pn_mean': float(grp['Pn'].mean()),
            'Rn_mean': float(grp['Rn'].mean()),
        }

    n_profiles = len(profile_data)
    n_samples = len(df)
    print(f'\nData: {n_samples} samples, {n_profiles} profiles')
    for pid, d in profile_data.items():
        print(f"  {d['name']}: {len(d['z'])} samples, "
              f"T={d['T_C']:.1f}C P={d['P_mm']:.0f}mm R={d['R_m']:.2f}m/yr, "
              f"C_atm={d['atm']:.1f}ppm")

    # 建网络
    device = torch.device('cpu')
    pinn = REE_PINN(64, 5).to(device)
    tr = Trainer(pinn, profile_data, atm_global)

    print(f'\nDevice: {device}')
    print(f'Parameters: {sum(p.numel() for p in pinn.parameters()):,}')

    # 训练
    H = []
    N_ITER = 5000
    print(f'\nTraining {N_ITER} iterations (per-profile weighted)...')
    print(f'{"Iter":>6}  {"Total":>12}  {"Data":>12}  {"BC":>12}')

    for i in range(N_ITER + 1):
        m = tr.step()
        H.append(m)
        if i % 500 == 0 or i < 10:
            print(f'{i:>6}  {m["total"]:>12.4f}  '
                  f'{m["data"]:>12.4f}  {m["bc"]:>12.4f}')

    # ---- 评估 ----
    pinn.eval()
    results = {}
    for pid, d in profile_data.items():
        x = torch.cat([d['z'], d['clim']], dim=1)
        with torch.no_grad():
            Cp = pinn(x).numpy().ravel()
        Co = d['C'].numpy().ravel()
        r2 = np.corrcoef(Co, Cp)[0, 1] ** 2
        rmse = np.sqrt(np.mean((Cp - Co) ** 2))
        mae = np.mean(np.abs(Cp - Co))
        results[pid] = {'Cp': Cp, 'Co': Co, 'r2': r2, 'rmse': rmse, 'mae': mae}

    # 全局
    Cp_all = np.concatenate([results[pid]['Cp'] for pid in results])
    Co_all = np.concatenate([results[pid]['Co'] for pid in results])
    r2_global = np.corrcoef(Co_all, Cp_all)[0, 1] ** 2
    rmse_global = np.sqrt(np.mean((Cp_all - Co_all) ** 2))

    print(f'\n=== Global Metrics ===')
    print(f'R² = {r2_global:.4f}  RMSE = {rmse_global:.1f} ppm')

    print(f'\n=== Per-Profile Metrics ===')
    for pid, d in profile_data.items():
        r = results[pid]
        status = 'OK' if r['r2'] > 0.9 else 'WARN' if r['r2'] > 0.5 else 'FAIL'
        print(f'  [{status}] {d["name"]}  R²={r["r2"]:.3f}  '
              f'RMSE={r["rmse"]:.0f}ppm  MAE={r["mae"]:.0f}ppm')

    # 绘图
    plot(pinn, profile_data, results, H, r2_global, rmse_global, atm_global)

    # 保存
    sd = '/Users/suheng/Desktop/claudecode/pinns/ree_pinns/v10'
    torch.save(pinn.state_dict(), f'{sd}/model_v10.pt')
    print(f'\nSaved: {sd}/model_v10.pt')
    return pinn, profile_data, results


# =====================================================================
# 绘图
# =====================================================================
def plot(pinn, profile_data, results, H, r2, rmse, atm_global):
    sd = '/Users/suheng/Desktop/claudecode/pinns/ree_pinns/v10'
    n = len(profile_data)

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ---- 图1: 全局 1:1 ----
    ax = fig.add_subplot(gs[0, 0])
    cs = plt.cm.tab10(np.linspace(0, 1, n))
    for (pid, r), c in zip(results.items(), cs):
        ax.scatter(r['Co'], r['Cp'], c=[c], s=40, alpha=0.7,
                   label=f'{profile_data[pid]["name"]}')
    m = max(np.concatenate([r['Co'] for r in results.values()]).max(),
             np.concatenate([r['Cp'] for r in results.values()]).max())
    ax.plot([0, m], [0, m], 'k--', lw=1.5)
    ax.set_xlabel('Observed (ppm)')
    ax.set_ylabel('Predicted (ppm)')
    ax.set_title(f'Global 1:1  R²={r2:.3f}  RMSE={rmse:.0f}ppm')
    ax.set_xlim(0); ax.set_ylim(0)
    ax.legend(fontsize=6)

    # ---- 图2: 残差 ----
    ax = fig.add_subplot(gs[0, 1])
    for (pid, r), c in zip(results.items(), cs):
        ax.scatter(r['Co'], r['Cp'] - r['Co'], c=[c], s=40, alpha=0.7)
    ax.axhline(0, color='k', ls='--', lw=1.5)
    ax.set_xlabel('Observed (ppm)')
    ax.set_ylabel('Residual (pred - obs)')
    ax.set_title('Residual Analysis')

    # ---- 图3: 训练曲线 ----
    ax = fig.add_subplot(gs[0, 2])
    losses = np.array([h['total'] for h in H])
    ax.semilogy(losses, lw=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Total Loss')
    ax.set_title('Training Curve (per-profile)')
    ax.grid(alpha=0.3)

    # ---- 图4-9: 各剖面 ----
    names = {
        1: 'Nagasawa\ngranite\nT=13C',
        2: 'Li2019\nA-type granite\nT=18C',
        3: 'Li&Zhou\nA-type granite\nT=18C',
        4: 'Fu2019\nrhyolite\nT=20C',
        5: 'Yaraghi\n2-mica granite\nT=24C',
        6: 'Luo\ndiorite\nT=22C',
    }
    profile_ids = sorted(profile_data.keys())
    for idx, pid in enumerate(profile_ids):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row + 1, col])

        d = profile_data[pid]
        r = results[pid]

        # 预测曲线（平均气候）
        Ta = d['Tn_mean']
        Pa = d['Pn_mean']
        Ra = d['Rn_mean']
        zv = np.linspace(0, d['z'].max() * Z_MAX, 200)
        zn = torch.tensor(zv / Z_MAX, dtype=torch.float32).reshape(-1, 1)
        clim = torch.tensor([[Ta, Pa, Ra]] * len(zv), dtype=torch.float32)
        x_in = torch.cat([zn, clim], dim=1)
        with torch.no_grad():
            Cv = pinn(x_in).numpy().ravel()

        # 剖面曲线
        ax.scatter(d['C'].numpy(), d['z'].numpy() * Z_MAX,
                   c='red', s=40, alpha=0.8, label='Obs', zorder=5)
        ax.plot(Cv, zv, 'b-', lw=2, label='PINN v10')

        # 边界线
        ax.axhline(0, color='gray', ls=':', lw=1)
        ax.axhline(1, color='gray', ls=':', lw=1)

        ax.invert_yaxis()
        ax.set_xlabel('TotalREE (ppm)')
        ax.set_ylabel('Depth (m)')
        ax.set_title(f'{names.get(pid, str(pid))}\n'
                     f'R²={r["r2"]:.3f}  RMSE={r["rmse"]:.0f}ppm  n={len(d["z"])}')
        ax.legend(fontsize=8)
        ax.set_xlim(0)

    plt.suptitle(f'REE PINN v10 — Per-Profile Equal-Weight  (n={len(profile_data)} profiles)\n'
                 f'R²={r2:.3f}  RMSE={rmse:.0f} ppm', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{sd}/results_v10.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Figure: {sd}/results_v10.png')


if __name__ == '__main__':
    main()
