"""
REE PINN v12 — LREE/HREE 分馏模型
======================================
目标：定量描述轻重稀土在风化过程中如何分馏，形成风化壳稀土富集。

核心物理：
  LREE (La-Sm) 和 HREE (Tb-Lu) 有不同的吸附-解吸特性。
  LREE 更倾向于被黏土矿物吸附（K_d^L > K_d^H），停留在浅层；
  HREE 更亲水，随渗流向下迁移。

  稳态方程（每种组分）：
    C_i(z) = C_i^atm * exp(-beta_i * z) + C_i^parent * (1 - exp(-beta_i * z))

  beta_i = f(T, P, R; theta_i) — 气候驱动的分馏系数
  LREE 和 HREE 有不同的 beta_i

损失函数：
  L_data:  拟合 C_LREE(z) 和 C_HREE(z)
  L_frac:  物理约束 beta_L > beta_H（分馏方向）
  L_bound: 边界条件（C at z=0, z->inf）

数据：
  从现有 TotalREE 数据中拆分 LREE/HREE（La~Sm = LREE, Tb~Lu = HREE）
  Y 单独建模或归入 HREE
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
SAVE  = '/Users/suheng/Desktop/claudecode/pinns/ree_pinns/v11'

Z_MAX = 33.0
T_MEAN, T_STD = 290.62, 3.78
P_MEAN, P_STD = 1604.84, 710.27
R_MEAN, R_STD = 0.5210, 0.2310

LREE_COLS = ['La_ppm', 'Ce_ppm', 'Pr_ppm', 'Nd_ppm', 'Sm_ppm']
HREE_COLS = ['Tb_ppm', 'Dy_ppm', 'Ho_ppm', 'Er_ppm', 'Tm_ppm', 'Yb_ppm', 'Lu_ppm']
Y_COL     = 'Y_ppm'


# =====================================================================
# 网络（双输出：LREE + HREE）
# =====================================================================
class SineLayer(nn.Module):
    def __init__(self, inf, outf, first=False, omega=30):
        super().__init__()
        self.omega = omega
        self.l = nn.Linear(inf, outf)
        nn.init.constant_(self.l.bias, 0.0)
        if first:
            nn.init.uniform_(self.l.weight, -1.0/inf, 1.0/inf)
        else:
            nn.init.uniform_(self.l.weight,
                -np.sqrt(6/inf)/omega, np.sqrt(6/inf)/omega)

    def forward(self, x):
        return torch.sin(self.omega * self.l(x))


class FractionationPINN(nn.Module):
    """
    输入：[z_norm, T_norm, P_norm, R_norm]
    输出：[C_LREE, C_HREE] 两个浓度 (ppm)
    """
    def __init__(self, h=64, n=5):
        super().__init__()
        self.net = nn.Sequential(
            SineLayer(4, h, first=True),
            *[SineLayer(h, h) for _ in range(n-1)]
        )
        self.head_L = nn.Sequential(
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, 1), nn.Softplus()
        )
        self.head_H = nn.Sequential(
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, 1), nn.Softplus()
        )

    def forward(self, x):
        if x.dim() == 1: x = x.reshape(1, -1)
        feat = self.net(x)
        CL = self.head_L(feat)
        CH = self.head_H(feat)
        return torch.cat([CL, CH], dim=1)  # (N, 2)


# =====================================================================
# 可学习分馏参数（气候驱动）
# =====================================================================
class FractionationParams(nn.Module):
    """
    beta_L(z) = beta_L0 * exp(bLT * T_n + bLP * P_n + bLR * R_n)
    beta_H(z) = beta_H0 * exp(bHT * T_n + bHP * P_n + bHR * R_n)

    物理约束：beta_L > beta_H（LREE吸附更强，更快趋于母岩值）
    通过 loss_frac 软约束强制 beta_L >= beta_H
    """
    def __init__(self):
        super().__init__()
        # 基准分馏系数（对数域）
        self.log_beta_L0 = nn.Parameter(torch.tensor(-1.0))  # beta_L0 ≈ 0.37
        self.log_beta_H0 = nn.Parameter(torch.tensor(-1.5))  # beta_H0 ≈ 0.22

        # 气候敏感系数
        self.bLT = nn.Parameter(torch.tensor(0.0))
        self.bLP = nn.Parameter(torch.tensor(0.0))
        self.bHR = nn.Parameter(torch.tensor(0.0))
        self.bHP = nn.Parameter(torch.tensor(0.0))

    def get_beta(self, T_norm, P_norm, R_norm):
        """返回 beta_L, beta_H（逐样本）"""
        bL0 = torch.exp(self.log_beta_L0).clamp(min=1e-6)
        bH0 = torch.exp(self.log_beta_H0).clamp(min=1e-6)

        bLT = torch.tanh(self.bLT) * 1.5
        bLP = torch.tanh(self.bLP) * 1.5
        bHR = torch.tanh(self.bHR) * 1.5
        bHP = torch.tanh(self.bHP) * 1.5

        beta_L = bL0 * torch.exp(bLT * T_norm + bLP * P_norm)
        beta_H = bH0 * torch.exp(bHP * P_norm + bHR * R_norm)

        return beta_L, beta_H


# =====================================================================
# 解析分馏曲线（物理基准）
# =====================================================================
def analytical_profile(z, C_atm, C_parent, beta):
    """
    稳态解析解：C(z) = C_atm * exp(-beta*z) + C_parent * (1 - exp(-beta*z))
    beta 越大：表层→母岩过渡越快（吸附越强）
    """
    return C_atm * np.exp(-beta * z) + C_parent * (1 - np.exp(-beta * z))


# =====================================================================
# Trainer
# =====================================================================
class FractionationTrainer:
    def __init__(self, pinn, params, profile_data):
        self.pinn   = pinn
        self.params = params
        self.profile_data = profile_data
        self.device = next(pinn.parameters()).device
        self.opt = torch.optim.AdamW(
            list(pinn.parameters()) + list(params.parameters()),
            lr=1e-3, weight_decay=1e-4
        )
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=5000, eta_min=1e-6
        )
        self.hist = {
            'total': [], 'data': [], 'frac': [], 'bound': [],
            'beta_L0': [], 'beta_H0': [], 'bLT': [], 'bLP': [], 'bHP': [], 'bHR': []
        }

    def step(self, epoch, n_phys=200):
        # 分阶段权重
        warmup = min(1.0, epoch / 2000.0)
        w_frac  = 1.0 * warmup
        w_bound = 10.0
        w_data  = 50.0

        self.opt.zero_grad()

        # (1) 数据损失（双输出：LREE + HREE）
        loss_data_L = []
        loss_data_H = []
        for pid, d in self.profile_data.items():
            x  = torch.cat([d['z'].to(self.device), d['clim'].to(self.device)], dim=1)
            Cp = self.pinn(x)  # (N, 2): [C_L, C_H]
            loss_data_L.append(torch.mean((Cp[:, 0:1] - d['CL'].to(self.device)) ** 2))
            loss_data_H.append(torch.mean((Cp[:, 1:2] - d['CH'].to(self.device)) ** 2))
        loss_data = (torch.stack(loss_data_L).mean() +
                     torch.stack(loss_data_H).mean())

        # (2) 分馏约束（beta_L >= beta_H 软约束）
        loss_frac = torch.tensor(0.0, device=self.device)
        for pid, d in self.profile_data.items():
            T_n = d['clim'][:, 0:1].to(self.device)
            P_n = d['clim'][:, 1:2].to(self.device)
            R_n = d['clim'][:, 2:3].to(self.device)
            bL, bH = self.params.get_beta(T_n, P_n, R_n)
            # 强制 beta_L >= beta_H
            violation = torch.relu(bH - bL)  # >0 表示违反约束
            loss_frac = loss_frac + torch.mean(violation ** 2)
        loss_frac = loss_frac / len(self.profile_data)

        # (3) 边界损失（表层和深处）
        loss_bound = torch.tensor(0.0, device=self.device)
        for pid, d in self.profile_data.items():
            # z=0: C(z=0) = C_atm
            z_bc0 = torch.zeros(50, 1, device=self.device)
            clim_bc = d['clim'].to(self.device)[:1].repeat(50, 1)
            C_bc0 = self.pinn(torch.cat([z_bc0, clim_bc], dim=1))
            loss_bound = loss_bound + torch.mean(
                (C_bc0 - d['CL_atm'].to(self.device)[:, :1]) ** 2
            )
            loss_bound = loss_bound + torch.mean(
                (C_bc0 - d['CH_atm'].to(self.device)[:, :1]) ** 2
            )
        loss_bound = loss_bound / len(self.profile_data) / 2

        # 总损失
        loss = w_data * loss_data + w_bound * loss_bound + w_frac * loss_frac
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.pinn.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.params.parameters(), 1.0)
        self.opt.step()
        self.sched.step()

        self.hist['total'].append(loss.item())
        self.hist['data'].append(loss_data.item())
        self.hist['frac'].append(loss_frac.item())
        self.hist['bound'].append(loss_bound.item())
        self.hist['beta_L0'].append(float(torch.exp(self.params.log_beta_L0).item()))
        self.hist['beta_H0'].append(float(torch.exp(self.params.log_beta_H0).item()))
        self.hist['bLT'].append(float(torch.tanh(self.params.bLT).item() * 1.5))
        self.hist['bLP'].append(float(torch.tanh(self.params.bLP).item() * 1.5))
        self.hist['bHP'].append(float(torch.tanh(self.params.bHP).item() * 1.5))
        self.hist['bHR'].append(float(torch.tanh(self.params.bHR).item() * 1.5))

        return loss.item(), loss_data.item(), loss_frac.item(), loss_bound.item()


# =====================================================================
# 主程序
# =====================================================================
def main():
    print('=' * 60)
    print('REE PINN v12 — LREE/HREE 分馏模型')
    print('=' * 60)

    df  = pd.read_csv(DATA)
    dfl = pd.read_csv(LIT)
    df = df.merge(dfl[['id', 'parent_rock']],
                  left_on='literature_id', right_on='id',
                  how='left', suffixes=('', '_l'))
    df.parent_rock = df.parent_rock.fillna(df.Bedrock)

    # 只保留有完整 LREE/HREE 数据的样品
    all_ree_cols = LREE_COLS + HREE_COLS + [Y_COL]
    df = df.dropna(subset=['Depth_m'] + all_ree_cols).reset_index(drop=True)

    # 计算 LREE, HREE, Ratio
    df['C_LREE'] = df[LREE_COLS].sum(axis=1)
    df['C_HREE'] = df[HREE_COLS].sum(axis=1) + df[Y_COL]
    df['L_H_ratio'] = df['C_LREE'] / (df['C_HREE'] + 1e-6)

    # 归一化
    df['zn'] = df.Depth_m / Z_MAX
    df['Tn'] = (df.T_annual_mean_K - T_MEAN) / T_STD
    df['Pn'] = (df.P_annual_mean_mm_yr - P_MEAN) / P_STD
    df['Rn'] = (df.runoff_m_yr - R_MEAN) / R_STD

    # 按剖面组织
    profile_data = {}
    for lid, grp in df.groupby('literature_id'):
        z    = torch.tensor(grp.zn.values, dtype=torch.float32).reshape(-1, 1)
        c    = torch.tensor(grp[['Tn', 'Pn', 'Rn']].values, dtype=torch.float32)
        CL   = torch.tensor(grp.C_LREE.values, dtype=torch.float32).reshape(-1, 1)
        CH   = torch.tensor(grp.C_HREE.values, dtype=torch.float32).reshape(-1, 1)
        RAT  = torch.tensor(grp['L_H_ratio'].values, dtype=torch.float32).reshape(-1, 1)

        # 表层浓度（深度 < 1m）
        surf = grp[grp.Depth_m < 1.0]
        CL_atm = float(surf.C_LREE.mean()) if len(surf) > 0 else float(grp.C_LREE.iloc[:2].mean())
        CH_atm = float(surf.C_HREE.mean()) if len(surf) > 0 else float(grp.C_HREE.iloc[:2].mean())

        # 母岩浓度（最深样品）
        CL_par = float(grp.C_LREE.iloc[-1])
        CH_par = float(grp.C_HREE.iloc[-1])

        profile_data[int(lid)] = {
            'z': z, 'clim': c,
            'CL': CL, 'CH': CH, 'RAT': RAT,
            'CL_atm': torch.tensor([[CL_atm]]),
            'CH_atm': torch.tensor([[CH_atm]]),
            'CL_parent': CL_par, 'CH_parent': CH_par,
            'name': f"[{int(lid)}] {grp.parent_rock.iloc[0][:18]}",
            'T_C': grp.T_annual_mean_K.iloc[0] - 273.15,
            'P_mm': grp.P_annual_mean_mm_yr.iloc[0],
            'R_m': grp.runoff_m_yr.iloc[0],
            'Tn_mean': float(grp['Tn'].mean()),
            'Pn_mean': float(grp['Pn'].mean()),
            'Rn_mean': float(grp['Rn'].mean()),
        }

    n_profiles = len(profile_data)
    print(f'\nData: {len(df)} samples, {n_profiles} profiles')
    for pid, d in profile_data.items():
        ratio = d['CL_parent'] / (d['CH_parent'] + 1e-6)
        print(f"  {d['name']}: n={len(d['z'])}  "
              f"T={d['T_C']:.1f}C  P={d['P_mm']:.0f}mm  "
              f"CL_par={d['CL_parent']:.0f}ppm  CH_par={d['CH_parent']:.0f}ppm  "
              f"L/H_parent={ratio:.2f}")

    # 建网
    device = torch.device('cpu')
    pinn   = FractionationPINN(64, 5).to(device)
    params = FractionationParams().to(device)
    tr     = FractionationTrainer(pinn, params, profile_data)

    n_params = sum(p.numel() for p in pinn.parameters())
    print(f'\nNetwork params: {n_params:,}')
    print(f'Output: [C_LREE, C_HREE] (ppm)')
    print(f'\nTraining 5000 iterations...')
    print(f'{"Iter":>6}  {"Total":>12}  {"Data":>12}  {"Frac":>12}  {"Bound":>12}')

    for i in range(5001):
        loss, ld, lf, lb = tr.step(i, n_phys=200)
        if i % 500 == 0 or i < 10:
            bL0 = float(torch.exp(params.log_beta_L0).item())
            bH0 = float(torch.exp(params.log_beta_H0).item())
            w   = min(1.0, i / 2000.0)
            print(f'{i:>6}  {loss:>12.4f}  {ld:>12.4f}  '
                  f'{lf:>12.4f}  {lb:>12.4f}')
            print(f'       beta_L0={bL0:.4f}  beta_H0={bH0:.4f}  '
                  f'ratio={bL0/bH0:.3f}  w_frac={w:.2f}')

    # 评估
    pinn.eval()
    results = {}
    for pid, d in profile_data.items():
        x = torch.cat([d['z'], d['clim']], dim=1)
        with torch.no_grad():
            Cp = pinn(x).numpy()  # (N, 2)
        CL_obs = d['CL'].numpy().ravel()
        CH_obs = d['CH'].numpy().ravel()
        RAT_obs = d['RAT'].numpy().ravel()
        RAT_pred = (Cp[:, 0] / (Cp[:, 1] + 1e-6))

        r2_CL = np.corrcoef(CL_obs, Cp[:, 0])[0, 1] ** 2
        r2_CH = np.corrcoef(CH_obs, Cp[:, 1])[0, 1] ** 2
        r2_R  = np.corrcoef(RAT_obs, RAT_pred)[0, 1] ** 2
        rmse_L = np.sqrt(np.mean((Cp[:, 0] - CL_obs) ** 2))
        rmse_H = np.sqrt(np.mean((Cp[:, 1] - CH_obs) ** 2))
        results[pid] = dict(
            CL=Cp[:, 0], CH=Cp[:, 1],
            CL_obs=CL_obs, CH_obs=CH_obs,
            RAT_obs=RAT_obs, RAT_pred=RAT_pred,
            r2_CL=r2_CL, r2_CH=r2_CH, r2_R=r2_R,
            rmse_L=rmse_L, rmse_H=rmse_H
        )

    print(f'\n=== Fractionation Results ===')
    for pid, d in profile_data.items():
        r = results[pid]
        print(f'  [{pid}] {d["name"]}')
        print(f'       LREE: R²={r["r2_CL"]:.3f}  RMSE={r["rmse_L"]:.0f}ppm')
        print(f'       HREE: R²={r["r2_CH"]:.3f}  RMSE={r["rmse_H"]:.0f}ppm')
        print(f'       L/H:  R²={r["r2_R"]:.3f}')

    # 解析解对比
    print(f'\n=== Learned Fractionation Coefficients ===')
    bL0 = float(torch.exp(params.log_beta_L0).item())
    bH0 = float(torch.exp(params.log_beta_H0).item())
    bLT = float(torch.tanh(params.bLT).item() * 1.5)
    bLP = float(torch.tanh(params.bLP).item() * 1.5)
    bHP = float(torch.tanh(params.bHP).item() * 1.5)
    bHR = float(torch.tanh(params.bHR).item() * 1.5)
    print(f'  beta_L = {bL0:.4f} * exp({bLT:.3f}*T_n + {bLP:.3f}*P_n)')
    print(f'  beta_H = {bH0:.4f} * exp({bHP:.3f}*P_n + {bHR:.3f}*R_n)')
    print(f'  beta_L/beta_H ratio = {bL0/bH0:.3f}  '
          f'(>1 means LREE accumulates shallower than HREE)')

    plot(pinn, params, profile_data, results, tr.hist, SAVE)

    import os
    os.makedirs(SAVE, exist_ok=True)
    torch.save({
        'pinn': pinn.state_dict(),
        'params': params.state_dict()
    }, f'{SAVE}/model_v12.pt')
    print(f'\nSaved: {SAVE}/model_v12.pt')


# =====================================================================
# 绘图
# =====================================================================
def plot(pinn, params, profile_data, results, H, save_dir):
    import os
    n = len(profile_data)
    fig = plt.figure(figsize=(18, 16))
    gs  = GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)
    colors = plt.cm.tab10(np.linspace(0, 1, n))

    names = {
        1: 'Nagasawa\n花岗岩\nT=13C',
        2: 'Li2019\nA型花岗岩\nT=18C',
        3: 'Li&Zhou\nA型花岗岩\nT=18C',
        4: 'Fu2019\n流纹岩\nT=20C',
        5: 'Yaraghi\n二云母花岗岩\nT=24C',
        6: 'Luo\n闪长岩\nT=22C',
    }

    # Row 1: LREE profiles
    for idx, pid in enumerate(sorted(profile_data.keys())):
        row = idx // 3; col = idx % 3
        ax  = fig.add_subplot(gs[row, col])
        d = profile_data[pid]; r = results[pid]

        Ta = d['Tn_mean']; Pa = d['Pn_mean']; Ra = d['Rn_mean']
        zv = np.linspace(0, float(d['z'].max()) * Z_MAX, 200)
        zn = torch.tensor(zv / Z_MAX, dtype=torch.float32).reshape(-1, 1)
        clim = torch.tensor([[Ta, Pa, Ra]] * len(zv), dtype=torch.float32)
        with torch.no_grad():
            Cp = pinn(torch.cat([zn, clim], dim=1)).numpy()

        # 解析解（对比）
        CL_anal = analytical_profile(zv, d['CL_atm'].item(), d['CL_parent'], 0.15)
        CH_anal = analytical_profile(zv, d['CH_atm'].item(), d['CH_parent'], 0.10)

        ax.scatter(d['CL'].numpy(), d['z'].numpy() * Z_MAX,
                   c='red', s=50, alpha=0.8, label='LREE_obs', marker='o', zorder=5)
        ax.plot(Cp[:, 0], zv, 'b-', lw=2, label='LREE_PINN')
        ax.plot(CL_anal, zv, 'b--', lw=1, alpha=0.5, label='LREE_anal(β=0.15)')
        ax.plot(CH_anal, zv, 'g--', lw=1, alpha=0.5, label='HREE_anal(β=0.10)')
        ax.scatter(d['CH'].numpy(), d['z'].numpy() * Z_MAX,
                   c='orange', s=50, alpha=0.8, label='HREE_obs', marker='^', zorder=5)
        ax.plot(Cp[:, 1], zv, 'green', lw=2, label='HREE_PINN')

        ax.invert_yaxis()
        ax.set_xlabel('Concentration (ppm)'); ax.set_ylabel('Depth (m)')
        ax.set_title(f'{names.get(pid, str(pid))}\n'
                     f'LREE R²={r["r2_CL"]:.3f}  HREE R²={r["r2_CH"]:.3f}\n'
                     f'L/H R²={r["r2_R"]:.3f}')
        ax.legend(fontsize=6); ax.set_xlim(0)
        ax.grid(alpha=0.2)

    # Row 4: L/H ratio profiles
    for idx, pid in enumerate(sorted(profile_data.keys())):
        if idx >= 3: break
        row = 3; col = idx
        ax  = fig.add_subplot(gs[row, col])
        d = profile_data[pid]; r = results[pid]
        zv_m = d['z'].numpy().ravel() * Z_MAX

        RAT_pred_smooth = r['CL'] / (r['CH'] + 1e-6)
        ax.plot(RAT_pred_smooth, zv_m, 'b-', lw=2, label='PINN L/H')
        ax.plot(r['RAT_obs'], zv_m, 'ro', ms=5, alpha=0.7, label='Obs L/H')
        ax.invert_yaxis()
        ax.set_xlabel('LREE/HREE Ratio')
        ax.set_ylabel('Depth (m)')
        ax.set_title(f'L/H Ratio — {names.get(pid, str(pid))}\n'
                     f'R²={r["r2_R"]:.3f}')
        ax.legend(fontsize=7); ax.grid(alpha=0.2)

    # 训练曲线
    ax = fig.add_subplot(gs[3, 1])
    phys_arr = np.array(H['frac'])
    phys_arr = phys_arr[phys_arr > 0]
    ax.semilogy(phys_arr, lw=1.2, color='purple', label='Frac constraint')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Fractionation Loss')
    ax.set_title('Fractionation Constraint (beta_L >= beta_H)'); ax.grid(alpha=0.3)

    # 物理参数
    ax = fig.add_subplot(gs[3, 2])
    steps = np.arange(0, len(H['beta_L0']))[::50]
    bL = np.array(H['beta_L0'])[::50]
    bH = np.array(H['beta_H0'])[::50]
    ax.plot(steps, bL, lw=1.5, label='beta_L0', color='blue')
    ax.plot(steps, bH, lw=1.5, label='beta_H0', color='green')
    ax.set_xlabel('Iteration'); ax.set_ylabel('beta coefficient')
    ax.set_title('Learned Fractionation Coefficients\n'
                 f'beta_L/beta_H = {bL[-1]/bH[-1]:.3f}')
    ax.legend(); ax.grid(alpha=0.3)

    bL0 = float(torch.exp(params.log_beta_L0).item())
    bH0 = float(torch.exp(params.log_beta_H0).item())
    plt.suptitle(
        f'REE PINN v12 — LREE/HREE 分馏模型\n'
        f'beta_L={bL0:.4f}  beta_H={bH0:.4f}  '
        f'ratio={bL0/bH0:.3f} (LREE吸附更强 → 比值>1)',
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(f'{save_dir}/results_v12.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Figure: {save_dir}/results_v12.png')


if __name__ == '__main__':
    main()
