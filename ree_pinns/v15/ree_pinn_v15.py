"""
REE PINN v15 — 完整物理化学框架 + 对流-弥散-吸附分馏模型
================================================================

## 1. 物理化学框架

### 核心方程（稳态，∂C/∂t = 0）：
    D_eff · C''(z) - v · C'(z) - λ_sorp · R · (C - C_parent) = 0

    解析解：
    C(z) = C_parent + (C_atm - C_parent) · exp(-λ · z)

    其中 λ = (v/2D_eff) · [sqrt(1 + 4·R·D_eff·λ_sorp/v²) - 1]

    简化（当 R·D_eff·λ_sorp << v²）：
        λ ≈ (v/D_eff) · R/2

### LREE/HREE 分馏：
    λ_L = λ_base · sqrt(DK)   # LREE 阻滞更强
    λ_H = λ_base / sqrt(DK)   # HREE 阻滞更弱

    DK = K_d^L / K_d^H > 1（LREE 吸附更强）
    → λ_L > λ_H
    → LREE 更早趋于母岩值
    → L/H 比值在浅层最大，随深度减小

### 气候/径流作用（通过 λ_base）：
    λ_base(T, P, R) = λ_0 · exp(bT · T_n + bP · P_n + bR · R_n)

    - 温度↑ → 化学风化增强 → K_d↑ → λ↑
    - 降水↑ → 淋溶增强 → v↑ → λ↑
    - 径流↑ → 渗流通量↑ → λ↑

### 黏土矿物作用：
    f_clay(z) = α_K · Kaolinite(z) + α_I · Illite(z) + α_V · Vermiculite(z)
    K_d(z) = K_d0 · f_clay(z)

    （当有黏土数据时加入，K_d 随黏土含量增加）


## 2. PINN 边界条件

    BC1 (z=0): C(z=0) = C_atm（实测表层浓度）
    BC2 (z→∞): C(z→∞) → C_parent（实测最深样品）


## 3. 损失函数

    L = w_data · L_data + w_bc · L_bc + w_phys · L_phys + w_frac · L_frac

    L_data:    实测浓度拟合（双输出：LREE + HREE）
    L_bc:      边界条件（z=0 → C_atm, z=z_max → C_parent）
    L_phys:    稳态 A-D 方程残差（自动微分）
    L_frac:    分馏约束（λ_L > λ_H，软约束）


## 4. 展示目标

    A. 单剖面拟合：L/H(z) 曲线 + 实测点
    B. 物理参数：λ_L, λ_H, DK, bT, bP 学习曲线
    C. 跨剖面对比：所有剖面 L/H-z 叠加
    D. 敏感性分析：给定不同 T/P/R，预测 L/H 曲线（PINN 的外推能力）
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

DATA = '/Users/suheng/Desktop/claudecode/pinns/ree_pinns/data/REE_samples_20260424_with_climate.csv'
LIT   = '/Users/suheng/Desktop/claudecode/pinns/ree_pinns/data/REE_literature_20260424.csv'
SAVE  = '/Users/suheng/Desktop/claudecode/pinns/ree_pinns/v15'

Z_MAX = 33.0
T_MEAN, T_STD = 290.62, 3.78
P_MEAN, P_STD = 1604.84, 710.27
R_MEAN, R_STD = 0.5210, 0.2310

LREE_COLS = ['La_ppm', 'Ce_ppm', 'Pr_ppm', 'Nd_ppm', 'Sm_ppm']
HREE_COLS = ['Tb_ppm', 'Dy_ppm', 'Ho_ppm', 'Er_ppm', 'Tm_ppm', 'Yb_ppm', 'Lu_ppm']


# =====================================================================
# 网络（SIREN）
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


class REE_PINF(nn.Module):
    """
    物理-informed 网络

    输入：[z_norm, T_norm, P_norm, R_norm]
    输出：[C_LREE, C_HREE] (ppm)

    物理约束：
      - C >= 0（Softplus 输出层）
      - C 单调从 C_atm 变到 C_parent
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
        return torch.cat([self.head_L(feat), self.head_H(feat)], dim=1)


# =====================================================================
# 可学习物理参数
# =====================================================================
class PhysicalParams(nn.Module):
    """
    物理参数由气候驱动：

    λ_base = λ_0 · exp(bT · T_n + bP · P_n + bR · R_n)

    λ_L = λ_base · sqrt(DK)   # LREE
    λ_H = λ_base / sqrt(DK)  # HREE

    DK = K_d^L / K_d^H > 1（LREE 吸附更强）
    λ_0: 基准分馏系数（m⁻¹）
    bT, bP, bR: 气候敏感系数
    """
    def __init__(self):
        super().__init__()
        self.log_lambda0 = nn.Parameter(torch.tensor(-2.0))
        self.bT = nn.Parameter(torch.tensor(0.0))
        self.bP = nn.Parameter(torch.tensor(0.0))
        self.bR = nn.Parameter(torch.tensor(0.0))
        self.log_DK = nn.Parameter(torch.tensor(0.5))  # DK = e^0.5 ≈ 1.65

    def get_lambda(self, T_norm, P_norm, R_norm):
        lambda0 = torch.exp(self.log_lambda0).clamp(min=1e-6)
        bT = torch.tanh(self.bT) * 2.0
        bP = torch.tanh(self.bP) * 2.0
        bR = torch.tanh(self.bR) * 2.0

        lambda_base = lambda0 * torch.exp(
            bT * T_norm + bP * P_norm + bR * R_norm
        )

        DK = torch.exp(self.log_DK).clamp(min=1.0)
        lambda_H = lambda_base / torch.sqrt(DK)
        lambda_L = lambda_base * torch.sqrt(DK)

        return lambda_L, lambda_H


# =====================================================================
# 物理损失（稳态 A-D 方程）
# =====================================================================
def advect_diffuse_residual(pinn, params, z_norm, clim, C_parent_L, C_parent_H):
    """
    稳态对流-弥散方程残差（归一化）：
        C_i*'' - Pe_i · C_i*' - Ka_i · (C_i* - 1) = 0

    其中 C_i* = C_i / C_parent
          Pe_i = v · Z_max / D_eff
          Ka_i = R_i · λ_sorp · Z_max² / D_eff

    简化：不用 Pe 和 Ka，而是用 λ 直接编码
        稳态 A-D 的解是 exp(-λ·z)
        所以网络输出应满足：dC/dz ≈ -λ · (C - C_parent)

    物理损失：
        L_phys = mean[ (dC_L/dz + λ_L · (C_L - C_parent_L))² ]
                + mean[ (dC_H/dz + λ_H · (C_H - C_parent_H))² ]
    """
    z_phys = z_norm.clone().detach().requires_grad_(True)
    clim_phys = clim.clone().detach().requires_grad_(False)
    T_n = clim_phys[:, 0:1]
    P_n = clim_phys[:, 1:2]
    R_n = clim_phys[:, 2:3]

    C = pinn(torch.cat([z_phys, clim_phys], dim=1))
    CL, CH = C[:, 0:1], C[:, 1:2]

    # 自动微分
    dCL_dz = torch.autograd.grad(
        CL, z_phys, torch.ones_like(CL),
        create_graph=True, retain_graph=True
    )[0]
    dCH_dz = torch.autograd.grad(
        CH, z_phys, torch.ones_like(CH),
        create_graph=True, retain_graph=True
    )[0]

    # 气候驱动的 λ
    lambda_L, lambda_H = params.get_lambda(T_n, P_n, R_n)

    # 稳态 A-D 约束：dC/dz = -λ · (C - C_parent)
    # 残差
    Cp_L = C_parent_L + 1e-8
    Cp_H = C_parent_H + 1e-8

    resid_L = dCL_dz + lambda_L * (CL - Cp_L)
    resid_H = dCH_dz + lambda_H * (CH - Cp_H)

    # 相对量归一化（避免量纲问题）
    loss_L = torch.mean((resid_L / (CL + 1.0)) ** 2)
    loss_H = torch.mean((resid_H / (CH + 1.0)) ** 2)

    return loss_L + loss_H


# =====================================================================
# Trainer
# =====================================================================
class PINFTrainer:
    def __init__(self, pinn, params, profile_data):
        self.pinn = pinn
        self.params = params
        self.profile_data = profile_data
        self.device = next(pinn.parameters()).device
        self.opt = torch.optim.AdamW(
            list(pinn.parameters()) + list(params.parameters()),
            lr=1e-3, weight_decay=1e-4
        )
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=8000, eta_min=1e-6
        )
        self.hist = {
            'total': [], 'data': [], 'phys': [], 'frac': [], 'bound': [],
            'lambda0': [], 'bT': [], 'bP': [], 'bR': [], 'DK': []
        }

    def step(self, epoch, n_phys=300):
        # 分阶段：前2000步专注数据拟合，后逐步加入物理约束
        warmup = min(1.0, epoch / 2000.0)
        w_data  = 50.0
        w_phys  = 1.0 * warmup
        w_frac  = 0.5 * warmup
        w_bound = 10.0

        self.opt.zero_grad()

        # (1) 数据损失（按剖面平等）
        loss_data_list = []
        for pid, d in self.profile_data.items():
            x = torch.cat([d['z'], d['clim']], dim=1).to(self.device)
            Cp = self.pinn(x)
            loss_data_list.append(
                torch.mean((Cp[:, 0:1] - d['CL'].to(self.device)) ** 2)
                + torch.mean((Cp[:, 1:2] - d['CH'].to(self.device)) ** 2)
            )
        loss_data = torch.stack(loss_data_list).mean()

        # (2) 物理损失（A-D 方程）
        loss_phys_list = []
        for pid, d in self.profile_data.items():
            n_p = n_phys // len(self.profile_data)
            z_phys = (torch.rand(n_p, 1, device=self.device)
                       * d['z'].max().to(self.device))
            clim_phys = d['clim'].to(self.device)[:1].repeat(n_p, 1)
            Cp_L = d['CL_parent'].to(self.device).float()
            Cp_H = d['CH_parent'].to(self.device).float()
            lp = advect_diffuse_residual(
                self.pinn, self.params, z_phys, clim_phys, Cp_L, Cp_H
            )
            loss_phys_list.append(lp)
        loss_phys = torch.stack(loss_phys_list).mean()

        # (3) 分馏约束（λ_L > λ_H）
        loss_frac = torch.tensor(0.0, device=self.device)
        for pid, d in self.profile_data.items():
            T_n = d['clim'][:, 0:1].to(self.device)
            P_n = d['clim'][:, 1:2].to(self.device)
            R_n = d['clim'][:, 2:3].to(self.device)
            lL, lH = self.params.get_lambda(T_n, P_n, R_n)
            violation = torch.relu(lH - lL)
            loss_frac = loss_frac + torch.mean(violation ** 2)
        loss_frac = loss_frac / len(self.profile_data)

        # (4) 边界损失
        loss_bound = torch.tensor(0.0, device=self.device)
        for pid, d in self.profile_data.items():
            z0 = torch.zeros(50, 1, device=self.device)
            clim_bc = d['clim'].to(self.device)[:1].repeat(50, 1)
            C_bc = self.pinn(torch.cat([z0, clim_bc], dim=1))
            loss_bound = loss_bound + torch.mean(
                (C_bc[:, 0:1] - d['CL_atm'].to(self.device)) ** 2
            )
            loss_bound = loss_bound + torch.mean(
                (C_bc[:, 1:2] - d['CH_atm'].to(self.device)) ** 2
            )
        loss_bound = loss_bound / len(self.profile_data) / 2

        # 总损失
        loss = (w_data * loss_data + w_phys * loss_phys
                + w_frac * loss_frac + w_bound * loss_bound)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.pinn.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.params.parameters(), 1.0)
        self.opt.step()
        self.sched.step()

        self.hist['total'].append(loss.item())
        self.hist['data'].append(loss_data.item())
        self.hist['phys'].append(loss_phys.item())
        self.hist['frac'].append(loss_frac.item())
        self.hist['bound'].append(loss_bound.item())
        self.hist['lambda0'].append(float(torch.exp(self.params.log_lambda0).item()))
        self.hist['bT'].append(float(torch.tanh(self.params.bT).item() * 2))
        self.hist['bP'].append(float(torch.tanh(self.params.bP).item() * 2))
        self.hist['bR'].append(float(torch.tanh(self.params.bR).item() * 2))
        self.hist['DK'].append(float(torch.exp(self.params.log_DK).item()))

        return (loss.item(), loss_data.item(), loss_phys.item(),
                loss_frac.item(), loss_bound.item())


# =====================================================================
# 主程序
# =====================================================================
def main():
    print('=' * 60)
    print('REE PINF v15 — 物理化学框架 + 对流-弥散-吸附分馏模型')
    print('=' * 60)

    df  = pd.read_csv(DATA)
    dfl = pd.read_csv(LIT)
    df = df.merge(dfl[['id', 'parent_rock']],
                  left_on='literature_id', right_on='id',
                  how='left', suffixes=('', '_l'))
    df.parent_rock = df.parent_rock.fillna(df.Bedrock)
    df = df.dropna(subset=['Depth_m'] + LREE_COLS + HREE_COLS).reset_index(drop=True)
    df = df[df['Depth_m'] > 0].reset_index(drop=True)

    df['CL'] = df[LREE_COLS].sum(axis=1)
    df['CH'] = df[HREE_COLS].sum(axis=1)
    df['LH'] = df['CL'] / (df['CH'] + 1e-6)

    df['zn'] = df.Depth_m / Z_MAX
    df['Tn'] = (df.T_annual_mean_K - T_MEAN) / T_STD
    df['Pn'] = (df.P_annual_mean_mm_yr - P_MEAN) / P_STD
    df['Rn'] = (df.runoff_m_yr - R_MEAN) / R_STD

    profile_data = {}
    for lid, grp in df.groupby('literature_id'):
        z    = torch.tensor(grp.zn.values, dtype=torch.float32).reshape(-1, 1)
        c    = torch.tensor(grp[['Tn', 'Pn', 'Rn']].values, dtype=torch.float32)
        CL   = torch.tensor(grp.CL.values, dtype=torch.float32).reshape(-1, 1)
        CH   = torch.tensor(grp.CH.values, dtype=torch.float32).reshape(-1, 1)
        LH   = torch.tensor(grp.LH.values, dtype=torch.float32).reshape(-1, 1)

        surf = grp[grp.Depth_m < 1.0]
        CL_atm = float(surf.CL.mean()) if len(surf) > 0 else float(grp.CL.iloc[:2].mean())
        CH_atm = float(surf.CH.mean()) if len(surf) > 0 else float(grp.CH.iloc[:2].mean())
        CL_par = float(grp.CL.iloc[-1])
        CH_par = float(grp.CH.iloc[-1])

        profile_data[int(lid)] = {
            'z': z, 'clim': c, 'CL': CL, 'CH': CH, 'LH': LH,
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

    print(f'\n{len(df)} samples, {len(profile_data)} profiles')
    for pid, d in profile_data.items():
        LH_atm = d['CL_atm'].item() / (d['CH_atm'].item() + 1e-6)
        LH_par = d['CL_parent'] / (d['CH_parent'] + 1e-6)
        print(f"  {d['name']}: n={len(d['z'])}  "
              f"T={d['T_C']:.1f}C  P={d['P_mm']:.0f}mm  R={d['R_m']:.3f}m/yr  "
              f"LH_atm={LH_atm:.1f}  LH_parent={LH_par:.1f}")

    device = torch.device('cpu')
    pinn   = REE_PINF(64, 5).to(device)
    params = PhysicalParams().to(device)
    tr     = PINFTrainer(pinn, params, profile_data)

    print(f'\nNetwork: {sum(p.numel() for p in pinn.parameters()):,} params')
    print(f'Training 8000 iterations...')

    for i in range(8001):
        loss, ld, lp, lf, lb = tr.step(i, n_phys=300)
        if i % 1000 == 0 or i < 10:
            l0 = float(torch.exp(params.log_lambda0).item())
            bT = float(torch.tanh(params.bT).item() * 2)
            bP = float(torch.tanh(params.bP).item() * 2)
            bR = float(torch.tanh(params.bR).item() * 2)
            DK = float(torch.exp(params.log_DK).item())
            w  = min(1.0, i / 2000.0)
            print(f'Iter {i:>5}: L={loss:.2f}  Ld={ld:.2f}  '
                  f'Lp={lp:.4f}  Lf={lf:.4f}  Lb={lb:.2f}')
            print(f'       lambda0={l0:.4f}  DK={DK:.3f}  '
                  f'bT={bT:+.2f}  bP={bP:+.2f}  bR={bR:+.2f}  w={w:.2f}')

    # 评估
    pinn.eval()
    results = {}
    for pid, d in profile_data.items():
        x = torch.cat([d['z'], d['clim']], dim=1)
        with torch.no_grad():
            Cp = pinn(x).numpy()
        CL_obs = d['CL'].numpy().ravel()
        CH_obs = d['CH'].numpy().ravel()
        LH_obs = d['LH'].numpy().ravel()
        LH_pred = Cp[:, 0] / (Cp[:, 1] + 1e-6)

        r2_CL = pearsonr(CL_obs, Cp[:, 0])[0]**2
        r2_CH = pearsonr(CH_obs, Cp[:, 1])[0]**2
        r2_LH = pearsonr(LH_obs, LH_pred)[0]**2
        rmse_L = np.sqrt(np.mean((Cp[:, 0] - CL_obs)**2))
        rmse_H = np.sqrt(np.mean((Cp[:, 1] - CH_obs)**2))
        rmse_LH = np.sqrt(np.mean((LH_pred - LH_obs)**2))

        results[pid] = dict(
            CL=Cp[:, 0], CH=Cp[:, 1], LH_pred=LH_pred,
            r2_CL=r2_CL, r2_CH=r2_CH, r2_LH=r2_LH,
            rmse_L=rmse_L, rmse_H=rmse_H, rmse_LH=rmse_LH
        )

    print(f'\n=== Results ===')
    for pid, d in profile_data.items():
        r = results[pid]
        print(f'  [{pid}] {d["name"]}: '
              f'LREE_R2={r["r2_CL"]:.3f}  '
              f'HREE_R2={r["r2_CH"]:.3f}  '
              f'LH_R2={r["r2_LH"]:.3f}')

    plot(pinn, params, profile_data, results, tr.hist, SAVE)

    import os
    os.makedirs(SAVE, exist_ok=True)
    torch.save({'pinn': pinn.state_dict(), 'params': params.state_dict()},
               f'{SAVE}/model_v15.pt')
    print(f'\nSaved: {SAVE}/model_v15.pt')


# =====================================================================
# 绘图（展示目标）
# =====================================================================
def plot(pinn, params, profile_data, results, H, save_dir):
    import os
    n = len(profile_data)
    fig = plt.figure(figsize=(18, 20))
    gs  = GridSpec(5, 3, figure=fig, hspace=0.45, wspace=0.35)

    names = {
        1: 'Nagasawa\n花岗岩\nT=13C',
        2: 'Li2019\nA型花岗\nT=18C',
        3: 'Li&Zhou\nA型花岗\nT=18C',
        4: 'Fu2019\n流纹岩\nT=20C',
        5: 'Yaraghi\n二云母花岗\nT=24C',
        6: 'Luo\n闪长岩\nT=22C',
    }
    colors = plt.cm.tab10(np.linspace(0, 1, n))

    # ===================================================================
    # A. 单剖面 L/H(z) 曲线
    # ===================================================================
    for idx, pid in enumerate(sorted(profile_data.keys())):
        row = idx // 3; col = idx % 3
        ax  = fig.add_subplot(gs[row, col])
        d = profile_data[pid]; r = results[pid]
        z = d['z'].numpy().ravel() * Z_MAX

        # 预测曲线（平滑）
        Ta, Pa, Ra = d['Tn_mean'], d['Pn_mean'], d['Rn_mean']
        zv = np.linspace(0, float(d['z'].max()) * Z_MAX, 200)
        zn = torch.tensor(zv / Z_MAX, dtype=torch.float32).reshape(-1, 1)
        clim = torch.tensor([[Ta, Pa, Ra]] * len(zv), dtype=torch.float32)
        with torch.no_grad():
            Cp = pinn(torch.cat([zn, clim], dim=1)).numpy()
        LH_curve = Cp[:, 0] / (Cp[:, 1] + 1e-6)

        ax.scatter(d['LH'].numpy(), d['z'].numpy() * Z_MAX,
                   c='red', s=40, alpha=0.8, label='Obs L/H', zorder=5)
        ax.plot(LH_curve, zv, 'b-', lw=2, label='PINF L/H')
        ax.axhline(0, color='gray', ls=':', lw=1)
        ax.invert_yaxis()
        ax.set_xlabel('LREE/HREE Ratio')
        ax.set_ylabel('Depth (m)')
        ax.set_title(f'{names.get(pid, str(pid))}\n'
                     f'L/H R²={r["r2_LH"]:.3f}  '
                     f'LREE R²={r["r2_CL"]:.3f}  '
                     f'HREE R²={r["r2_CH"]:.3f}')
        ax.legend(fontsize=7); ax.grid(alpha=0.2)

    # ===================================================================
    # B. 所有剖面 L/H-z 叠加图
    # ===================================================================
    ax = fig.add_subplot(gs[3, 0])
    for idx, pid in enumerate(sorted(profile_data.keys())):
        d = profile_data[pid]; r = results[pid]
        Ta, Pa, Ra = d['Tn_mean'], d['Pn_mean'], d['Rn_mean']
        zv = np.linspace(0, float(d['z'].max()) * Z_MAX, 200)
        zn = torch.tensor(zv / Z_MAX, dtype=torch.float32).reshape(-1, 1)
        clim = torch.tensor([[Ta, Pa, Ra]] * len(zv), dtype=torch.float32)
        with torch.no_grad():
            Cp = pinn(torch.cat([zn, clim], dim=1)).numpy()
        LH_curve = Cp[:, 0] / (Cp[:, 1] + 1e-6)
        ax.plot(LH_curve, zv, '-', lw=2, color=colors[idx],
                label=f'[{pid}] {d["name"][1:15]}', alpha=0.8)
    ax.set_xlabel('LREE/HREE')
    ax.set_ylabel('Depth (m)')
    ax.set_title('All Profiles: L/H(z) Comparison\n'
                 '(PINF predictions at observed T/P/R)')
    ax.legend(fontsize=6, loc='upper right'); ax.invert_yaxis(); ax.grid(alpha=0.2)

    # ===================================================================
    # C. 气候敏感性分析（固定均值，变化 T）
    # ===================================================================
    ax = fig.add_subplot(gs[3, 1])
    T_mean_phys = np.mean([d['Tn_mean'] for d in profile_data.values()])
    P_mean_phys = np.mean([d['Pn_mean'] for d in profile_data.values()])
    R_mean_phys = np.mean([d['Rn_mean'] for d in profile_data.values()])
    zv = np.linspace(0, Z_MAX, 200)
    zn_base = torch.tensor(zv / Z_MAX, dtype=torch.float32).reshape(-1, 1)

    for delta_T in [-1.5, -0.5, 0.0, 0.5, 1.5]:
        T_n = T_mean_phys + delta_T
        clim = torch.tensor([[T_n, P_mean_phys, R_mean_phys]] * len(zv),
                           dtype=torch.float32)
        with torch.no_grad():
            Cp = pinn(torch.cat([zn_base, clim], dim=1)).numpy()
        LH_c = Cp[:, 0] / (Cp[:, 1] + 1e-6)
        ax.plot(LH_c, zv, lw=1.5, alpha=0.7,
                label=f'T + {delta_T*3.78:.1f}K')
    ax.set_xlabel('LREE/HREE')
    ax.set_ylabel('Depth (m)')
    ax.set_title('Sensitivity: T perturbation\n'
                 '(Holding P, R constant)')
    ax.legend(fontsize=7); ax.invert_yaxis(); ax.grid(alpha=0.2)

    # ===================================================================
    # D. 降水敏感性分析
    # ===================================================================
    ax = fig.add_subplot(gs[3, 2])
    for delta_P in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        P_n = P_mean_phys + delta_P
        clim = torch.tensor([[T_mean_phys, P_n, R_mean_phys]] * len(zv),
                           dtype=torch.float32)
        with torch.no_grad():
            Cp = pinn(torch.cat([zn_base, clim], dim=1)).numpy()
        LH_c = Cp[:, 0] / (Cp[:, 1] + 1e-6)
        ax.plot(LH_c, zv, lw=1.5, alpha=0.7,
                label=f'P + {delta_P*710:.0f}mm')
    ax.set_xlabel('LREE/HREE')
    ax.set_ylabel('Depth (m)')
    ax.set_title('Sensitivity: P perturbation\n'
                 '(Holding T, R constant)')
    ax.legend(fontsize=7); ax.invert_yaxis(); ax.grid(alpha=0.2)

    # ===================================================================
    # E. 物理参数演化
    # ===================================================================
    ax = fig.add_subplot(gs[4, 0])
    steps = np.arange(0, len(H['lambda0']))[::20]
    ax.semilogy(steps, np.array(H['lambda0'])[::20], lw=1.5, label='lambda0 (m⁻¹)')
    ax.set_xlabel('Iteration'); ax.set_ylabel('lambda0')
    ax.set_title('Learned lambda0 (base fractionation coeff)'); ax.legend()
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[4, 1])
    ax.plot(steps, np.array(H['bT'])[::20], lw=1.5, label='bT (T sensitivity)')
    ax.plot(steps, np.array(H['bP'])[::20], lw=1.5, label='bP (P sensitivity)')
    ax.plot(steps, np.array(H['bR'])[::20], lw=1.5, label='bR (R sensitivity)')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Coefficient')
    ax.set_title('Climate Sensitivity Coefficients'); ax.legend(); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[4, 2])
    ax.plot(steps, np.array(H['DK'])[::20], lw=1.5, label='DK = K_d^L / K_d^H',
            color='purple')
    ax.axhline(1.0, color='gray', ls='--', lw=1)
    ax.set_xlabel('Iteration'); ax.set_ylabel('DK')
    ax.set_title(f'L/H Fractionation Strength (DK)\n'
                 f'Final DK = {H["DK"][-1]:.3f}  '
                 f'→ λ_L/λ_H = sqrt(DK) = {np.sqrt(H["DK"][-1]):.3f}')
    ax.legend(); ax.grid(alpha=0.3)

    l0 = H['lambda0'][-1]
    DK = H['DK'][-1]
    bT = H['bT'][-1]
    bP = H['bP'][-1]
    bR = H['bR'][-1]

    plt.suptitle(
        f'REE PINF v15 — 物理化学框架\n'
        f'lambda0={l0:.4f} m⁻¹  DK={DK:.3f} (lambda_L/lambda_H={np.sqrt(DK):.3f})\n'
        f'bT={bT:+.3f}  bP={bP:+.3f}  bR={bR:+.3f}\n'
        f'物理：A-D方程 + 吸附阻滞 + 气候响应 | 展示：拟合 + 敏感性分析',
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(f'{save_dir}/results_v15.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Figure: {save_dir}/results_v15.png')


if __name__ == '__main__':
    main()
