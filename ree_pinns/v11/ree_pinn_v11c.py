"""
REE PINN v11c — Physics-Informed (气候驱动参数 + 分级权重)
=========================================================
改进：
1. 物理参数由气候条件驱动（不是全局常数）：
   D = D0 · exp[(T-T0)/ET + (P-P0)/EP]
   u = u0 · exp[(P-P0)/EP]   # 降水驱动渗流
2. 物理损失用残差相对量归一化，不爆炸
3. 分阶段训练：前2000步专注拟合数据，后3000步逐步加入物理
4. 剖面平等权重（和v10一致）
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


# =====================================================================
# 网络
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


class REE_PINN(nn.Module):
    def __init__(self, h=64, n=5):
        super().__init__()
        self.net = nn.Sequential(
            SineLayer(4, h, first=True),
            *[SineLayer(h, h) for _ in range(n-1)]
        )
        self.head = nn.Sequential(
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, 1), nn.Softplus()
        )

    def forward(self, x):
        if x.dim() == 1: x = x.reshape(1, -1)
        return self.head(self.net(x))


# =====================================================================
# 可学习物理基准参数（气候驱动）
# =====================================================================
class ClimatePhysicsParams(nn.Module):
    """
    物理参数由气候条件驱动，不是全局常数：

    D(T,P) = D0 · exp(bT · (T-T_MEAN)/T_STD + bP · (P-P_MEAN)/P_STD)
    u(T,P) = u0 · exp(bP · (P-P_MEAN)/P_STD)

    D0, u0: 基准弥散系数/渗流速度（实数域，用 softplus 变换）
    bT, bP: 气候敏感系数（可正可负，用 tanh 约束在 [-2, 2]）
    """
    def __init__(self):
        super().__init__()
        # 基准值（对数域）
        self.log_D0 = nn.Parameter(torch.tensor(-1.0))   # D0 = 0.368 m²/yr
        self.log_u0 = nn.Parameter(torch.tensor(-3.0))   # u0 = 0.05 m/yr
        # 气候敏感系数
        self.bT = nn.Parameter(torch.tensor(0.0))   # T 敏感
        self.bP = nn.Parameter(torch.tensor(0.0))   # P 敏感

    def get_params(self, T_norm, P_norm):
        """给定归一化气候输入，返回 D, u（逐样本）"""
        D0 = torch.exp(self.log_D0).clamp(min=1e-6)
        u0 = torch.exp(self.log_u0).clamp(min=1e-6)
        bT = torch.tanh(self.bT) * 2.0   # 约束在 [-2, 2]
        bP = torch.tanh(self.bP) * 2.0

        D = D0 * torch.exp(bT * T_norm + bP * P_norm)
        u = u0 * torch.exp(bP * P_norm)
        return D, u


# =====================================================================
# 物理残差（稳态对流-弥散，质量守恒形式）
# =====================================================================
def physics_residual(pinn, params, z_norm, clim, C_parent):
    """
    方程（归一化形式）：
        D* · C*'' - Pe* · C*' - Ka* · C* = -Ka*

    其中：
        D* = D(T,P) / D_ref  （归一化弥散系数）
        Pe* = Pe0(T,P)        （归一化Peclet数）
        Ka* = Ka0(T,P)        （归一化Damköhler数）

    残差用相对值归一化：(残差 / C_parent²) 的均值
    """
    z_phys = z_norm.clone().detach().requires_grad_(True)
    clim_phys = clim.clone().detach().requires_grad_(False)
    T_n, P_n, R_n = clim_phys[:, 0:1], clim_phys[:, 1:2], clim_phys[:, 2:3]

    # 网络输出
    C = pinn(torch.cat([z_phys, clim_phys], dim=1))

    # 反归一化
    C_star = C / (C_parent + 1e-8)

    # 一阶、二阶导数
    dC_dz = torch.autograd.grad(
        C_star, z_phys, torch.ones_like(C_star),
        create_graph=True, retain_graph=True
    )[0]
    d2C_dz2 = torch.autograd.grad(
        dC_dz, z_phys, torch.ones_like(dC_dz), create_graph=True
    )[0]

    # 气候驱动的物理参数
    D_star, Pe_star = params.get_params(T_n, P_n)
    Ka_star = Pe_star * 0.5   # 简化：Ka ∝ Pe

    # 归一化方程残差
    residual = d2C_dz2 - Pe_star * dC_dz - Ka_star * C_star + Ka_star

    # 相对量归一化（防止爆炸）
    rel_residual = residual ** 2 / (C_star ** 2 + 1.0)

    return rel_residual.mean()


# =====================================================================
# Trainer
# =====================================================================
class PINNTrainer:
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
        self.hist = {'total': [], 'data': [], 'bc': [], 'phys': [],
                     'D0': [], 'u0': [], 'bT': [], 'bP': []}

    def step(self, epoch, n_phys=200):
        # 分阶段权重：前2000步物理权重为0，逐步增大
        warmup = min(1.0, epoch / 2000.0)
        w_phys = 0.5 * warmup
        w_data = 100.0
        w_bc   = 10.0

        self.opt.zero_grad()

        # (1) 数据损失（剖面平等）
        loss_data_list = []
        for pid, d in self.profile_data.items():
            x = torch.cat([d['z'].to(self.device), d['clim'].to(self.device)], dim=1)
            Cp = self.pinn(x)
            loss_data_list.append(
                torch.mean((Cp - d['C'].to(self.device)) ** 2)
            )
        loss_data = torch.stack(loss_data_list).mean()

        # (2) 边界损失（表层）
        loss_bc_list = []
        for pid, d in self.profile_data.items():
            z_bc = torch.zeros(50, 1, device=self.device)
            clim_bc = d['clim'].to(self.device)[:1].repeat(50, 1)
            C_bc = self.pinn(torch.cat([z_bc, clim_bc], dim=1))
            loss_bc_list.append(
                torch.mean((C_bc - d['C_atm'].to(self.device)) ** 2)
            )
        loss_bc = torch.stack(loss_bc_list).mean()

        # (3) 物理损失（气候驱动）
        loss_phys_list = []
        for pid, d in self.profile_data.items():
            n_p = n_phys // len(self.profile_data)
            z_phys = torch.rand(n_p, 1, device=self.device) * d['z'].max()
            clim_phys = d['clim'].to(self.device)[:1].repeat(n_p, 1)
            C_parent = d['C_parent'].to(self.device).float()
            lp = physics_residual(self.pinn, self.params, z_phys, clim_phys, C_parent)
            loss_phys_list.append(lp)
        loss_phys = torch.stack(loss_phys_list).mean()

        # 总损失
        loss = w_data * loss_data + w_bc * loss_bc + w_phys * loss_phys
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.pinn.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.params.parameters(), 1.0)
        self.opt.step()
        self.sched.step()

        self.hist['total'].append(loss.item())
        self.hist['data'].append(loss_data.item())
        self.hist['bc'].append(loss_bc.item())
        self.hist['phys'].append(loss_phys.item())
        self.hist['D0'].append(float(torch.exp(self.params.log_D0).item()))
        self.hist['u0'].append(float(torch.exp(self.params.log_u0).item()))
        self.hist['bT'].append(float(torch.tanh(self.params.bT).item() * 2))
        self.hist['bP'].append(float(torch.tanh(self.params.bP).item() * 2))

        return loss.item(), loss_data.item(), loss_bc.item(), loss_phys.item()


# =====================================================================
# 主程序
# =====================================================================
def main():
    print('=' * 60)
    print('REE PINN v11c — Physics-Informed (气候驱动参数 + 分级权重)')
    print('=' * 60)

    df  = pd.read_csv(DATA)
    dfl = pd.read_csv(LIT)
    df = df.merge(dfl[['id', 'parent_rock']],
                  left_on='literature_id', right_on='id',
                  how='left', suffixes=('', '_l'))
    df.parent_rock = df.parent_rock.fillna(df.Bedrock)
    df = df.dropna(subset=['Depth_m']).reset_index(drop=True)

    df['zn'] = df.Depth_m / Z_MAX
    df['Tn'] = (df.T_annual_mean_K - T_MEAN) / T_STD
    df['Pn'] = (df.P_annual_mean_mm_yr - P_MEAN) / P_STD
    df['Rn'] = (df.runoff_m_yr - R_MEAN) / R_STD

    profile_data = {}
    for lid, grp in df.groupby('literature_id'):
        z    = torch.tensor(grp.zn.values, dtype=torch.float32).reshape(-1, 1)
        c    = torch.tensor(grp[['Tn', 'Pn', 'Rn']].values, dtype=torch.float32)
        C    = torch.tensor(grp.TotalREE_ppm.values, dtype=torch.float32).reshape(-1, 1)
        surf = grp[grp.Depth_m < 1.0]
        C_atm   = float(surf.TotalREE_ppm.mean()) if len(surf) > 0 \
                  else float(grp.TotalREE_ppm.iloc[:2].mean())
        C_parent = float(grp.TotalREE_ppm.iloc[-1])

        profile_data[int(lid)] = {
            'z': z, 'clim': c, 'C': C,
            'C_atm': torch.tensor([[C_atm]]),
            'C_parent': torch.tensor([[C_parent]]),
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
        print(f"  {d['name']}: n={len(d['z'])}  "
              f"T={d['T_C']:.1f}C  P={d['P_mm']:.0f}mm  "
              f"C_atm={d['C_atm'].item():.0f}ppm  "
              f"C_parent={d['C_parent'].item():.0f}ppm")

    device = torch.device('cpu')
    pinn   = REE_PINN(64, 5).to(device)
    params = ClimatePhysicsParams().to(device)
    tr     = PINNTrainer(pinn, params, profile_data)

    print(f'\nNetwork params: {sum(p.numel() for p in pinn.parameters()):,}')
    print(f'Physics params: D0, u0, bT, bP (climate-driven)')
    print(f'Device: {device}')
    print(f'\nTraining 5000 iterations (warmup 2000)...')
    print(f'{"Iter":>6}  {"Total":>12}  {"Data":>12}  {"BC":>12}  {"Physics":>12}')

    for i in range(5001):
        loss, ld, lb, lp = tr.step(i, n_phys=200)
        if i % 500 == 0 or i < 10:
            D0 = float(torch.exp(params.log_D0).item())
            u0 = float(torch.exp(params.log_u0).item())
            bT = float(torch.tanh(params.bT).item() * 2)
            bP = float(torch.tanh(params.bP).item() * 2)
            w  = min(1.0, i / 2000.0)
            print(f'{i:>6}  {loss:>12.4f}  {ld:>12.4f}  '
                  f'{lb:>12.4f}  {lp:>12.4f}')
            print(f'       D0={D0:.2e}  u0={u0:.2e}  bT={bT:+.2f}  bP={bP:+.2f}  '
                  f'w_phys={w:.2f}')

    # 评估
    pinn.eval()
    results = {}
    for pid, d in profile_data.items():
        x = torch.cat([d['z'], d['clim']], dim=1)
        with torch.no_grad():
            Cp = pinn(x).numpy().ravel()
        Co = d['C'].numpy().ravel()
        r2   = np.corrcoef(Co, Cp)[0, 1] ** 2
        rmse = np.sqrt(np.mean((Cp - Co) ** 2))
        results[pid] = dict(Cp=Cp, Co=Co, r2=r2, rmse=rmse)

    Cp_all = np.concatenate([results[pid]['Cp'] for pid in results])
    Co_all = np.concatenate([results[pid]['Co'] for pid in results])
    r2_g   = np.corrcoef(Co_all, Cp_all)[0, 1] ** 2
    rmse_g = np.sqrt(np.mean((Cp_all - Co_all) ** 2))

    print(f'\n=== Results ===')
    print(f'Global R²={r2_g:.4f}  RMSE={rmse_g:.1f}ppm')
    for pid, d in profile_data.items():
        r = results[pid]
        flag = 'OK' if r['r2'] > 0.9 else 'WARN' if r['r2'] > 0.5 else 'FAIL'
        print(f'  [{flag}] {d["name"]}  R²={r["r2"]:.3f}  RMSE={r["rmse"]:.0f}ppm')

    plot(pinn, params, profile_data, results, tr.hist, r2_g, rmse_g, SAVE)

    import os
    os.makedirs(SAVE, exist_ok=True)
    torch.save({
        'pinn': pinn.state_dict(),
        'params': params.state_dict()
    }, f'{SAVE}/model_v11c.pt')
    print(f'\nSaved: {SAVE}/model_v11c.pt')


# =====================================================================
# 绘图
# =====================================================================
def plot(pinn, params, profile_data, results, H, r2_g, rmse_g, save_dir):
    import os
    n = len(profile_data)
    fig = plt.figure(figsize=(18, 14))
    gs  = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
    colors = plt.cm.tab10(np.linspace(0, 1, n))

    # 1:1
    ax = fig.add_subplot(gs[0, 0])
    for (pid, r), c in zip(results.items(), colors):
        ax.scatter(r['Co'], r['Cp'], c=[c], s=40, alpha=0.7,
                   label=profile_data[pid]['name'])
    m = max(np.concatenate([r['Co'] for r in results.values()]).max(),
             np.concatenate([r['Cp'] for r in results.values()]).max())
    ax.plot([0, m], [0, m], 'k--', lw=1.5)
    ax.set_xlabel('Observed (ppm)'); ax.set_ylabel('Predicted (ppm)')
    ax.set_title(f'1:1  R²={r2_g:.3f}  RMSE={rmse_g:.0f}ppm')
    ax.set_xlim(0); ax.set_ylim(0); ax.legend(fontsize=6)

    # 物理参数演化
    ax = fig.add_subplot(gs[0, 1])
    steps = np.arange(0, len(H['D0']))[::50]
    ax.semilogy(steps, np.array(H['D0'])[::50], lw=1.5, label='D0 (m²/yr)')
    ax.semilogy(steps, np.array(H['u0'])[::50], lw=1.5, label='u0 (m/yr)')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Physical Parameters')
    ax.set_title('Physical Parameter Evolution'); ax.legend(); ax.grid(alpha=0.3)

    # 物理损失曲线
    ax = fig.add_subplot(gs[0, 2])
    phys_arr = np.array(H['phys'])
    ax.semilogy(phys_arr[phys_arr > 1e-6], lw=1.2, color='purple')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Physics Loss (norm)')
    ax.set_title('Physics Residual (归一化)'); ax.grid(alpha=0.3)

    names = {
        1: 'Nagasawa\n花岗岩\nT=13C',
        2: 'Li2019\nA型花岗岩\nT=18C',
        3: 'Li&Zhou\nA型花岗岩\nT=18C',
        4: 'Fu2019\n流纹岩\nT=20C',
        5: 'Yaraghi\n二云母花岗岩\nT=24C',
        6: 'Luo\n闪长岩\nT=22C',
    }
    for idx, pid in enumerate(sorted(profile_data.keys())):
        row = idx // 3; col = idx % 3
        ax  = fig.add_subplot(gs[row + 1, col])
        d = profile_data[pid]; r = results[pid]

        Ta = d['Tn_mean']; Pa = d['Pn_mean']; Ra = d['Rn_mean']
        zv = np.linspace(0, float(d['z'].max()) * Z_MAX, 200)
        zn = torch.tensor(zv / Z_MAX, dtype=torch.float32).reshape(-1, 1)
        clim = torch.tensor([[Ta, Pa, Ra]] * len(zv), dtype=torch.float32)
        with torch.no_grad():
            Cv = pinn(torch.cat([zn, clim], dim=1)).numpy().ravel()

        ax.scatter(d['C'].numpy(), d['z'].numpy() * Z_MAX,
                   c='red', s=50, alpha=0.8, label='Obs', zorder=5)
        ax.plot(Cv, zv, 'b-', lw=2, label='PINN v11c')

        ax.axhline(0, color='gray', ls=':', lw=1)
        ax.invert_yaxis()
        ax.set_xlabel('TotalREE (ppm)'); ax.set_ylabel('Depth (m)')
        ax.set_title(f'{names.get(pid, str(pid))}\n'
                     f'R²={r["r2"]:.3f}  RMSE={r["rmse"]:.0f}ppm')
        ax.legend(fontsize=7); ax.set_xlim(0)

    D0 = float(torch.exp(params.log_D0).item())
    u0 = float(torch.exp(params.log_u0).item())
    bT = float(torch.tanh(params.bT).item() * 2)
    bP = float(torch.tanh(params.bP).item() * 2)

    plt.suptitle(
        f'REE PINN v11c — Physics-Informed (气候驱动参数)\n'
        f'D0={D0:.2e} m²/yr  u0={u0:.2e} m/yr  '
        f'bT={bT:+.2f}  bP={bP:+.2f}\n'
        f'R²={r2_g:.3f}  RMSE={rmse_g:.0f} ppm',
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(f'{save_dir}/results_v11c.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Figure: {save_dir}/results_v11c.png')


if __name__ == '__main__':
    main()
