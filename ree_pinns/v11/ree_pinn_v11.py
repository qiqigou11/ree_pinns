"""
REE PINN v11 — Physics-Informed Neural Network (稳态对流-弥散框架)
================================================================
核心：物理损失直接作用在浓度上，而非任意量纲的归一化坐标。

物理方程（稳态，∂C/∂t = 0）：
    D·C''(z) - u·C'(z) - k·C(z) = -k·C_parent

其中：
- D   : 有效弥散系数 (m²/yr)，刻画深度方向的浓度梯度
- u   : 渗流速度 (m/yr)，刻画表层-深处浓度差的驱动力
- k   : 吸附-解吸速率常数 (1/yr)，刻画REE在液相和固相间的交换
- C_parent: 母岩浓度 (ppm)，由数据直接给出

损失函数：
    L = λ_data · L_data + λ_bc · L_bc + λ_phys · L_physics

网络输入：[z_norm, T_norm, P_norm, R_norm] → 输出：C(z)
物理残差：对网络输出做自动微分得到 C'(z) 和 C''(z)，代入方程。
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

# =====================================================================
# 路径配置
# =====================================================================
DATA = '/Users/suheng/Desktop/claudecode/pinns/ree_pinns/data/REE_samples_20260424_with_climate.csv'
LIT   = '/Users/suheng/Desktop/claudecode/pinns/ree_pinns/data/REE_literature_20260424.csv'
SAVE  = '/Users/suheng/Desktop/claudecode/pinns/ree_pinns/v11'

# =====================================================================
# 物理常数
# =====================================================================
Z_MAX  = 33.0     # 最大深度 (m)
T_MEAN, T_STD = 290.62, 3.78
P_MEAN, P_STD = 1604.84, 710.27
R_MEAN, R_STD = 0.5210, 0.2310


# =====================================================================
# 网络（SIREN，和v9相同）
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
    """输入：[z_norm, T_norm, P_norm, R_norm]，输出：C(z) in ppm"""
    def __init__(self, h=64, n=5):
        super().__init__()
        self.net = nn.Sequential(
            SineLayer(4, h, first=True),
            *[SineLayer(h, h) for _ in range(n-1)]
        )
        self.head = nn.Sequential(
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, 1),
            nn.Softplus()   # 非负约束
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.reshape(1, -1)
        return self.head(self.net(x))


# =====================================================================
# 可学习物理参数（标量）
# =====================================================================
class PhysicsParams(nn.Module):
    """
    可学习的物理参数，物理约束：
        D > 0, u >= 0, k > 0
        log(D) 和 log(k) 在实数域无约束，用 softplus 变换
    """
    def __init__(self):
        super().__init__()
        # log(D) in m²/yr，初始值 1e-4
        self.log_D  = nn.Parameter(torch.tensor(0.0))
        # u in m/yr，初始值 1e-3
        self.log_u  = nn.Parameter(torch.tensor(-1.0))
        # log(k) in 1/yr，初始值 1e-2
        self.log_k  = nn.Parameter(torch.tensor(0.0))

    @property
    def D(self):
        return torch.exp(self.log_D)          # D > 0

    @property
    def u(self):
        return torch.exp(self.log_u)          # u >= 0 (clamp to >=0)

    @property
    def k(self):
        return torch.exp(self.log_k)          # k > 0


# =====================================================================
# 物理残差计算
# =====================================================================
def physics_residual(pinn, params, z, clim, C_parent):
    """
    计算稳态对流-弥散方程残差
    方程: D·C''(z) - u·C'(z) - k·C(z) = -k·C_parent

    z:       (N, 1) 归一化深度
    clim:    (N, 3) [T_norm, P_norm, R_norm]
    C_parent: float  母岩浓度 (ppm)

    返回: (N,) 的残差向量
    """
    # 克隆并要求梯度（z 需要追踪梯度，clim 不需要）
    z_phys = z.clone().detach().requires_grad_(True)
    clim_phys = clim.clone().detach().requires_grad_(False)

    # 网络输出 C(z, clim)
    C = pinn(torch.cat([z_phys, clim_phys], dim=1))

    # 自动微分：dC/dz 和 d²C/dz²
    dC_dz = torch.autograd.grad(
        C, z_phys,
        torch.ones_like(C),
        create_graph=True,
        retain_graph=True
    )[0]

    d2C_dz2 = torch.autograd.grad(
        dC_dz, z_phys,
        torch.ones_like(dC_dz),
        create_graph=True
    )[0]

    D  = params.D
    u  = params.u
    k  = params.k
    Cp = C_parent

    # 物理残差（浓度量纲，ppm）
    # 方程：D·C'' - u·C' - k·C = -k·C_parent
    residual = D * d2C_dz2 - u * dC_dz - k * C + k * Cp

    return residual.squeeze()


# =====================================================================
# Trainer
# =====================================================================
class PINNTrainer:
    """
    损失 = λ_data·L_data + λ_bc·L_bc + λ_phys·L_physics
    - L_data: 剖面平均浓度数据
    - L_bc:   顶部边界（表层实测浓度）
    - L_phys: 稳态对流-弥散方程残差
    """
    def __init__(self, pinn, params, profile_data):
        self.pinn   = pinn
        self.params = params
        self.profile_data = profile_data  # {pid: {z, clim, C, C_parent, C_atm}}
        self.device = next(pinn.parameters()).device

        self.opt = torch.optim.AdamW(
            list(pinn.parameters()) + list(params.parameters()),
            lr=1e-3, weight_decay=1e-4
        )
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, patience=500, factor=0.5, min_lr=1e-6
        )

        # 存储训练历史
        self.hist = {'total': [], 'data': [], 'bc': [], 'phys': []}

    def step(self, n_phys=200):
        self.opt.zero_grad()

        # ---- (1) 数据损失（按剖面平均）----
        loss_data_list = []
        for pid, d in self.profile_data.items():
            x = torch.cat([d['z'].to(self.device), d['clim'].to(self.device)], dim=1)
            Cp = self.pinn(x)
            loss_data_list.append(
                torch.mean((Cp - d['C'].to(self.device)) ** 2)
            )
        loss_data = torch.stack(loss_data_list).mean()

        # ---- (2) 边界损失（表层浓度）----
        loss_bc_list = []
        for pid, d in self.profile_data.items():
            z_bc = torch.zeros(50, 1, device=self.device)
            clim_bc = d['clim'].to(self.device)[:1].repeat(50, 1)
            x_bc = torch.cat([z_bc, clim_bc], dim=1)
            C_bc = self.pinn(x_bc)
            loss_bc_list.append(
                torch.mean((C_bc - d['C_atm'].to(self.device)) ** 2)
            )
        loss_bc = torch.stack(loss_bc_list).mean()

        # ---- (3) 物理损失（稳态对流-弥散）----
        loss_phys_list = []
        for pid, d in self.profile_data.items():
            # 在 [0, z_max] 区间采样
            n_p = n_phys // len(self.profile_data)
            z_phys = torch.rand(n_p, 1, device=self.device) * d['z'].max()
            clim_phys = d['clim'].to(self.device)[:1].repeat(n_p, 1)
            C_parent = d['C_parent'].to(self.device)

            resid = physics_residual(
                self.pinn, self.params,
                z_phys, clim_phys, C_parent
            )
            loss_phys_list.append(torch.mean(resid ** 2))
        loss_phys = torch.stack(loss_phys_list).mean()

        # ---- 总损失 ----
        loss = 100.0 * loss_data + 10.0 * loss_bc + 1.0 * loss_phys
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.pinn.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.params.parameters(), 1.0)
        self.opt.step()
        self.sched.step(loss_data + loss_bc + 0.1 * loss_phys)

        self.hist['total'].append(loss.item())
        self.hist['data'].append(loss_data.item())
        self.hist['bc'].append(loss_bc.item())
        self.hist['phys'].append(loss_phys.item())

        return loss.item(), loss_data.item(), loss_bc.item(), loss_phys.item()


# =====================================================================
# 主程序
# =====================================================================
def main():
    print('=' * 60)
    print('REE PINN v11 — Physics-Informed (稳态对流-弥散)')
    print('=' * 60)

    # ---- 加载数据 ----
    df  = pd.read_csv(DATA)
    dfl = pd.read_csv(LIT)
    df = df.merge(dfl[['id', 'parent_rock']],
                  left_on='literature_id', right_on='id',
                  how='left', suffixes=('', '_l'))
    df.parent_rock = df.parent_rock.fillna(df.Bedrock)
    df = df.dropna(subset=['Depth_m']).reset_index(drop=True)

    # 归一化
    df['zn'] = df.Depth_m / Z_MAX
    df['Tn'] = (df.T_annual_mean_K - T_MEAN) / T_STD
    df['Pn'] = (df.P_annual_mean_mm_yr - P_MEAN) / P_STD
    df['Rn'] = (df.runoff_m_yr - R_MEAN) / R_STD

    # ---- 按剖面组织 ----
    profile_data = {}
    for lid, grp in df.groupby('literature_id'):
        z   = torch.tensor(grp.zn.values, dtype=torch.float32).reshape(-1, 1)
        c   = torch.tensor(grp[['Tn', 'Pn', 'Rn']].values, dtype=torch.float32)
        C   = torch.tensor(grp.TotalREE_ppm.values, dtype=torch.float32).reshape(-1, 1)
        T_C = grp.T_annual_mean_K.iloc[0] - 273.15
        P_mm = grp.P_annual_mean_mm_yr.iloc[0]
        R_m  = grp.runoff_m_yr.iloc[0]

        # 表层浓度（深度 < 1m 的均值）
        surf = grp[grp.Depth_m < 1.0]
        C_atm = float(surf.TotalREE_ppm.mean()) if len(surf) > 0 \
                else float(grp.TotalREE_ppm.iloc[:2].mean())

        # 母岩浓度（深度最大样本）
        C_parent = float(grp.TotalREE_ppm.iloc[-1])

        profile_data[int(lid)] = {
            'z': z, 'clim': c, 'C': C,
            'C_atm': torch.tensor([[C_atm]]),
            'C_parent': torch.tensor([[C_parent]]),
            'name': f"[{int(lid)}] {grp.parent_rock.iloc[0][:15]}",
            'T_C': T_C, 'P_mm': P_mm, 'R_m': R_m,
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

    # ---- 建网 + 参数 ----
    device = torch.device('cpu')
    pinn   = REE_PINN(64, 5).to(device)
    params = PhysicsParams().to(device)
    tr     = PINNTrainer(pinn, params, profile_data)

    print(f'\nNetwork params: {sum(p.numel() for p in pinn.parameters()):,}')
    print(f'Physics params: D, u, k (log-scale)')
    print(f'Device: {device}')

    # ---- 训练 ----
    N_ITER = 5000
    print(f'\nTraining {N_ITER} iterations...')
    print(f'{"Iter":>6}  {"Total":>12}  {"Data":>12}  {"BC":>12}  {"Physics":>12}')

    for i in range(N_ITER + 1):
        loss, ld, lb, lp = tr.step(n_phys=200)
        if i % 500 == 0 or i < 10:
            D = float(params.D.item())
            u = float(params.u.item())
            k = float(params.k.item())
            print(f'{i:>6}  {loss:>12.4f}  {ld:>12.4f}  '
                  f'{lb:>12.4f}  {lp:>12.4f}')
            print(f'       D={D:.2e} m²/yr  u={u:.2e} m/yr  k={k:.2e} 1/yr')
        if i == 1000:
            print(f'{i:>6}  {loss:>12.4f}  {ld:>12.4f}  '
                  f'{lb:>12.4f}  {lp:>12.4f}')

    # ---- 评估 ----
    pinn.eval()
    results = {}
    for pid, d in profile_data.items():
        x = torch.cat([d['z'], d['clim']], dim=1)
        with torch.no_grad():
            Cp = pinn(x).numpy().ravel()
        Co = d['C'].numpy().ravel()
        r2    = np.corrcoef(Co, Cp)[0, 1] ** 2
        rmse  = np.sqrt(np.mean((Cp - Co) ** 2))
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

    # ---- 绘图 ----
    plot(pinn, params, profile_data, results, tr.hist, r2_g, rmse_g, SAVE)

    # ---- 保存 ----
    import os
    os.makedirs(SAVE, exist_ok=True)
    torch.save({'pinn': pinn.state_dict(), 'params': params.state_dict()},
               f'{SAVE}/model_v11.pt')
    print(f'\nSaved: {SAVE}/model_v11.pt')


# =====================================================================
# 绘图
# =====================================================================
def plot(pinn, params, profile_data, results, H, r2_g, rmse_g, save_dir):
    n = len(profile_data)
    fig = plt.figure(figsize=(18, 14))
    gs  = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    colors = plt.cm.tab10(np.linspace(0, 1, n))

    # --- (1) 全局 1:1 ---
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

    # --- (2) 物理参数曲线 ---
    ax = fig.add_subplot(gs[0, 1])
    D_arr  = [h['phys'] for h in H[::50]]  # placeholder
    ax.set_title('Training History')

    # --- (3) 物理损失曲线 ---
    ax = fig.add_subplot(gs[0, 2])
    phys_arr = np.array(H['phys'])
    ax.semilogy(phys_arr[phys_arr > 0], lw=1.2, label='Physics')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Physics Loss')
    ax.set_title('Physics Residual (稳态A-D方程)')
    ax.grid(alpha=0.3)

    # --- (4-9) 各剖面 ---
    names = {
        1: 'Nagasawa\n花岗岩风化壳\nT=13C',
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

        # 预测曲线（平均气候）
        Ta = d['Tn_mean']; Pa = d['Pn_mean']; Ra = d['Rn_mean']
        zv = np.linspace(0, float(d['z'].max()) * Z_MAX, 200)
        zn = torch.tensor(zv / Z_MAX, dtype=torch.float32).reshape(-1, 1)
        clim = torch.tensor([[Ta, Pa, Ra]] * len(zv), dtype=torch.float32)
        with torch.no_grad():
            Cv = pinn(torch.cat([zn, clim], dim=1)).numpy().ravel()

        # 物理预测（解析解）
        D  = float(params.D.item())
        u  = float(params.u.item())
        k  = float(params.k.item())
        Cp = float(d['C_parent'].item())
        Ca = float(d['C_atm'].item())
        if k > 1e-8 and D > 1e-10:
            alpha = np.sqrt(u**2 / (4*D**2) + k/D)
            z_phys = zv / Z_MAX  # 归一化（简化）
            C_anal = Cp + (Ca - Cp) * np.exp(-np.sqrt(k/D) * zv)

        ax.scatter(d['C'].numpy(), d['z'].numpy() * Z_MAX,
                   c='red', s=50, alpha=0.8, label='Obs', zorder=5)
        ax.plot(Cv, zv, 'b-', lw=2, label='PINN v11')
        if k > 1e-8:
            ax.plot(C_anal, zv, 'g--', lw=1.5, label=f'Analytic (k={k:.2e})')

        ax.axhline(0, color='gray', ls=':', lw=1)
        ax.invert_yaxis()
        ax.set_xlabel('TotalREE (ppm)'); ax.set_ylabel('Depth (m)')
        ax.set_title(f'{names.get(pid, str(pid))}\n'
                     f'R²={r["r2"]:.3f}  RMSE={r["rmse"]:.0f}ppm')
        ax.legend(fontsize=7); ax.set_xlim(0)

    # 物理参数注释
    D = float(params.D.item())
    u = float(params.u.item())
    k = float(params.k.item())

    plt.suptitle(
        f'REE PINN v11 — Physics-Informed (稳态对流-弥散)\n'
        f'D={D:.2e} m²/yr  u={u:.2e} m/yr  k={k:.2e} 1/yr\n'
        f'R²={r2_g:.3f}  RMSE={rmse_g:.0f} ppm',
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(f'{SAVE}/results_v11.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Figure: {SAVE}/results_v11.png')


if __name__ == '__main__':
    main()
