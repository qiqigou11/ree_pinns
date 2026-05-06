"""
REE PINN v15 — 混合参数预测网络
==============================================================
物理框架：对流-弥散-吸附方程的解析解

    C(z) = C_parent + (C_atm - C_parent) · exp(-λ·z)

    λ = (v/2D) · [1 + sqrt(1 + 4kD/v²)]

核心创新：
1. 预测λ参数而非直接预测浓度（小数据集更稳定）
2. LREE和HREE分离建模（λ_H > λ_L，HREE吸附更强）
3. 粘土矿物从Profile 3比例模板映射

输入: [z_norm, T_norm, P_norm, R_norm, clay_norm]
输出: [λ_L, λ_H]
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

# ============================================================================
# 路径配置
# ============================================================================
DATA = '/Users/suheng/Desktop/claudecode/pinns/ree_pinns/data/REE_samples_20260424_with_climate.csv'
LIT   = '/Users/suheng/Desktop/claudecode/pinns/ree_pinns/data/REE_literature_2024.csv'
SAVE  = '/Users/suheng/Desktop/claudecode/pinns/ree_pinns/v11'

# ============================================================================
# REE列定义
# ============================================================================
LREE_COLS = ['La_ppm', 'Ce_ppm', 'Pr_ppm', 'Nd_ppm', 'Sm_ppm']
HREE_COLS = ['Tb_ppm', 'Dy_ppm', 'Ho_ppm', 'Er_ppm', 'Tm_ppm', 'Yb_ppm', 'Lu_ppm']

# ============================================================================
# 归一化参数
# ============================================================================
Z_MAX  = 33.0
T_MEAN, T_STD = 290.62, 3.78
P_MEAN, P_STD = 1604.84, 710.27
R_MEAN, R_STD = 0.5210, 0.2310
CLAY_MAX = 1.0  # 归一化粘土总量

# ============================================================================
# 物理公式
# ============================================================================
def analytical_C(z, C_atm, C_parent, lam):
    """
    稳态对流-弥散-吸附解析解
    C(z) = C_parent + (C_atm - C_parent) · exp(-λ·z)
    """
    return C_parent + (C_atm - C_parent) * np.exp(-lam * z)


def analytical_LH_ratio(z, CL_atm, CH_atm, CL_par, CH_par, lam_L, lam_H):
    """L/H比值随深度的变化"""
    CL = analytical_C(z, CL_atm, CL_par, lam_L)
    CH = analytical_C(z, CH_atm, CH_par, lam_H)
    return CL / (CH + 1e-6)


# ============================================================================
# 粘土矿物映射
# ============================================================================
# Profile 3 (Li&Zhou) 的粘土矿物比例（归一化到sum=1）
P3_CLAY_RATIOS = {
    'Kaolinite': 0.68 / 0.854,    # ~0.80
    'Vermiculite': 0.11 / 0.854,  # ~0.13
    'Illite': 0.06 / 0.854,       # ~0.07
}


def estimate_clay_for_profile(T_C, P_mm, z_max):
    """
    根据气候估算总粘土量（当没有实测数据时）

    气候越暖湿 →  化学风化越强 → 粘土矿物越多
    """
    # 简化的气候-粘土关系
    # T > 20°C, P > 1500mm/yr → 高粘土
    # T < 15°C, P < 1000mm/yr → 低粘土
    climate_score = (max(0, T_C - 10) / 20) * (max(0, P_mm - 500) / 1500)
    total_clay = min(0.9, 0.3 + 0.6 * climate_score)  # 0.3-0.9范围
    return total_clay


def map_clay_to_profile(profile_z, profile_id, T_C, P_mm, measured_clay=None):
    """
    将Profile 3的粘土矿物比例映射到目标剖面

    策略：
    1. 有实测数据的剖面（id=3,6）：用实测值
    2. 无实测数据的剖面：根据气候估算总粘土量
    3. 粘土矿物组成：按Profile 3的比例分配
    """
    # 对于每个深度点，计算粘土矿物含量
    clay_by_depth = []

    # Profile 3的粘土矿物随深度的变化（近似：高岭石随深度减少，蛭石/伊利石随深度增加）
    # 这是简化的示意，实际应根据剖面具体情况调整
    for z in profile_z:
        # 归一化深度
        z_norm = min(z / max(profile_z) if max(profile_z) > 0 else 0, 1.0)

        if measured_clay is not None and len(measured_clay) > 0:
            # 有实测数据：使用实测总粘土量
            total_clay = np.mean(measured_clay) / 100.0  # 假设实测是百分比
        else:
            # 无实测数据：根据气候估算
            total_clay = estimate_clay_for_profile(T_C, P_mm, max(profile_z))

        # 按Profile 3的比例分配
        clay_i = {
            'Kaolinite': total_clay * P3_CLAY_RATIOS['Kaolinite'],
            'Vermiculite': total_clay * P3_CLAY_RATIOS['Vermiculite'],
            'Illite': total_clay * P3_CLAY_RATIOS['Illite'],
        }
        clay_i['total'] = sum(clay_i.values())
        clay_by_depth.append(clay_i)

    return clay_by_depth


# ============================================================================
# LambdaPINN 网络
# ============================================================================
class LambdaPINN(nn.Module):
    """
    预测λ_L和λ_H的神经网络

    输入: [z_norm, T_norm, P_norm, R_norm, clay_norm] (5个特征)
    输出: [λ_L, λ_H] (2个正参数)
    """

    def __init__(self, hidden=32, n_layers=3):
        super().__init__()

        layers = []
        layers.append(nn.Linear(5, hidden))
        layers.append(nn.ReLU())

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden, 2))
        layers.append(nn.Softplus())  # 确保λ > 0

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # [λ_L, λ_H], both positive


# ============================================================================
# Trainer
# ============================================================================
class Trainer:
    def __init__(self, pinn, profile_data, device='cpu'):
        self.pinn = pinn.to(device)
        self.profile_data = profile_data
        self.device = device
        self.pids = list(profile_data.keys())

        self.opt = torch.optim.AdamW(pinn.parameters(), lr=1e-3, weight_decay=1e-4)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, patience=500, factor=0.5, min_lr=1e-6)

    def compute_loss(self, lambda_pred_dict, profile_data):
        """
        计算总损失

        L_total = 100·L_data + 1·L_phys + 0.1·L_reg
        """
        loss_data_list = []
        loss_phys_list = []

        for pid in self.pids:
            d = profile_data[pid]
            lam_L, lam_H = lambda_pred_dict[pid]

            # 解析解预测浓度
            z = torch.tensor(d['z'], dtype=torch.float32, device=self.device)
            CL_atm = d['CL_atm']
            CH_atm = d['CH_atm']
            CL_parent = d['CL_parent']
            CH_parent = d['CH_parent']

            CL_pred = CL_parent + (CL_atm - CL_parent) * torch.exp(-lam_L * z)
            CH_pred = CH_parent + (CH_atm - CH_parent) * torch.exp(-lam_H * z)

            # 数据损失（相对MSE，避免浓度尺度差异）
            CL_obs = torch.tensor(d['CL'], dtype=torch.float32, device=self.device)
            CH_obs = torch.tensor(d['CH'], dtype=torch.float32, device=self.device)

            rel_err_L = ((CL_pred - CL_obs) / (CL_obs + 1.0)) ** 2
            rel_err_H = ((CH_pred - CH_obs) / (CH_obs + 1.0)) ** 2
            profile_loss = torch.mean(rel_err_L) + torch.mean(rel_err_H)
            loss_data_list.append(profile_loss)

            # 物理约束：λ_H > λ_L (HREE对粘土吸附更强)
            phys_violation = torch.clamp(lam_L - lam_H, min=0) ** 2
            loss_phys_list.append(phys_violation)

        L_data = torch.stack(loss_data_list).mean()
        L_phys = torch.stack(loss_phys_list).mean()

        # 正则化：防止λ过大
        L_reg = 0.0
        for lam_L, lam_H in lambda_pred_dict.values():
            L_reg += torch.clamp(lam_L - 10.0, min=0) ** 2
            L_reg += torch.clamp(lam_H - 10.0, min=0) ** 2
            L_reg += torch.clamp(0.01 - lam_L, min=0) ** 2
            L_reg += torch.clamp(0.01 - lam_H, min=0) ** 2

        return 1.0 * L_data + 10.0 * L_phys + 0.01 * L_reg, L_data, L_phys

    def step(self, profile_data=None):
        if profile_data is None:
            profile_data = self.profile_data

        self.opt.zero_grad()
        lambda_pred_dict = {}

        for pid in self.pids:
            d = profile_data[pid]
            # 使用标量输入（每个剖面共享气候和粘土，用均值）
            x = torch.tensor([[
                float(d['z_norm'].mean()),  # 深度用均值
                float(d['Tn']),
                float(d['Pn']),
                float(d['Rn']),
                float(d['clay_norm'])
            ]], dtype=torch.float32, device=self.device)

            lam = self.pinn(x).squeeze()
            lambda_pred_dict[pid] = (lam[0], lam[1])

        loss, L_data, L_phys = self.compute_loss(lambda_pred_dict, profile_data)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.pinn.parameters(), 1.0)
        self.opt.step()
        self.sched.step(L_data)

        return {
            'total': loss.item(),
            'data': L_data.item(),
            'phys': L_phys.item(),
        }

    def predict(self, profile_data):
        """预测并返回λ参数和浓度"""
        self.pinn.eval()
        results = {}

        with torch.no_grad():
            for pid, d in profile_data.items():
                x = torch.tensor([[
                    float(d['z_norm'].mean()),
                    float(d['Tn']),
                    float(d['Pn']),
                    float(d['Rn']),
                    float(d['clay_norm'])
                ]], dtype=torch.float32, device=self.device)

                lam = self.pinn(x).squeeze()
                lam_L, lam_H = lam[0].item(), lam[1].item()

                # 解析解计算浓度
                z = d['z']
                CL_pred = analytical_C(z, d['CL_atm'], d['CL_parent'], lam_L)
                CH_pred = analytical_C(z, d['CH_atm'], d['CH_parent'], lam_H)
                LH_pred = CL_pred / (CH_pred + 1e-6)

                results[pid] = {
                    'lam_L': lam_L, 'lam_H': lam_H,
                    'CL_pred': CL_pred, 'CH_pred': CH_pred, 'LH_pred': LH_pred,
                    'CL_obs': d['CL'], 'CH_obs': d['CH'], 'L_H': d['L_H'],
                    'z': z,
                }

        return results


# ============================================================================
# 留一交叉验证
# ============================================================================
def leave_one_out_cv(profile_data, hidden=32, n_layers=3, n_iter=5000, device='cpu'):
    """留一交叉验证"""
    pids = list(profile_data.keys())
    cv_results = {}

    for leave_pid in pids:
        print(f"\n--- CV: 剔除 Profile {leave_pid} ---")

        # 训练数据（不包含被剔除的profile）
        train_data = {pid: profile_data[pid] for pid in pids if pid != leave_pid}

        # 初始化网络
        pinn = LambdaPINN(hidden=hidden, n_layers=n_layers)
        trainer = Trainer(pinn, train_data, device=device)

        # 训练
        for i in range(n_iter + 1):
            m = trainer.step()
            if i % 1000 == 0:
                print(f"  Iter {i}: loss={m['total']:.4f}, data={m['data']:.4f}, phys={m['phys']:.4f}")

        # 在被剔除的profile上验证
        val_data = {leave_pid: profile_data[leave_pid]}
        val_results = trainer.predict(val_data)

        # 计算指标
        r = val_results[leave_pid]
        if len(r['z']) > 2:
            r2_CL = pearsonr(r['CL_obs'], r['CL_pred'])[0] ** 2
            r2_CH = pearsonr(r['CH_obs'], r['CH_pred'])[0] ** 2
            r2_LH = pearsonr(r['L_H'], r['LH_pred'])[0] ** 2
        else:
            r2_CL = r2_CH = r2_LH = np.nan

        rmse_CL = np.sqrt(np.mean((r['CL_pred'] - r['CL_obs']) ** 2))
        rmse_CH = np.sqrt(np.mean((r['CH_pred'] - r['CH_obs']) ** 2))

        cv_results[leave_pid] = {
            'lam_L': r['lam_L'], 'lam_H': r['lam_H'],
            'r2_CL_cv': r2_CL, 'r2_CH_cv': r2_CH, 'r2_LH_cv': r2_LH,
            'rmse_CL': rmse_CL, 'rmse_CH': rmse_CH,
            'CL_pred': r['CL_pred'], 'CH_pred': r['CH_pred'], 'LH_pred': r['LH_pred'],
        }

        print(f"  CV结果: R²(L/H)={r2_LH:.3f}, R²(L)={r2_CL:.3f}, R²(H)={r2_CH:.3f}")

    return cv_results


# ============================================================================
# 主程序
# ============================================================================
def main():
    print('=' * 60)
    print('REE PINN v15 — 混合参数预测网络')
    print('=' * 60)

    # ---- 加载数据 ----
    df = pd.read_csv(DATA)
    try:
        dfl = pd.read_csv(LIT)
        df = df.merge(dfl[['id', 'parent_rock']], left_on='literature_id',
                      right_on='id', how='left', suffixes=('', '_l'))
        df.parent_rock = df.parent_rock.fillna(df.Bedrock)
    except:
        pass

    # 计算LREE, HREE, L/H
    df['CL'] = df[LREE_COLS].sum(axis=1)
    df['CH'] = df[HREE_COLS].sum(axis=1)
    df['L_H'] = df['CL'] / (df['CH'] + 1e-6)

    # ---- 按剖面组织数据 ----
    profile_data = {}
    for lid, grp_orig in df.groupby('literature_id'):
        # 分离母岩样品和非母岩样品
        parent_mask = grp_orig['Horizon'].str.contains('Parent', case=False, na=False)

        if parent_mask.any():
            # 有母岩样品：使用母岩样品的浓度
            parent_grp = grp_orig[parent_mask]
            CL_parent = float(parent_grp['CL'].mean())
            CH_parent = float(parent_grp['CH'].mean())
            print(f"Profile {int(lid)}: 使用母岩样品 (n_parent={parent_mask.sum()})")
        else:
            # 无母岩样品：使用最深样品
            grp = grp_orig.dropna(subset=['Depth_m'])
            grp = grp[grp['Depth_m'] > 0].sort_values('Depth_m')
            if len(grp) > 0:
                CL_parent = float(grp['CL'].iloc[-1])
                CH_parent = float(grp['CH'].iloc[-1])
            else:
                CL_parent = float(grp_orig['CL'].mean())
                CH_parent = float(grp_orig['CH'].mean())

        # 风化壳样品（排除母岩样品，用于训练）
        if parent_mask.any():
            grp = grp_orig[~parent_mask].dropna(subset=['Depth_m'])
            grp = grp[grp['Depth_m'] > 0].sort_values('Depth_m')
        else:
            grp = grp_orig.dropna(subset=['Depth_m'])
            grp = grp[grp['Depth_m'] > 0].sort_values('Depth_m')

        if len(grp) == 0:
            continue

        z = grp['Depth_m'].values.astype(float)
        CL = grp['CL'].values.astype(float)
        CH = grp['CH'].values.astype(float)
        LH = grp['L_H'].values.astype(float)

        # 表层浓度（深度最小的几个样品均值）
        n_surf = min(3, len(z))
        CL_atm = float(np.mean(CL[:n_surf]))
        CH_atm = float(np.mean(CH[:n_surf]))

        # 气候参数
        T_K = grp.T_annual_mean_K.iloc[0]
        P_mm = grp.P_annual_mean_mm_yr.iloc[0]
        R_m = grp.runoff_m_yr.iloc[0]
        T_C = T_K - 273.15

        # 粘土矿物（尝试从数据获取）
        clay_cols = ['Kaolinite', 'Vermiculite', 'Illite', 'Halloysite']
        measured_clay = grp[clay_cols].sum(axis=1).dropna().values if any(c in grp.columns for c in clay_cols) else None

        # 估算/映射粘土矿物
        z_max = max(z)
        clay_by_depth = map_clay_to_profile(z, lid, T_C, P_mm, measured_clay)
        total_clay_arr = np.array([c['total'] for c in clay_by_depth])
        clay_norm = float(np.mean(total_clay_arr)) / CLAY_MAX

        profile_data[int(lid)] = {
            'z': z, 'CL': CL, 'CH': CH, 'L_H': LH,
            'CL_atm': CL_atm, 'CH_atm': CH_atm,
            'CL_parent': CL_parent, 'CH_parent': CH_parent,
            'T_C': T_C, 'P_mm': P_mm, 'R_m': R_m,
            'Tn': (T_K - T_MEAN) / T_STD,
            'Pn': (P_mm - P_MEAN) / P_STD,
            'Rn': (R_m - R_MEAN) / R_STD,
            'z_norm': z / Z_MAX,
            'clay_norm': clay_norm,
            'name': f"[{int(lid)}]",
            'n': len(z),
        }

    print(f'\n数据: {len(df)} samples, {len(profile_data)} profiles')
    for pid, d in profile_data.items():
        print(f"  {d['name']}: n={d['n']}, "
              f"T={d['T_C']:.1f}C, P={d['P_mm']:.0f}mm, "
              f"clay_norm={d['clay_norm']:.3f}")

    # ---- 训练所有数据 ----
    print('\n=== 训练 (所有数据) ===')
    device = torch.device('cpu')
    pinn = LambdaPINN(hidden=32, n_layers=3).to(device)
    trainer = Trainer(pinn, profile_data, device=device)

    history = []
    for i in range(5000 + 1):
        m = trainer.step()
        history.append(m)
        if i % 1000 == 0:
            print(f"Iter {i}: loss={m['total']:.4f}, data={m['data']:.4f}, phys={m['phys']:.4f}")

    # ---- 评估 ----
    print('\n=== 训练集结果 ===')
    train_results = trainer.predict(profile_data)
    for pid, r in train_results.items():
        d = profile_data[pid]
        r2_CL = pearsonr(r['CL_obs'], r['CL_pred'])[0] ** 2
        r2_CH = pearsonr(r['CH_obs'], r['CH_pred'])[0] ** 2
        r2_LH = pearsonr(r['L_H'], r['LH_pred'])[0] ** 2
        print(f"  {d['name']}: λ_L={r['lam_L']:.4f}, λ_H={r['lam_H']:.4f}, "
              f"R²(L/H)={r2_LH:.3f}, R²(L)={r2_CL:.3f}, R²(H)={r2_CH:.3f}")

    # ---- LOO-CV ----
    print('\n=== 留一交叉验证 ===')
    cv_results = leave_one_out_cv(profile_data, hidden=32, n_layers=3,
                                  n_iter=5000, device=device)

    avg_r2_LH = np.nanmean([r['r2_LH_cv'] for r in cv_results.values()])
    avg_r2_CL = np.nanmean([r['r2_CL_cv'] for r in cv_results.values()])
    avg_r2_CH = np.nanmean([r['r2_CH_cv'] for r in cv_results.values()])
    print(f'\n平均CV R²: L/H={avg_r2_LH:.3f}, LREE={avg_r2_CL:.3f}, HREE={avg_r2_CH:.3f}')

    # ---- 绘图 ----
    plot_results(profile_data, train_results, cv_results, history, SAVE, pinn)

    # ---- 保存模型 ----
    torch.save(pinn.state_dict(), f'{SAVE}/model_v15.pt')
    print(f'\n模型已保存: {SAVE}/model_v15.pt')


# ============================================================================
# 绘图
# ============================================================================
def plot_results(profile_data, train_results, cv_results, history, save_dir, pinn):
    n = len(profile_data)
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    names = {
        1: 'Nagasawa\n花岗岩\nT=13C',
        2: 'Li2019\nA型花岗\nT=18C',
        3: 'Li&Zhou\nA型花岗\nT=18C',
        4: 'Fu2019\n流纹岩\nT=20C',
        5: 'Yaraghi\n二云母花岗\nT=24C',
        6: 'Luo\n闪长岩\nT=22C',
    }
    colors = plt.cm.tab10(np.linspace(0, 1, n))

    # ---- 图1: λ_L vs λ_H ----
    ax = fig.add_subplot(gs[0, 0])
    for pid in sorted(profile_data.keys()):
        r = train_results[pid]
        ax.scatter(r['lam_L'], r['lam_H'], c=[colors[pid - 1]], s=120, zorder=5)
        ax.annotate(str(pid), (r['lam_L'], r['lam_H']), fontsize=10, fontweight='bold')
    lmax = 5
    ax.plot([0, lmax], [0, lmax], 'k--', lw=1, label='λ_L = λ_H')
    ax.set_xlabel('λ_L (LREE)')
    ax.set_ylabel('λ_H (HREE)')
    ax.set_title('λ_L vs λ_H\n(应在下方 = HREE吸附更强)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, lmax)
    ax.set_ylim(0, lmax)

    # ---- 图2: 训练曲线 ----
    ax = fig.add_subplot(gs[0, 1])
    losses = np.array([h['total'] for h in history])
    ax.semilogy(losses, lw=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Total Loss')
    ax.set_title('Training Curve')
    ax.grid(alpha=0.3)

    # ---- 图3: CV R² ----
    ax = fig.add_subplot(gs[0, 2])
    pids = sorted(cv_results.keys())
    x_pos = np.arange(len(pids))
    w = 0.25
    r2_CL_cv = [cv_results[p]['r2_CL_cv'] for p in pids]
    r2_CH_cv = [cv_results[p]['r2_CH_cv'] for p in pids]
    r2_LH_cv = [cv_results[p]['r2_LH_cv'] for p in pids]
    ax.bar(x_pos - w, r2_CL_cv, w, label='LREE CV R²', color='blue', alpha=0.7)
    ax.bar(x_pos, r2_CH_cv, w, label='HREE CV R²', color='green', alpha=0.7)
    ax.bar(x_pos + w, r2_LH_cv, w, label='L/H CV R²', color='red', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(p) for p in pids])
    ax.set_xlabel('Profile ID')
    ax.set_ylabel('CV R²')
    ax.set_title('Leave-One-Out CV R²')
    ax.axhline(0.5, color='orange', ls='--', lw=1)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis='y')

    # ---- 剖面拟合图 ----
    for idx, pid in enumerate(sorted(profile_data.keys())):
        row = 1 + idx // 3
        col = idx % 3
        if row > 2:
            break
        ax = fig.add_subplot(gs[row, col])
        d = profile_data[pid]
        r = train_results[pid]
        cv = cv_results[pid]
        z = d['z']

        # 平滑曲线
        z_sm = np.linspace(0, max(z) * 1.05, 200)
        CL_fit = analytical_C(z_sm, d['CL_atm'], d['CL_parent'], r['lam_L'])
        CH_fit = analytical_C(z_sm, d['CH_atm'], d['CH_parent'], r['lam_H'])

        # 观测点
        ax.scatter(d['CL'], d['z'], c='blue', s=40, alpha=0.7,
                  label='LREE_obs', zorder=5)
        ax.scatter(d['CH'], d['z'], c='green', s=40, alpha=0.7,
                  label='HREE_obs', marker='^', zorder=5)

        # 拟合曲线
        ax.plot(CL_fit, z_sm, 'b--', lw=2, label=f'LREE_fit (λ={r["lam_L"]:.3f})')
        ax.plot(CH_fit, z_sm, 'g--', lw=2, label=f'HREE_fit (λ={r["lam_H"]:.3f})')

        ax.invert_yaxis()
        ax.set_xlabel('Concentration (ppm)')
        ax.set_ylabel('Depth (m)')
        ax.set_title(f'{names.get(pid, str(pid))}\n'
                     f'R²(L/H)={r["lam_L"]:.3f}/{r["lam_H"]:.3f}, '
                     f'CV R²(L/H)={cv["r2_LH_cv"]:.3f}')
        ax.legend(fontsize=6)
        ax.set_xlim(0)
        ax.grid(alpha=0.2)

    # ---- 图7: L/H比值剖面 ----
    ax = fig.add_subplot(gs[2, 0])
    for pid in sorted(profile_data.keys()):
        d = profile_data[pid]
        r = train_results[pid]
        z_sm = np.linspace(0, max(d['z']) * 1.05, 200)
        LH_fit = analytical_LH_ratio(z_sm, d['CL_atm'], d['CH_atm'],
                                      d['CL_parent'], d['CH_parent'],
                                      r['lam_L'], r['lam_H'])
        ax.plot(LH_fit, z_sm, '-', lw=2, color=colors[pid - 1], alpha=0.8,
                label=f'[{pid}]')
        ax.scatter(d['L_H'], d['z'], c=[colors[pid - 1]], s=20, alpha=0.5)
    ax.set_xlabel('LREE/HREE')
    ax.set_ylabel('Depth (m)')
    ax.set_title('L/H Ratio Profiles')
    ax.legend(fontsize=7)
    ax.invert_yaxis()
    ax.grid(alpha=0.2)

    # ---- 图8: 气候 vs λ ----
    ax = fig.add_subplot(gs[2, 1])
    pids = sorted(profile_data.keys())
    T_arr = np.array([profile_data[p]['T_C'] for p in pids])
    lL_arr = np.array([train_results[p]['lam_L'] for p in pids])
    lH_arr = np.array([train_results[p]['lam_H'] for p in pids])
    ax.scatter(T_arr, lL_arr, c='blue', s=80, label='λ_L', zorder=5)
    ax.scatter(T_arr, lH_arr, c='green', s=80, label='λ_H', marker='^', zorder=5)
    for i, pid in enumerate(pids):
        ax.annotate(str(pid), (T_arr[i], lL_arr[i]), fontsize=8, color='blue')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('lambda')
    ax.set_title('λ vs Temperature')
    ax.legend()
    ax.grid(alpha=0.3)

    # ---- 图9: 粘土 vs λ ----
    ax = fig.add_subplot(gs[2, 2])
    clay_arr = np.array([profile_data[p]['clay_norm'] for p in pids])
    ax.scatter(clay_arr, lL_arr, c='blue', s=80, label='λ_L', zorder=5)
    ax.scatter(clay_arr, lH_arr, c='green', s=80, label='λ_H', marker='^', zorder=5)
    for i, pid in enumerate(pids):
        ax.annotate(str(pid), (clay_arr[i], lL_arr[i]), fontsize=8, color='blue')
    ax.set_xlabel('Clay Content (normalized)')
    ax.set_ylabel('lambda')
    ax.set_title('λ vs Clay')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle('REE PINN v15 — Hybrid Parameter Prediction\n'
                 'C(z) = C_parent + (C_atm - C_parent) · exp(-λ·z)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/results_v15.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nFigure: {save_dir}/results_v15.png')


if __name__ == '__main__':
    main()
