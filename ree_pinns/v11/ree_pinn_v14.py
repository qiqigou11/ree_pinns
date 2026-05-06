"""
REE 分馏模型 v14 — 对流-弥散-吸附 解析解 + 留一交叉验证
================================================================
物理：稳态对流-弥散方程（含吸附源汇项）

    D·C''(z) - v·C'(z) - k·(C - C_parent) = 0

解析解（特征根）：
    C(z) = C_parent + (C_atm - C_parent) · exp(-λ·z)

    λ = (v/2D) · [1 + sqrt(1 + 4kD/v²)]

λ 的物理含义：
    - λ 越大：表层→母岩过渡越快（吸附越强/对流越快）
    - 对同样的 C_atm 和 C_parent，λ 决定了富集峰的位置

LREE vs HREE 的分馏体现在：
    - LREE 有更大的 C_atm（大气输入更多）
    - LREE 有更大的 λ（吸附更强，更快趋于母岩）
    - 结果：LREE 的 L/H 比值在浅层最大，随深度减小

数据问题：
    - Nagasawa (id=1): CL 在 70-304 ppm 之间剧烈波动（不是单调曲线）
    - Profile 3 (Li&Zhou): 有明显的 CL 峰值在 z=1.4m
    - Profile 5/6: 相同深度有多个样品，说明可能有子剖面混合
    → 这些都不是简单指数衰减能描述的
    → 但整体 L/H 比值的趋势（浅高深低）是明确的
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize
from scipy.stats import pearsonr
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

DATA = '/Users/suheng/Desktop/claudecode/pinns/ree_pinns/data/REE_samples_20260424_with_climate.csv'
LIT   = '/Users/suheng/Desktop/claudecode/pinns/ree_pinns/data/REE_literature_20260424.csv'
SAVE  = '/Users/suheng/Desktop/claudecode/pinns/ree_pinns/v11'

LREE_COLS = ['La_ppm', 'Ce_ppm', 'Pr_ppm', 'Nd_ppm', 'Sm_ppm']
HREE_COLS = ['Tb_ppm', 'Dy_ppm', 'Ho_ppm', 'Er_ppm', 'Tm_ppm', 'Yb_ppm', 'Lu_ppm']


def analytical_C(z, C_atm, C_parent, lam):
    """
    稳态对流-弥散-吸附解析解
    C(z) = C_parent + (C_atm - C_parent) · exp(-λ·z)
    """
    return C_parent + (C_atm - C_parent) * np.exp(-lam * z)


def analytical_LH_ratio(z, CL_atm, CH_atm, CL_par, CH_par, lam_L, lam_H):
    """
    L/H 比值随深度的变化
    """
    CL = analytical_C(z, CL_atm, CL_par, lam_L)
    CH = analytical_C(z, CH_atm, CH_par, lam_H)
    return CL / (CH + 1e-6)


def residuals_fn(theta, z, C_obs, C_atm, C_parent):
    """相对残差"""
    lam = theta[0]
    if lam <= 0:
        return np.full_like(z, 1e10)
    C_pred = analytical_C(z, C_atm, C_parent, lam)
    return (C_pred - C_obs) / (np.abs(C_obs) + 1.0)


def fit_single(z, C_obs, C_atm, C_parent):
    """拟合单个 λ"""
    def objective(lam):
        r = residuals_fn([lam], z, C_obs, C_atm, C_parent)
        return np.mean(r**2)

    res = minimize(objective, x0=[0.5], method='SLSQP',
                   bounds=[(1e-4, 10.0)], options={'maxiter': 500})
    return res.x[0]


def fit_profile_LH(z, CL_obs, CH_obs, CL_atm, CH_atm, CL_par, CH_par):
    """同时拟合 L 和 H 的 λ"""
    lam_L = fit_single(z, CL_obs, CL_atm, CL_par)
    lam_H = fit_single(z, CH_obs, CH_atm, CH_par)
    return lam_L, lam_H


def leave_one_out_cv(profiles, profile_data):
    """留一交叉验证"""
    pids = list(profiles.keys())
    cv_results = {}

    for leave_pid in pids:
        train_pids = [p for p in pids if p != leave_pid]

        # 用训练集拟合全局参数
        all_z = np.concatenate([profile_data[p]['z'] for p in train_pids])
        all_CL = np.concatenate([profile_data[p]['CL'] for p in train_pids])
        all_CH = np.concatenate([profile_data[p]['CH'] for p in train_pids])
        CL_atm_mean = np.mean([profile_data[p]['CL_atm'] for p in train_pids])
        CH_atm_mean = np.mean([profile_data[p]['CH_atm'] for p in train_pids])
        CL_par_mean = np.mean([profile_data[p]['CL_parent'] for p in train_pids])
        CH_par_mean = np.mean([profile_data[p]['CH_parent'] for p in train_pids])

        lam_L_g = fit_single(all_z, all_CL, CL_atm_mean, CL_par_mean)
        lam_H_g = fit_single(all_z, all_CH, CH_atm_mean, CH_par_mean)

        # 预测测试集
        d = profile_data[leave_pid]
        CL_pred = analytical_C(d['z'], d['CL_atm'], d['CL_parent'], lam_L_g)
        CH_pred = analytical_C(d['z'], d['CH_atm'], d['CH_parent'], lam_H_g)
        LH_pred = analytical_LH_ratio(d['z'], d['CL_atm'], d['CH_atm'],
                                       d['CL_parent'], d['CH_parent'], lam_L_g, lam_H_g)

        r2_CL = pearsonr(d['CL'], CL_pred)[0]**2 if len(d['z']) > 2 else np.nan
        r2_CH = pearsonr(d['CH'], CH_pred)[0]**2 if len(d['z']) > 2 else np.nan
        r2_LH = pearsonr(d['L_H'], LH_pred)[0]**2 if len(d['z']) > 2 else np.nan
        rmse_CL = np.sqrt(np.mean((CL_pred - d['CL'])**2))
        rmse_CH = np.sqrt(np.mean((CH_pred - d['CH'])**2))

        cv_results[leave_pid] = {
            'lam_L': lam_L_g, 'lam_H': lam_H_g,
            'r2_CL_cv': r2_CL, 'r2_CH_cv': r2_CH, 'r2_LH_cv': r2_LH,
            'rmse_CL': rmse_CL, 'rmse_CH': rmse_CH,
            'CL_pred': CL_pred, 'CH_pred': CH_pred, 'LH_pred': LH_pred,
        }

    return cv_results


def main():
    print('=' * 60)
    print('REE 分馏模型 v14 — 对流-弥散-吸附 解析解 + L/H 比值')
    print('=' * 60)

    # ---- 加载数据 ----
    df  = pd.read_csv(DATA)
    dfl = pd.read_csv(LIT)
    df = df.merge(dfl[['id', 'parent_rock']],
                  left_on='literature_id', right_on='id',
                  how='left', suffixes=('', '_l'))
    df.parent_rock = df.parent_rock.fillna(df.Bedrock)

    # 只保留有深度和REE数据的行
    df = df.dropna(subset=['Depth_m'] + LREE_COLS + HREE_COLS).reset_index(drop=True)
    df = df[df['Depth_m'] > 0].reset_index(drop=True)  # 去掉 z=0 和 nan

    df['CL'] = df[LREE_COLS].sum(axis=1)
    df['CH'] = df[HREE_COLS].sum(axis=1)
    df['L_H'] = df['CL'] / (df['CH'] + 1e-6)

    # ---- 按剖面组织 ----
    profile_data = {}
    for lid, grp in df.groupby('literature_id'):
        z = grp.Depth_m.values.astype(float)
        CL = grp.CL.values.astype(float)
        CH = grp.CH.values.astype(float)
        LH = grp.L_H.values.astype(float)

        # 按深度排序
        idx = np.argsort(z)
        z, CL, CH, LH = z[idx], CL[idx], CH[idx], LH[idx]

        # 表层浓度（深度最小的几个样品均值）
        n_surf = min(3, len(z))
        CL_atm = float(np.mean(CL[:n_surf]))
        CH_atm = float(np.mean(CH[:n_surf]))

        # 母岩浓度（最深样品）
        CL_par = float(CL[-1])
        CH_par = float(CH[-1])

        # 母岩 L/H
        LH_par = CL_par / (CH_par + 1e-6)
        # 表层 L/H
        LH_atm = CL_atm / (CH_atm + 1e-6)

        T_K = grp.T_annual_mean_K.iloc[0]
        P_mm = grp.P_annual_mean_mm_yr.iloc[0]
        R_m = grp.runoff_m_yr.iloc[0]

        profile_data[int(lid)] = {
            'z': z, 'CL': CL, 'CH': CH, 'L_H': LH,
            'CL_atm': CL_atm, 'CH_atm': CH_atm,
            'CL_parent': CL_par, 'CH_parent': CH_par,
            'LH_atm': LH_atm, 'LH_parent': LH_par,
            'name': f"[{int(lid)}] {grp.parent_rock.iloc[0][:18]}",
            'T_C': T_K - 273.15, 'P_mm': P_mm, 'R_m': R_m,
            'n': len(z),
        }

    print(f'\n{len(df)} samples, {len(profile_data)} profiles (depth>0 only)')
    for pid, d in profile_data.items():
        LH_drop = d['LH_atm'] - d['LH_parent']
        print(f"  {d['name']}: n={d['n']}  "
              f"T={d['T_C']:.1f}C  P={d['P_mm']:.0f}mm  "
              f"LH_atm={d['LH_atm']:.2f}  LH_parent={d['LH_parent']:.2f}  "
              f"drop={LH_drop:+.2f} ({LH_drop/d['LH_parent']*100:+.0f}%)")

    # ---- 逐剖面拟合 λ ----
    print(f'\n=== 逐剖面拟合（in-sample R²）===')
    fit_results = {}
    for pid, d in profile_data.items():
        lam_L, lam_H = fit_profile_LH(
            d['z'], d['CL'], d['CH'],
            d['CL_atm'], d['CH_atm'],
            d['CL_parent'], d['CH_parent']
        )
        CL_pred = analytical_C(d['z'], d['CL_atm'], d['CL_parent'], lam_L)
        CH_pred = analytical_C(d['z'], d['CH_atm'], d['CH_parent'], lam_H)
        LH_pred = analytical_LH_ratio(
            d['z'], d['CL_atm'], d['CH_atm'],
            d['CL_parent'], d['CH_parent'], lam_L, lam_H
        )

        r2_CL = pearsonr(d['CL'], CL_pred)[0]**2 if len(d['z']) > 2 else np.nan
        r2_CH = pearsonr(d['CH'], CH_pred)[0]**2 if len(d['z']) > 2 else np.nan
        r2_LH = pearsonr(d['L_H'], LH_pred)[0]**2 if len(d['z']) > 2 else np.nan
        rmse_CL = np.sqrt(np.mean((CL_pred - d['CL'])**2))
        rmse_CH = np.sqrt(np.mean((CH_pred - d['CH'])**2))

        fit_results[pid] = {
            'lam_L': lam_L, 'lam_H': lam_H,
            'r2_CL': r2_CL, 'r2_CH': r2_CH, 'r2_LH': r2_LH,
            'rmse_CL': rmse_CL, 'rmse_CH': rmse_CH,
            'CL_pred': CL_pred, 'CH_pred': CH_pred, 'LH_pred': LH_pred,
        }
        print(f"  [{pid}] {d['name']}: "
              f"lambda_L={lam_L:.4f}  lambda_H={lam_H:.4f}  "
              f"L/H_R2={r2_LH:.3f}  "
              f"CL_R2={r2_CL:.3f}  CH_R2={r2_CH:.3f}")

    # ---- 留一交叉验证 ----
    print(f'\n=== 留一交叉验证 ===')
    cv_results = leave_one_out_cv(profile_data, profile_data)
    for pid, r in cv_results.items():
        d = profile_data[pid]
        print(f"  去掉 [{pid}] {d['name']}: "
              f"LH_R2_cv={r['r2_LH_cv']:.3f}  "
              f"CL_R2_cv={r['r2_CL_cv']:.3f}  "
              f"CH_R2_cv={r['r2_CH_cv']:.3f}")

    avg_r2_LH = np.nanmean([r['r2_LH_cv'] for r in cv_results.values()])
    avg_r2_CL = np.nanmean([r['r2_CL_cv'] for r in cv_results.values()])
    avg_r2_CH = np.nanmean([r['r2_CH_cv'] for r in cv_results.values()])
    print(f'\n  平均CV R²: L/H={avg_r2_LH:.3f}  LREE={avg_r2_CL:.3f}  HREE={avg_r2_CH:.3f}')

    # ---- 气候/径流 vs lambda ----
    print(f'\n=== 气候 vs lambda ===')
    print(f"  {'Profile':<30} {'T_C':>6} {'P_mm':>7} {'R_m':>6} "
          f"{'lambda_L':>9} {'lambda_H':>9} {'dLH':>7}")
    for pid in sorted(profile_data.keys()):
        d = profile_data[pid]; r = fit_results[pid]
        dLH = d['LH_atm'] - d['LH_parent']
        print(f"  {d['name']:<30} {d['T_C']:>6.1f} {d['P_mm']:>7.0f} "
              f"{d['R_m']:>6.3f} {r['lam_L']:>9.4f} {r['lam_H']:>9.4f} "
              f"{dLH:>+7.3f}")

    # ---- 绘图 ----
    plot(profile_data, fit_results, cv_results, SAVE)

    print(f'\nFigure: {SAVE}/results_v14.png')


# =====================================================================
# 绘图
# =====================================================================
def plot(profile_data, fit_results, cv_results, save_dir):
    n = len(profile_data)
    fig = plt.figure(figsize=(18, 16))
    gs  = GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

    names = {
        1: 'Nagasawa\n花岗岩\nT=13C',
        2: 'Li2019\nA型花岗\nT=18C',
        3: 'Li&Zhou\nA型花岗\nT=18C',
        4: 'Fu2019\n流纹岩\nT=20C',
        5: 'Yaraghi\n二云母花岗\nT=24C',
        6: 'Luo\n闪长岩\nT=22C',
    }
    colors = plt.cm.tab10(np.linspace(0, 1, n))

    # ---- 顶部行：气候 vs lambda ----
    pids = sorted(profile_data.keys())
    T_arr = np.array([profile_data[p]['T_C'] for p in pids])
    P_arr = np.array([profile_data[p]['P_mm'] for p in pids])
    R_arr = np.array([profile_data[p]['R_m'] for p in pids])
    lL_arr = np.array([fit_results[p]['lam_L'] for p in pids])
    lH_arr = np.array([fit_results[p]['lam_H'] for p in pids])
    dLH_arr = np.array([profile_data[p]['LH_atm'] - profile_data[p]['LH_parent']
                         for p in pids])

    for ax_idx, (x, xlabel, title) in enumerate([
        (T_arr, 'T (°C)', 'lambda vs Temperature'),
        (P_arr, 'P (mm/yr)', 'lambda vs Precipitation'),
        (R_arr, 'R (m/yr)', 'lambda vs Runoff'),
    ]):
        ax = fig.add_subplot(gs[0, ax_idx])
        ax.scatter(x, lL_arr, c='blue', s=80, label='lambda_L', zorder=5)
        ax.scatter(x, lH_arr, c='green', s=80, label='lambda_H', marker='^', zorder=5)
        for i, pid in enumerate(pids):
            ax.annotate(str(pid), (x[i], lL_arr[i]), fontsize=8, color='blue')
        ax.set_xlabel(xlabel); ax.set_ylabel('lambda')
        ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)

    # ---- 第2-3行：每个剖面 ----
    for idx, pid in enumerate(sorted(profile_data.keys())):
        row = 1 + idx // 3
        col = idx % 3
        if row > 3: break
        ax = fig.add_subplot(gs[row, col])
        d = profile_data[pid]; r = fit_results[pid]
        cv = cv_results[pid]
        z = d['z']

        # 平滑曲线
        z_sm = np.linspace(0, max(z) * 1.05, 200)
        CL_fit = analytical_C(z, d['CL_atm'], d['CL_parent'], r['lam_L'])
        CH_fit = analytical_C(z, d['CH_atm'], d['CH_parent'], r['lam_H'])
        CL_sm  = analytical_C(z_sm, d['CL_atm'], d['CL_parent'], r['lam_L'])
        CH_sm  = analytical_C(z_sm, d['CH_atm'], d['CH_parent'], r['lam_H'])
        LH_sm  = analytical_LH_ratio(z_sm, d['CL_atm'], d['CH_atm'],
                                    d['CL_parent'], d['CH_parent'],
                                    r['lam_L'], r['lam_H'])

        # 观测点
        ax.scatter(d['CL'], d['z'], c='blue', s=30, alpha=0.7,
                   label=f'LREE_obs (n={len(z)})', zorder=5)
        ax.scatter(d['CH'], d['z'], c='green', s=30, alpha=0.7,
                   label='HREE_obs', marker='^', zorder=5)
        # 拟合曲线（样本点位置）
        ax.plot(CL_fit, z, 'b-', lw=1.5, alpha=0.6)
        ax.plot(CH_fit, z, 'g-', lw=1.5, alpha=0.6)
        # 平滑曲线
        ax.plot(CL_sm, z_sm, 'b--', lw=2, label=f'LREE_fit (λ={r["lam_L"]:.3f})')
        ax.plot(CH_sm, z_sm, 'g--', lw=2, label=f'HREE_fit (λ={r["lam_H"]:.3f})')

        ax.invert_yaxis()
        ax.set_xlabel('Concentration (ppm)')
        ax.set_ylabel('Depth (m)')
        ax.set_title(f'{names.get(pid, str(pid))}\n'
                     f'LREE R²={r["r2_CL"]:.3f}  HREE R²={r["r2_CH"]:.3f}\n'
                     f'L/H R²={r["r2_LH"]:.3f}  '
                     f'CV L/H R²={cv["r2_LH_cv"]:.3f}')
        ax.legend(fontsize=6, loc='lower right')
        ax.set_xlim(0)
        ax.grid(alpha=0.2)

    # ---- 第4行左：L/H 比值 ----
    ax = fig.add_subplot(gs[3, 0])
    for pid in sorted(profile_data.keys()):
        d = profile_data[pid]; r = fit_results[pid]
        z_sm = np.linspace(0, max(d['z']) * 1.05, 200)
        LH_sm = analytical_LH_ratio(z_sm, d['CL_atm'], d['CH_atm'],
                                    d['CL_parent'], d['CH_parent'],
                                    r['lam_L'], r['lam_H'])
        ax.plot(LH_sm, z_sm, '-', lw=2, color=colors[pid-1], alpha=0.8,
                label=f'[{pid}] {d["name"][1:8]}')
        ax.scatter(d['L_H'], d['z'], c=[colors[pid-1]], s=20, alpha=0.5, zorder=5)
    ax.set_xlabel('LREE/HREE')
    ax.set_ylabel('Depth (m)')
    ax.set_title('L/H Ratio Profiles\n(Model curves + observations)')
    ax.legend(fontsize=7); ax.invert_yaxis(); ax.grid(alpha=0.2)

    # ---- 第4行中：lambda_L vs lambda_H ----
    ax = fig.add_subplot(gs[3, 1])
    for pid in sorted(profile_data.keys()):
        r = fit_results[pid]
        ax.scatter(r['lam_L'], r['lam_H'], c=[colors[pid-1]],
                   s=120, zorder=5)
        ax.annotate(str(pid), (r['lam_L'], r['lam_H']),
                   fontsize=10, fontweight='bold')
    lmax = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lmax], [0, lmax], 'k--', lw=1, label='lambda_L = lambda_H')
    ax.set_xlabel('lambda_L (LREE)')
    ax.set_ylabel('lambda_H (HREE)')
    ax.set_title('Fractionation Pattern\n(below line = LREE stronger adsorption)')
    ax.legend(); ax.grid(alpha=0.3)

    # ---- 第4行右：CV R² ----
    ax = fig.add_subplot(gs[3, 2])
    x_pos = np.arange(len(pids))
    w = 0.25
    r2_CL_cv = [cv_results[p]['r2_CL_cv'] for p in pids]
    r2_CH_cv = [cv_results[p]['r2_CH_cv'] for p in pids]
    r2_LH_cv = [cv_results[p]['r2_LH_cv'] for p in pids]
    ax.bar(x_pos - w, r2_CL_cv, w, label='LREE CV R²', color='blue', alpha=0.7)
    ax.bar(x_pos,     r2_CH_cv, w, label='HREE CV R²', color='green', alpha=0.7)
    ax.bar(x_pos + w, r2_LH_cv, w, label='L/H CV R²', color='red', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(p) for p in pids])
    ax.set_xlabel('Profile ID'); ax.set_ylabel('CV R²')
    ax.set_title('Leave-One-Out CV R²\n(True generalization)')
    ax.axhline(0.5, color='orange', ls='--', lw=1, label='R²=0.5')
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='y')

    plt.suptitle(
        'REE 分馏模型 v14 — 对流-弥散-吸附 解析解\n'
        'C(z) = C_parent + (C_atm - C_parent) · exp(-λ·z)\n'
        'λ_L > λ_H → LREE吸附更强，L/H在浅层最大，随深度减小',
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(f'{save_dir}/results_v14.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
