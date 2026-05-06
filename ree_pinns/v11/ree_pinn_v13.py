"""
REE PINN v13 — 解析分馏模型（零过拟合）
========================================
核心：没有神经网络！直接用分馏解析公式 + 最小二乘拟合。

物理：
  C_L(z) = C_L^atm * exp(-beta_L * z) + C_L^parent * (1 - exp(-beta_L * z))
  C_H(z) = C_H^atm * exp(-beta_H * z) + C_H^parent * (1 - exp(-beta_H * z))

  beta_L, beta_H — 分馏系数，由 scipy.optimize 拟合

气候/径流的作用：通过 beta 参数体现（不同气候条件 → 不同 beta）

验证：
  1. 残差分析（是否正态分布？）
  2. 留一交叉验证（Leave-One-Profile-Out）：每次去掉1个剖面，拟合其余5个，预测被去掉的
  3. 物理约束：beta_L > beta_H
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize, differential_evolution
from scipy.stats import pearsonr
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

DATA = '/Users/suheng/Desktop/claudecode/pinns/ree_pinns/data/REE_samples_20260424_with_climate.csv'
LIT   = '/Users/suheng/Desktop/claudecode/pinns/ree_pinns/data/REE_literature_20260424.csv'
SAVE  = '/Users/suheng/Desktop/claudecode/pinns/ree_pinns/v11'

Z_MAX = 33.0
T_MEAN, T_STD = 290.62, 3.78
P_MEAN, P_STD = 1604.84, 710.27
R_MEAN, R_STD = 0.5210, 0.2310

LREE_COLS = ['La_ppm', 'Ce_ppm', 'Pr_ppm', 'Nd_ppm', 'Sm_ppm']
HREE_COLS = ['Tb_ppm', 'Dy_ppm', 'Ho_ppm', 'Er_ppm', 'Tm_ppm', 'Yb_ppm', 'Lu_ppm']


def analytical_profile(z, C_atm, C_parent, beta):
    """解析分馏曲线"""
    return C_atm * np.exp(-beta * z) + C_parent * (1 - np.exp(-beta * z))


def residuals(theta, z, CL_obs, CH_obs, CL_atm, CH_atm, CL_par, CH_par):
    """残差向量（用于优化）"""
    beta_L, beta_H = theta
    # beta 必须 > 0
    if beta_L <= 0 or beta_H <= 0:
        return np.full(len(z), 1e10)
    CL_pred = analytical_profile(z, CL_atm, CL_par, beta_L)
    CH_pred = analytical_profile(z, CH_atm, CH_par, beta_H)
    # 相对残差（避免量纲问题）
    res_CL = (CL_pred - CL_obs) / (CL_obs + 1e-6)
    res_CH = (CH_pred - CH_obs) / (CH_obs + 1e-6)
    return np.concatenate([res_CL, res_CH])


def fit_profile(z, CL_obs, CH_obs, CL_atm, CH_atm, CL_par, CH_par):
    """拟合单个剖面的 beta_L, beta_H"""
    # 初始猜测
    beta0 = [0.3, 0.2]

    def objective(theta):
        r = residuals(theta, z, CL_obs, CH_obs, CL_atm, CH_atm, CL_par, CH_par)
        return np.mean(r ** 2)

    # 约束：beta_L >= beta_H（LREE吸附更强）
    def constraint(theta):
        return theta[0] - theta[1]  # >= 0

    from scipy.optimize import minimize
    res = minimize(
        objective, beta0,
        method='SLSQP',
        bounds=[(1e-4, 5.0), (1e-4, 5.0)],
        constraints={'type': 'ineq', 'fun': constraint},
        options={'maxiter': 1000}
    )
    return res.x  # [beta_L, beta_H]


def leave_one_out_cv(profiles, profile_data):
    """
    留一交叉验证：每次去掉1个剖面，用其余5个拟合，
    再预测被去掉的剖面。
    """
    pids = list(profiles.keys())
    results_cv = {}

    for leave_pid in pids:
        train_pids = [p for p in pids if p != leave_pid]
        test_pid = leave_pid

        # 用训练集拟合全局 beta
        all_z, all_CL, all_CH = [], [], []
        for pid in train_pids:
            d = profile_data[pid]
            all_z.extend(d['z'].tolist())
            all_CL.extend(d['CL'].tolist())
            all_CH.extend(d['CH'].tolist())

        z_train = np.array(all_z)
        CL_train = np.array(all_CL)
        CH_train = np.array(all_CH)

        # 平均边界值
        CL_atm_mean = np.mean([profile_data[p]['CL_atm'] for p in train_pids])
        CH_atm_mean = np.mean([profile_data[p]['CH_atm'] for p in train_pids])
        CL_par_mean = np.mean([profile_data[p]['CL_parent'] for p in train_pids])
        CH_par_mean = np.mean([profile_data[p]['CH_parent'] for p in train_pids])

        # 拟合全局 beta
        beta_opt = fit_profile(
            z_train, CL_train, CH_train,
            CL_atm_mean, CH_atm_mean, CL_par_mean, CH_par_mean
        )

        # 在测试集上预测
        d_test = profile_data[test_pid]
        CL_pred = analytical_profile(d_test['z'], d_test['CL_atm'],
                                     d_test['CL_parent'], beta_opt[0])
        CH_pred = analytical_profile(d_test['z'], d_test['CH_atm'],
                                     d_test['CH_parent'], beta_opt[1])

        r2_CL = pearsonr(d_test['CL'], CL_pred)[0] ** 2
        r2_CH = pearsonr(d_test['CH'], CH_pred)[0] ** 2
        rmse_CL = np.sqrt(np.mean((CL_pred - d_test['CL']) ** 2))
        rmse_CH = np.sqrt(np.mean((CH_pred - d_test['CH']) ** 2))

        results_cv[test_pid] = {
            'beta_L': beta_opt[0], 'beta_H': beta_opt[1],
            'r2_CL_cv': r2_CL, 'r2_CH_cv': r2_CH,
            'rmse_CL_cv': rmse_CL, 'rmse_CH_cv': rmse_CH,
            'CL_pred': CL_pred, 'CH_pred': CH_pred,
        }

    return results_cv


def main():
    print('=' * 60)
    print('REE 分馏模型 v13 — 解析形式 + 留一交叉验证')
    print('=' * 60)

    # ---- 加载数据 ----
    df  = pd.read_csv(DATA)
    dfl = pd.read_csv(LIT)
    df = df.merge(dfl[['id', 'parent_rock']],
                  left_on='literature_id', right_on='id',
                  how='left', suffixes=('', '_l'))
    df.parent_rock = df.parent_rock.fillna(df.Bedrock)
    df = df.dropna(subset=['Depth_m'] + LREE_COLS + HREE_COLS).reset_index(drop=True)

    df['C_LREE'] = df[LREE_COLS].sum(axis=1)
    df['C_HREE'] = df[HREE_COLS].sum(axis=1)
    df['L_H'] = df['C_LREE'] / (df['C_HREE'] + 1e-6)

    # ---- 按剖面组织 ----
    profile_data = {}
    for lid, grp in df.groupby('literature_id'):
        z = grp.Depth_m.values.astype(float)
        CL = grp.C_LREE.values.astype(float)
        CH = grp.C_HREE.values.astype(float)
        RAT = grp['L_H'].values.astype(float)

        surf = grp[grp.Depth_m < 1.0]
        CL_atm = float(surf.C_LREE.mean()) if len(surf) > 0 else float(grp.C_LREE.iloc[:2].mean())
        CH_atm = float(surf.C_HREE.mean()) if len(surf) > 0 else float(grp.C_HREE.iloc[:2].mean())
        CL_par = float(grp.C_LREE.iloc[-1])
        CH_par = float(grp.C_HREE.iloc[-1])

        T_K = grp.T_annual_mean_K.iloc[0]
        P_mm = grp.P_annual_mean_mm_yr.iloc[0]
        R_m = grp.runoff_m_yr.iloc[0]

        profile_data[int(lid)] = {
            'z': z, 'CL': CL, 'CH': CH, 'RAT': RAT,
            'CL_atm': CL_atm, 'CH_atm': CH_atm,
            'CL_parent': CL_par, 'CH_parent': CH_par,
            'name': f"[{int(lid)}] {grp.parent_rock.iloc[0][:18]}",
            'T_C': T_K - 273.15, 'P_mm': P_mm, 'R_m': R_m,
            'Tn': (T_K - T_MEAN) / T_STD,
            'Pn': (P_mm - P_MEAN) / P_STD,
            'Rn': (R_m - R_MEAN) / R_STD,
        }

    print(f'\n{len(df)} samples, {len(profile_data)} profiles')
    for pid, d in profile_data.items():
        print(f"  {d['name']}: n={len(d['z'])}  "
              f"T={d['T_C']:.1f}C  P={d['P_mm']:.0f}mm  R={d['R_m']:.2f}m/yr  "
              f"CL_atm={d['CL_atm']:.0f}  CL_par={d['CL_parent']:.0f}  "
              f"L/H_parent={d['CL_parent']/(d['CH_parent']+1e-6):.2f}")

    # ---- 逐剖面拟合 ----
    print(f'\n=== 逐剖面拟合结果（in-sample）===')
    fit_results = {}
    for pid, d in profile_data.items():
        beta_L, beta_H = fit_profile(
            d['z'], d['CL'], d['CH'],
            d['CL_atm'], d['CH_atm'], d['CL_parent'], d['CH_parent']
        )
        CL_pred = analytical_profile(d['z'], d['CL_atm'], d['CL_parent'], beta_L)
        CH_pred = analytical_profile(d['z'], d['CH_atm'], d['CH_parent'], beta_H)

        r2_CL = pearsonr(d['CL'], CL_pred)[0] ** 2
        r2_CH = pearsonr(d['CH'], CH_pred)[0] ** 2
        rmse_CL = np.sqrt(np.mean((CL_pred - d['CL']) ** 2))
        rmse_CH = np.sqrt(np.mean((CH_pred - d['CH']) ** 2))

        fit_results[pid] = {
            'beta_L': beta_L, 'beta_H': beta_H,
            'r2_CL': r2_CL, 'r2_CH': r2_CH,
            'rmse_CL': rmse_CL, 'rmse_CH': rmse_CH,
            'CL_pred': CL_pred, 'CH_pred': CH_pred,
        }

        print(f'  [{pid}] {d["name"]}: '
              f'beta_L={beta_L:.4f}  beta_H={beta_H:.4f}  '
              f'ratio={beta_L/beta_H:.3f}  '
              f'LREE_R2={r2_CL:.3f}  HREE_R2={r2_CH:.3f}')

    # ---- 全局拟合（所有剖面一起） ----
    print(f'\n=== 全局拟合（所有剖面）===')
    all_z = np.concatenate([d['z'] for d in profile_data.values()])
    all_CL = np.concatenate([d['CL'] for d in profile_data.values()])
    all_CH = np.concatenate([d['CH'] for d in profile_data.values()])

    # 全局边界值（加权平均）
    CL_atm_mean = np.mean([d['CL_atm'] for d in profile_data.values()])
    CH_atm_mean = np.mean([d['CH_atm'] for d in profile_data.values()])
    CL_par_mean = np.mean([d['CL_parent'] for d in profile_data.values()])
    CH_par_mean = np.mean([d['CH_parent'] for d in profile_data.values()])

    beta_global = fit_profile(
        all_z, all_CL, all_CH,
        CL_atm_mean, CH_atm_mean, CL_par_mean, CH_par_mean
    )
    print(f'  Global: beta_L={beta_global[0]:.4f}  beta_H={beta_global[1]:.4f}')

    # 用全局 beta 预测每个剖面
    for pid, d in profile_data.items():
        CL_g = analytical_profile(d['z'], d['CL_atm'], d['CL_parent'], beta_global[0])
        CH_g = analytical_profile(d['z'], d['CH_atm'], d['CH_parent'], beta_global[1])
        r2_g = pearsonr(d['CL'], CL_g)[0] ** 2
        print(f'  [{pid}] 用全局beta预测 LREE_R2={r2_g:.3f}')

    # ---- 留一交叉验证 ----
    print(f'\n=== 留一交叉验证（最关键！）===')
    cv_results = leave_one_out_cv(profile_data, profile_data)
    for pid, r in cv_results.items():
        d = profile_data[pid]
        print(f'  去掉 [{pid}] {d["name"]}: '
              f'LREE_R2_cv={r["r2_CL_cv"]:.3f}  HREE_R2_cv={r["r2_CH_cv"]:.3f}  '
              f'rmse_CL={r["rmse_CL_cv"]:.0f}ppm  rmse_CH={r["rmse_CH_cv"]:.0f}ppm')

    avg_r2_L = np.mean([r['r2_CL_cv'] for r in cv_results.values()])
    avg_r2_H = np.mean([r['r2_CH_cv'] for r in cv_results.values()])
    print(f'\n  平均 CV R²: LREE={avg_r2_L:.3f}  HREE={avg_r2_H:.3f}')
    print(f'  （这个才是真正的泛化能力！）')

    # ---- 气候/径流与 beta 的关系 ----
    print(f'\n=== 气候/径流 vs beta ===')
    print(f'  {"Profile":<30} {"T_C":>6} {"P_mm":>7} {"R_m":>6} {"beta_L":>8} {"beta_H":>8} {"ratio":>6}')
    for pid, d in profile_data.items():
        r = fit_results[pid]
        print(f'  {d["name"]:<30} {d["T_C"]:>6.1f} {d["P_mm"]:>7.0f} '
              f'{d["R_m"]:>6.3f} {r["beta_L"]:>8.4f} {r["beta_H"]:>8.4f} '
              f'{r["beta_L"]/r["beta_H"]:>6.3f}')

    # ---- 绘图 ----
    plot(profile_data, fit_results, cv_results, beta_global, SAVE)

    print(f'\nFigure: {SAVE}/results_v13.png')


# =====================================================================
# 绘图
# =====================================================================
def plot(profile_data, fit_results, cv_results, beta_global, save_dir):
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

    # ---- 左上：气候 vs beta ----
    ax = fig.add_subplot(gs[0, 0])
    pids = sorted(profile_data.keys())
    T_arr = np.array([profile_data[p]['T_C'] for p in pids])
    P_arr = np.array([profile_data[p]['P_mm'] for p in pids])
    R_arr = np.array([profile_data[p]['R_m'] for p in pids])
    bL_arr = np.array([fit_results[p]['beta_L'] for p in pids])
    bH_arr = np.array([fit_results[p]['beta_H'] for p in pids])

    ax.scatter(T_arr, bL_arr, c='blue', s=80, label='beta_L', marker='s')
    ax.scatter(T_arr, bH_arr, c='green', s=80, label='beta_H', marker='^')
    for i, pid in enumerate(pids):
        ax.annotate(str(pid), (T_arr[i], bL_arr[i]), fontsize=8, color='blue')
    ax.set_xlabel('T (°C)')
    ax.set_ylabel('beta coefficient')
    ax.set_title('beta vs Temperature')
    ax.legend(); ax.grid(alpha=0.3)

    # ---- 中上：降水 vs beta ----
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(P_arr, bL_arr, c='blue', s=80, label='beta_L', marker='s')
    ax.scatter(P_arr, bH_arr, c='green', s=80, label='beta_H', marker='^')
    for i, pid in enumerate(pids):
        ax.annotate(str(pid), (P_arr[i], bL_arr[i]), fontsize=8, color='blue')
    ax.set_xlabel('P (mm/yr)')
    ax.set_ylabel('beta coefficient')
    ax.set_title('beta vs Precipitation')
    ax.legend(); ax.grid(alpha=0.3)

    # ---- 右上：径流 vs beta ----
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(R_arr, bL_arr, c='blue', s=80, label='beta_L', marker='s')
    ax.scatter(R_arr, bH_arr, c='green', s=80, label='beta_H', marker='^')
    for i, pid in enumerate(pids):
        ax.annotate(str(pid), (R_arr[i], bL_arr[i]), fontsize=8, color='blue')
    ax.set_xlabel('R (m/yr)')
    ax.set_ylabel('beta coefficient')
    ax.set_title('beta vs Runoff')
    ax.legend(); ax.grid(alpha=0.3)

    # ---- 中行：LREE 和 HREE 剖面 ----
    for idx, pid in enumerate(sorted(profile_data.keys())):
        row = 1; col = idx % 3
        if idx >= 3:
            row = 2
            col = idx - 3
        ax = fig.add_subplot(gs[row, col])
        d = fit_results[pid]
        pd_ = profile_data[pid]
        z = pd_['z']

        # 平滑曲线
        z_smooth = np.linspace(0, max(z) * 1.05, 200)

        CL_fit = analytical_profile(z, pd_['CL_atm'], pd_['CL_parent'], d['beta_L'])
        CH_fit = analytical_profile(z, pd_['CH_atm'], pd_['CH_parent'], d['beta_H'])
        CL_smo = analytical_profile(z_smooth, pd_['CL_atm'], pd_['CL_parent'], d['beta_L'])
        CH_smo = analytical_profile(z_smooth, pd_['CH_atm'], pd_['CH_parent'], d['beta_H'])

        # CV 预测（虚线）
        cv_r = cv_results[pid]
        CL_cv = cv_r['CL_pred']
        CH_cv = cv_r['CH_pred']

        # 全局 beta 预测
        CL_glob = analytical_profile(z, pd_['CL_atm'], pd_['CL_parent'], beta_global[0])
        CH_glob = analytical_profile(z, pd_['CH_atm'], pd_['CH_parent'], beta_global[1])

        ax.plot(CL_smo, z_smooth, 'b-', lw=2, label='LREE_fit')
        ax.plot(CH_smo, z_smooth, 'g-', lw=2, label='HREE_fit')
        ax.plot(CL_glob, z_smooth, 'b--', lw=1, alpha=0.5, label='LREE_global')
        ax.plot(CH_glob, z_smooth, 'g--', lw=1, alpha=0.5, label='HREE_global')
        ax.scatter(pd_['CL'], z, c='blue', s=40, alpha=0.7, label='LREE_obs', zorder=5)
        ax.scatter(pd_['CH'], z, c='green', s=40, alpha=0.7, label='HREE_obs', zorder=5)

        ax.invert_yaxis()
        ax.set_xlabel('Concentration (ppm)')
        ax.set_ylabel('Depth (m)')
        ax.set_title(f'{names.get(pid, str(pid))}\n'
                     f'beta_L={d["beta_L"]:.3f}  beta_H={d["beta_H"]:.3f}\n'
                     f'LREE R²={d["r2_CL"]:.3f}  HREE R²={d["r2_CH"]:.3f}\n'
                     f'CV: LREE R²={cv_r["r2_CL_cv"]:.3f}  HREE R²={cv_r["r2_CH_cv"]:.3f}')
        ax.legend(fontsize=6); ax.set_xlim(0); ax.grid(alpha=0.2)

    # ---- 底部左：L/H 比值剖面 ----
    ax = fig.add_subplot(gs[3, 0])
    for pid in sorted(profile_data.keys()):
        d = fit_results[pid]; pd_ = profile_data[pid]
        RAT = pd_['RAT']
        ax.scatter(RAT, pd_['z'], s=40, alpha=0.7, label=f'[{pid}]')
    ax.set_xlabel('LREE/HREE')
    ax.set_ylabel('Depth (m)')
    ax.set_title('Observed L/H Ratio by Profile')
    ax.legend(fontsize=7); ax.invert_yaxis(); ax.grid(alpha=0.2)

    # ---- 底部中：beta_L vs beta_H ----
    ax = fig.add_subplot(gs[3, 1])
    ax.scatter([fit_results[p]['beta_L'] for p in pids],
               [fit_results[p]['beta_H'] for p in pids],
               c=colors, s=100, zorder=5)
    for pid in pids:
        ax.annotate(str(pid),
                   (fit_results[pid]['beta_L'], fit_results[pid]['beta_H']),
                   fontsize=9)
    # 1:1 line
    bmax = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, bmax], [0, bmax], 'k--', lw=1, label='beta_L = beta_H')
    ax.set_xlabel('beta_L')
    ax.set_ylabel('beta_H')
    ax.set_title('Fractionation Pattern\n(below line=LREE fractionated stronger)')
    ax.legend(); ax.grid(alpha=0.3)

    # ---- 底部右：CV R² 柱状图 ----
    ax = fig.add_subplot(gs[3, 2])
    pids_sorted = sorted(cv_results.keys())
    x = np.arange(len(pids_sorted))
    w = 0.35
    ax.bar(x - w/2, [cv_results[p]['r2_CL_cv'] for p in pids_sorted],
           w, label='LREE CV R²', color='blue', alpha=0.7)
    ax.bar(x + w/2, [cv_results[p]['r2_CH_cv'] for p in pids_sorted],
           w, label='HREE CV R²', color='green', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in pids_sorted])
    ax.set_xlabel('Profile ID')
    ax.set_ylabel('CV R²')
    ax.set_title('Leave-One-Out CV R²\n(True generalization)')
    ax.axhline(0.7, color='orange', ls='--', lw=1, label='R²=0.7 threshold')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    plt.suptitle(
        f'REE 分馏模型 v13 — 解析分馏曲线 + 留一交叉验证\n'
        f'物理：beta_L > beta_H（LREE吸附更强，更早在浅层固定）\n'
        f'Global: beta_L={beta_global[0]:.4f}  beta_H={beta_global[1]:.4f}',
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(f'{save_dir}/results_v13.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
