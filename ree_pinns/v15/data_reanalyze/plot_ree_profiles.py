"""
稀土垂直分布可视化
===================
展示每个剖面的：
1. TotalREE vs Depth
2. LREE, HREE vs Depth
3. L/H Ratio vs Depth
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

DATA = '/Users/suheng/Desktop/claudecode/pinns/ree_pinns/data/REE_samples_20260424_with_climate.csv'
LIT = '/Users/suheng/Desktop/claudecode/pinns/ree_pinns/data/REE_literature_20260424.csv'
SAVE_DIR = '/Users/suheng/Desktop/claudecode/pinns/ree_pinns/v15/data_reanalyze'

LREE_COLS = ['La_ppm', 'Ce_ppm', 'Pr_ppm', 'Nd_ppm', 'Sm_ppm']
HREE_COLS = ['Tb_ppm', 'Dy_ppm', 'Ho_ppm', 'Er_ppm', 'Tm_ppm', 'Yb_ppm', 'Lu_ppm']

# Profile name mapping
NAMES = {
    1: 'Nagasawa\n(weathered granite)',
    2: 'Li2019\n(A-type granite)',
    3: 'Li&Zhou\n(A-type granite)',
    4: 'Fu2019\n(rhyolite)',
    5: 'Yaraghi\n(two-mica granite)',
    6: 'Luo\n(tonalite/diorite)',
}

COLORS = plt.cm.Set2(np.linspace(0, 1, 8))


def load_and_prepare_data():
    """加载并预处理数据"""
    df = pd.read_csv(DATA)
    dfl = pd.read_csv(LIT)

    df = df.merge(dfl[['id', 'parent_rock']], left_on='literature_id', right_on='id',
                  how='left', suffixes=('', '_l'))
    df.parent_rock = df.parent_rock.fillna(df.Bedrock)
    df = df.dropna(subset=['Depth_m'] + LREE_COLS + HREE_COLS).reset_index(drop=True)

    # Calculate concentrations
    df['LREE'] = df[LREE_COLS].sum(axis=1)
    df['HREE'] = df[HREE_COLS].sum(axis=1)
    df['TotalREE'] = df['LREE'] + df['HREE']
    df['LH_ratio'] = df['LREE'] / (df['HREE'] + 1e-6)

    return df


def plot_single_profile(ax, grp, pid, name, color):
    """绘制单个剖面的三项指标"""
    depth = grp['Depth_m'].values
    lree = grp['LREE'].values
    hree = grp['HREE'].values
    total = grp['TotalREE'].values
    lh = grp['LH_ratio'].values

    # Plot TotalREE
    ax[0].scatter(total, depth, c=color, s=60, alpha=0.8, edgecolors='white', linewidth=0.5)
    ax[0].plot(total, depth, '-', c=color, alpha=0.4, linewidth=1)
    ax[0].set_xlabel('Total REE (ppm)', fontsize=10)
    ax[0].set_ylabel('Depth (m)', fontsize=10)
    ax[0].set_title(f'{name}\nTotal REE', fontsize=10, fontweight='bold')
    ax[0].invert_yaxis()
    ax[0].grid(True, alpha=0.3)

    # Plot LREE and HREE separately
    ax[1].scatter(lree, depth, c='#E74C3C', s=50, alpha=0.7, label='LREE', edgecolors='white', linewidth=0.5)
    ax[1].scatter(hree, depth, c='#3498DB', s=50, alpha=0.7, label='HREE', edgecolors='white', linewidth=0.5)
    ax[1].plot(lree, depth, '-', c='#E74C3C', alpha=0.3, linewidth=1)
    ax[1].plot(hree, depth, '-', c='#3498DB', alpha=0.3, linewidth=1)
    ax[1].set_xlabel('Concentration (ppm)', fontsize=10)
    ax[1].set_ylabel('Depth (m)', fontsize=10)
    ax[1].set_title('LREE vs HREE', fontsize=10)
    ax[1].invert_yaxis()
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(loc='lower right', fontsize=8)

    # Plot L/H ratio
    ax[2].scatter(lh, depth, c='#27AE60', s=60, alpha=0.8, edgecolors='white', linewidth=0.5)
    ax[2].plot(lh, depth, '-', c='#27AE60', alpha=0.4, linewidth=1)
    ax[2].set_xlabel('L/H Ratio', fontsize=10)
    ax[2].set_ylabel('Depth (m)', fontsize=10)
    ax[2].set_title('L/H Ratio', fontsize=10)
    ax[2].invert_yaxis()
    ax[2].grid(True, alpha=0.3)


def plot_all_profiles(df):
    """绘制所有剖面的总览图"""
    profile_ids = sorted(df['literature_id'].unique())

    # Figure 1: Each profile in detail (3 columns x N profiles)
    n_profiles = len(profile_ids)
    fig, axes = plt.subplots(n_profiles, 3, figsize=(14, 4*n_profiles))

    for idx, pid in enumerate(profile_ids):
        grp = df[df['literature_id'] == pid].sort_values('Depth_m')
        name = NAMES.get(pid, f'Profile {pid}')

        # Get climate info for title
        T_C = grp['T_annual_mean_K'].iloc[0] - 273.15
        P_mm = grp['P_annual_mean_mm_yr'].iloc[0]

        row_axes = axes[idx] if n_profiles > 1 else axes
        plot_single_profile(row_axes, grp, pid, name, COLORS[idx])

        # Add climate info on the left side
        row_axes[0].annotate(f'T={T_C:.0f}°C  P={P_mm:.0f}mm',
                            xy=(0.02, 0.02), xycoords='axes fraction',
                            fontsize=8, color='gray')

    plt.tight_layout()
    fig.savefig(f'{SAVE_DIR}/all_profiles_detail.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {SAVE_DIR}/all_profiles_detail.png')


def plot_total_ree_overlay(df):
    """所有剖面TotalREE叠加对比"""
    profile_ids = sorted(df['literature_id'].unique())

    fig, ax = plt.subplots(figsize=(10, 8))

    for idx, pid in enumerate(profile_ids):
        grp = df[df['literature_id'] == pid].sort_values('Depth_m')
        name = NAMES.get(pid, f'Profile {pid}').replace('\n', ' ')

        ax.plot(grp['TotalREE'].values, grp['Depth_m'].values,
                'o-', color=COLORS[idx], label=name, linewidth=2, markersize=6, alpha=0.8)

    ax.set_xlabel('Total REE (ppm)', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_title('Total REE vs Depth - All Profiles', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f'{SAVE_DIR}/total_ree_overlay.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {SAVE_DIR}/total_ree_overlay.png')


def plot_lh_ratio_overlay(df):
    """所有剖面L/H比值叠加对比"""
    profile_ids = sorted(df['literature_id'].unique())

    fig, ax = plt.subplots(figsize=(10, 8))

    for idx, pid in enumerate(profile_ids):
        grp = df[df['literature_id'] == pid].sort_values('Depth_m')
        name = NAMES.get(pid, f'Profile {pid}').replace('\n', ' ')

        ax.plot(grp['LH_ratio'].values, grp['Depth_m'].values,
                'o-', color=COLORS[idx], label=name, linewidth=2, markersize=6, alpha=0.8)

    ax.set_xlabel('L/H Ratio', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_title('L/H Ratio vs Depth - All Profiles', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f'{SAVE_DIR}/lh_ratio_overlay.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {SAVE_DIR}/lh_ratio_overlay.png')


def plot_lree_hree_separate_overlay(df):
    """所有剖面LREE和HREE分别叠加对比"""
    profile_ids = sorted(df['literature_id'].unique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    for idx, pid in enumerate(profile_ids):
        grp = df[df['literature_id'] == pid].sort_values('Depth_m')
        name = NAMES.get(pid, f'Profile {pid}').replace('\n', ' ')

        axes[0].plot(grp['LREE'].values, grp['Depth_m'].values,
                     'o-', color=COLORS[idx], label=name, linewidth=2, markersize=6, alpha=0.8)
        axes[1].plot(grp['HREE'].values, grp['Depth_m'].values,
                     's-', color=COLORS[idx], label=name, linewidth=2, markersize=6, alpha=0.8)

    axes[0].set_xlabel('LREE Concentration (ppm)', fontsize=12)
    axes[0].set_ylabel('Depth (m)', fontsize=12)
    axes[0].set_title('LREE vs Depth', fontsize=14, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].legend(loc='lower right', fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('HREE Concentration (ppm)', fontsize=12)
    axes[1].set_ylabel('Depth (m)', fontsize=12)
    axes[1].set_title('HREE vs Depth', fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].legend(loc='lower right', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('LREE and HREE Vertical Distribution', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(f'{SAVE_DIR}/lree_hree_separate_overlay.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {SAVE_DIR}/lree_hree_separate_overlay.png')


def plot_summary_statistics(df):
    """剖面统计摘要"""
    profile_ids = sorted(df['literature_id'].unique())

    stats = []
    for pid in profile_ids:
        grp = df[df['literature_id'] == pid]
        name = NAMES.get(pid, f'P{pid}').replace('\n', ' ')
        T_C = grp['T_annual_mean_K'].iloc[0] - 273.15
        P_mm = grp['P_annual_mean_mm_yr'].iloc[0]

        # Find max TotalREE and its depth
        max_idx = grp['TotalREE'].idxmax()
        max_total = grp.loc[max_idx, 'TotalREE']
        max_depth = grp.loc[max_idx, 'Depth_m']

        # Depth trend
        shallow = grp[grp['Depth_m'] < 3]['TotalREE'].mean()
        deep = grp[grp['Depth_m'] >= 3]['TotalREE'].mean()

        stats.append({
            'Profile': name,
            'T (°C)': T_C,
            'P (mm)': P_mm,
            'n samples': len(grp),
            'Depth range (m)': f"{grp['Depth_m'].min():.1f}-{grp['Depth_m'].max():.1f}",
            'Max TotalREE (ppm)': f"{max_total:.0f}",
            'Max depth (m)': f"{max_depth:.1f}",
            'Mean L/H': f"{grp['LH_ratio'].mean():.2f}",
            'Shallow (<3m) Mean': f"{shallow:.0f}" if not np.isnan(shallow) else 'N/A',
            'Deep (>=3m) Mean': f"{deep:.0f}" if not np.isnan(deep) else 'N/A',
        })

    stats_df = pd.DataFrame(stats)
    print("\n" + "="*80)
    print("Profile Summary Statistics")
    print("="*80)
    print(stats_df.to_string(index=False))

    # Save to CSV
    stats_df.to_csv(f'{SAVE_DIR}/profile_statistics.csv', index=False)
    print(f"\nSaved: {SAVE_DIR}/profile_statistics.csv")


def main():
    print("Loading data...")
    df = load_and_prepare_data()
    print(f"Loaded {len(df)} samples from {df['literature_id'].nunique()} profiles")

    print("\nGenerating plots...")
    plot_all_profiles(df)
    plot_total_ree_overlay(df)
    plot_lh_ratio_overlay(df)
    plot_lree_hree_separate_overlay(df)

    print("\nGenerating statistics...")
    plot_summary_statistics(df)

    print("\nDone!")


if __name__ == '__main__':
    main()