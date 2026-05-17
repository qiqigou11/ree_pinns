"""
稀土数据归一化可视化
===================
1. 基于数据分布的深度归一化
2. HREE vs 归一化深度叠加图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

DATA = '/Users/suheng/Desktop/claudecode/pinns/ree_pinns/data/REE_samples_20260424_with_climate.csv'
LIT = '/Users/suheng/Desktop/claudecode/pinns/ree_pinns/data/REE_literature_20260424.csv'
SAVE_DIR = '/Users/suheng/Desktop/claudecode/pinns/ree_pinns/v15/data_reanalyze'

LREE_COLS = ['La_ppm', 'Ce_ppm', 'Pr_ppm', 'Nd_ppm', 'Sm_ppm']
HREE_COLS = ['Tb_ppm', 'Dy_ppm', 'Ho_ppm', 'Er_ppm', 'Tm_ppm', 'Yb_ppm', 'Lu_ppm']

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


def normalize_depth(df, method='minmax'):
    """
    深度归一化

    方法1 - minmax: (z - z_min) / (z_max - z_min)
    方法2 - zscore: (z - z_mean) / z_std
    方法3 - percentile: 使用百分位排名
    """
    z = df['Depth_m'].values

    if method == 'minmax':
        z_norm = (z - z.min()) / (z.max() - z.min())
    elif method == 'zscore':
        z_norm = (z - z.mean()) / z.std()
    elif method == 'percentile':
        z_norm = stats.rankdata(z) / len(z)

    return z_norm


def plot_normalized_depth_distribution(df):
    """深度归一化后的分布对比"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 原始深度分布
    ax = axes[0, 0]
    for idx, pid in enumerate(sorted(df['literature_id'].unique())):
        grp = df[df['literature_id'] == pid]
        ax.hist(grp['Depth_m'], bins=15, alpha=0.5, label=f'P{pid}', color=COLORS[idx])
    ax.set_xlabel('Depth (m)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Original Depth Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Min-Max归一化深度分布
    ax = axes[0, 1]
    df['z_norm_minmax'] = normalize_depth(df, 'minmax')
    for idx, pid in enumerate(sorted(df['literature_id'].unique())):
        grp = df[df['literature_id'] == pid]
        ax.hist(grp['z_norm_minmax'], bins=15, alpha=0.5, label=f'P{pid}', color=COLORS[idx])
    ax.set_xlabel('Normalized Depth (0-1)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Depth: Min-Max Normalized', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Z-score归一化深度分布
    ax = axes[1, 0]
    df['z_norm_zscore'] = normalize_depth(df, 'zscore')
    for idx, pid in enumerate(sorted(df['literature_id'].unique())):
        grp = df[df['literature_id'] == pid]
        ax.hist(grp['z_norm_zscore'], bins=15, alpha=0.5, label=f'P{pid}', color=COLORS[idx])
    ax.set_xlabel('Normalized Depth (z-score)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Depth: Z-Score Normalized', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. HREE vs 归一化深度 (minmax)
    ax = axes[1, 1]
    for idx, pid in enumerate(sorted(df['literature_id'].unique())):
        grp = df[df['literature_id'] == pid].sort_values('Depth_m')
        ax.scatter(grp['HREE'], grp['z_norm_minmax'],
                   c=[COLORS[idx]], label=NAMES.get(pid, f'P{pid}').replace('\n', ' '),
                   s=60, alpha=0.7, edgecolors='white', linewidth=0.5)
    ax.set_xlabel('HREE Concentration (ppm)', fontsize=11)
    ax.set_ylabel('Normalized Depth (0-1)', fontsize=11)
    ax.set_title('HREE vs Normalized Depth (All Profiles)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f'{SAVE_DIR}/normalized_depth_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {SAVE_DIR}/normalized_depth_analysis.png')


def plot_hree_overlay_normalized(df):
    """HREE vs 归一化深度 - 所有剖面叠加"""
    # 使用min-max归一化（0-1范围）
    df['z_norm'] = (df['Depth_m'] - df['Depth_m'].min()) / (df['Depth_m'].max() - df['Depth_m'].min())

    fig, ax = plt.subplots(figsize=(10, 8))

    for idx, pid in enumerate(sorted(df['literature_id'].unique())):
        grp = df[df['literature_id'] == pid].sort_values('Depth_m')
        name = NAMES.get(pid, f'Profile {pid}').replace('\n', ' ')

        ax.plot(grp['HREE'].values, grp['z_norm'].values,
                'o-', color=COLORS[idx], label=name, linewidth=2, markersize=6, alpha=0.8)

    ax.set_xlabel('HREE Concentration (ppm)', fontsize=12)
    ax.set_ylabel('Normalized Depth (0 = shallow, 1 = deep)', fontsize=12)
    ax.set_title('HREE vs Normalized Depth - All Profiles Overlay', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f'{SAVE_DIR}/hree_normalized_depth_overlay.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {SAVE_DIR}/hree_normalized_depth_overlay.png')


def plot_lree_hree_normalized_overlay(df):
    """LREE和HREE分别 vs 归一化深度叠加"""
    df['z_norm'] = (df['Depth_m'] - df['Depth_m'].min()) / (df['Depth_m'].max() - df['Depth_m'].min())

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    for idx, pid in enumerate(sorted(df['literature_id'].unique())):
        grp = df[df['literature_id'] == pid].sort_values('Depth_m')
        name = NAMES.get(pid, f'Profile {pid}').replace('\n', ' ')

        axes[0].plot(grp['LREE'].values, grp['z_norm'].values,
                     'o-', color=COLORS[idx], label=name, linewidth=2, markersize=6, alpha=0.8)
        axes[1].plot(grp['HREE'].values, grp['z_norm'].values,
                     's-', color=COLORS[idx], label=name, linewidth=2, markersize=6, alpha=0.8)

    axes[0].set_xlabel('LREE Concentration (ppm)', fontsize=12)
    axes[0].set_ylabel('Normalized Depth (0 = shallow, 1 = deep)', fontsize=12)
    axes[0].set_title('LREE vs Normalized Depth', fontsize=14, fontweight='bold')
    axes[0].legend(loc='lower right', fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('HREE Concentration (ppm)', fontsize=12)
    axes[1].set_ylabel('Normalized Depth (0 = shallow, 1 = deep)', fontsize=12)
    axes[1].set_title('HREE vs Normalized Depth', fontsize=14, fontweight='bold')
    axes[1].legend(loc='lower right', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('LREE and HREE vs Normalized Depth', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(f'{SAVE_DIR}/lree_hree_normalized_overlay.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {SAVE_DIR}/lree_hree_normalized_overlay.png')


def plot_per_profile_normalized(df):
    """每个剖面的归一化深度 vs HREE (单独子图)"""
    profile_ids = sorted(df['literature_id'].unique())
    n = len(profile_ids)

    # Min-max归一化（每个剖面独立归一化）
    df['z_norm_perprofile'] = df.groupby('literature_id')['Depth_m'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, pid in enumerate(profile_ids):
        ax = axes[idx]
        grp = df[df['literature_id'] == pid].sort_values('Depth_m')
        name = NAMES.get(pid, f'Profile {pid}')

        # 归一化深度
        z_norm = grp['z_norm_perprofile'].values

        # 绘制HREE和LREE
        ax.scatter(grp['LREE'].values, z_norm,
                   c='#E74C3C', s=60, alpha=0.7, label='LREE', edgecolors='white')
        ax.scatter(grp['HREE'].values, z_norm,
                   c='#3498DB', s=60, alpha=0.7, label='HREE', edgecolors='white')

        # 连接线
        ax.plot(grp['LREE'].values, z_norm, '-', c='#E74C3C', alpha=0.3, linewidth=1)
        ax.plot(grp['HREE'].values, z_norm, '-', c='#3498DB', alpha=0.3, linewidth=1)

        ax.set_xlabel('Concentration (ppm)', fontsize=10)
        ax.set_ylabel('Normalized Depth (0-1)', fontsize=10)
        ax.set_title(f'{name}\n(n={len(grp)})', fontsize=10, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f'{SAVE_DIR}/per_profile_normalized.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {SAVE_DIR}/per_profile_normalized.png')


def plot_combined_normalized(df):
    """综合归一化分析图"""
    # 全局归一化
    z_min, z_max = df['Depth_m'].min(), df['Depth_m'].max()
    df['z_norm_global'] = (df['Depth_m'] - z_min) / (z_max - z_min)

    # 每个剖面独立归一化
    df['z_norm_perprofile'] = df.groupby('literature_id')['Depth_m'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0
    )

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1. HREE vs 全局归一化深度
    ax1 = fig.add_subplot(gs[0, :2])
    for idx, pid in enumerate(sorted(df['literature_id'].unique())):
        grp = df[df['literature_id'] == pid].sort_values('Depth_m')
        name = NAMES.get(pid, f'P{pid}').replace('\n', ' ')
        ax1.plot(grp['HREE'].values, grp['z_norm_global'].values,
                 'o-', color=COLORS[idx], label=name, linewidth=2, markersize=5, alpha=0.8)
    ax1.set_xlabel('HREE (ppm)', fontsize=11)
    ax1.set_ylabel('Normalized Depth (Global 0-1)', fontsize=11)
    ax1.set_title('HREE vs Global Normalized Depth', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. 深度分布统计
    ax2 = fig.add_subplot(gs[0, 2])
    profile_ids_sorted = sorted(df['literature_id'].unique())
    for idx, pid in enumerate(profile_ids_sorted):
        grp = df[df['literature_id'] == pid]
        ax2.bar(idx, grp['Depth_m'].max(), color=COLORS[idx], alpha=0.7)
        ax2.bar(idx, grp['Depth_m'].min(), color='white', edgecolor=COLORS[idx], linewidth=2)
    ax2.set_xlabel('Profile', fontsize=11)
    ax2.set_ylabel('Depth (m)', fontsize=11)
    ax2.set_title('Max/Min Depth per Profile', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(profile_ids_sorted)))
    ax2.set_xticklabels([f'P{i}' for i in profile_ids_sorted])
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. LREE vs 每个剖面归一化深度
    ax3 = fig.add_subplot(gs[1, 0])
    for idx, pid in enumerate(sorted(df['literature_id'].unique())):
        grp = df[df['literature_id'] == pid].sort_values('Depth_m')
        name = NAMES.get(pid, f'P{pid}').replace('\n', ' ')
        ax3.plot(grp['LREE'].values, grp['z_norm_perprofile'].values,
                 'o-', color=COLORS[idx], label=name, linewidth=2, markersize=5, alpha=0.8)
    ax3.set_xlabel('LREE (ppm)', fontsize=11)
    ax3.set_ylabel('Normalized Depth (Per Profile 0-1)', fontsize=11)
    ax3.set_title('LREE vs Per-Profile Normalized Depth', fontsize=12, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=7)
    ax3.grid(True, alpha=0.3)

    # 4. HREE vs 每个剖面归一化深度
    ax4 = fig.add_subplot(gs[1, 1])
    for idx, pid in enumerate(sorted(df['literature_id'].unique())):
        grp = df[df['literature_id'] == pid].sort_values('Depth_m')
        name = NAMES.get(pid, f'P{pid}').replace('\n', ' ')
        ax4.plot(grp['HREE'].values, grp['z_norm_perprofile'].values,
                 's-', color=COLORS[idx], label=name, linewidth=2, markersize=5, alpha=0.8)
    ax4.set_xlabel('HREE (ppm)', fontsize=11)
    ax4.set_ylabel('Normalized Depth (Per Profile 0-1)', fontsize=11)
    ax4.set_title('HREE vs Per-Profile Normalized Depth', fontsize=12, fontweight='bold')
    ax4.legend(loc='lower right', fontsize=7)
    ax4.grid(True, alpha=0.3)

    # 5. TotalREE vs 每个剖面归一化深度
    ax5 = fig.add_subplot(gs[1, 2])
    for idx, pid in enumerate(sorted(df['literature_id'].unique())):
        grp = df[df['literature_id'] == pid].sort_values('Depth_m')
        name = NAMES.get(pid, f'P{pid}').replace('\n', ' ')
        ax5.plot(grp['TotalREE'].values, grp['z_norm_perprofile'].values,
                 '^-', color=COLORS[idx], label=name, linewidth=2, markersize=5, alpha=0.8)
    ax5.set_xlabel('TotalREE (ppm)', fontsize=11)
    ax5.set_ylabel('Normalized Depth (Per Profile 0-1)', fontsize=11)
    ax5.set_title('TotalREE vs Per-Profile Normalized Depth', fontsize=12, fontweight='bold')
    ax5.legend(loc='lower right', fontsize=7)
    ax5.grid(True, alpha=0.3)

    # 6. L/H比值 vs 每个剖面归一化深度
    ax6 = fig.add_subplot(gs[2, :2])
    for idx, pid in enumerate(sorted(df['literature_id'].unique())):
        grp = df[df['literature_id'] == pid].sort_values('Depth_m')
        name = NAMES.get(pid, f'P{pid}').replace('\n', ' ')
        ax6.plot(grp['LH_ratio'].values, grp['z_norm_perprofile'].values,
                 'o-', color=COLORS[idx], label=name, linewidth=2, markersize=5, alpha=0.8)
    ax6.set_xlabel('L/H Ratio', fontsize=11)
    ax6.set_ylabel('Normalized Depth (Per Profile 0-1)', fontsize=11)
    ax6.set_title('L/H Ratio vs Per-Profile Normalized Depth', fontsize=12, fontweight='bold')
    ax6.legend(loc='lower right', fontsize=8)
    ax6.grid(True, alpha=0.3)

    # 7. HREE/LREE比值 vs 深度
    ax7 = fig.add_subplot(gs[2, 2])
    for idx, pid in enumerate(sorted(df['literature_id'].unique())):
        grp = df[df['literature_id'] == pid].sort_values('Depth_m')
        name = NAMES.get(pid, f'P{pid}').replace('\n', ' ')
        hl_ratio = grp['HREE'].values / (grp['LREE'].values + 1e-6)
        ax7.plot(hl_ratio, grp['z_norm_perprofile'].values,
                 'o-', color=COLORS[idx], label=name, linewidth=2, markersize=5, alpha=0.8)
    ax7.set_xlabel('H/L Ratio', fontsize=11)
    ax7.set_ylabel('Normalized Depth (Per Profile 0-1)', fontsize=11)
    ax7.set_title('H/L Ratio vs Per-Profile Normalized Depth', fontsize=12, fontweight='bold')
    ax7.legend(loc='lower right', fontsize=7)
    ax7.grid(True, alpha=0.3)

    plt.suptitle('REE Distribution Analysis with Normalized Depth', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(f'{SAVE_DIR}/combined_normalized_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {SAVE_DIR}/combined_normalized_analysis.png')


def main():
    print("Loading data...")
    df = load_and_prepare_data()
    print(f"Loaded {len(df)} samples from {df['literature_id'].nunique()} profiles")

    print("\n--- 1. Depth Distribution Analysis ---")
    plot_normalized_depth_distribution(df)

    print("\n--- 2. HREE Overlay with Normalized Depth ---")
    plot_hree_overlay_normalized(df)

    print("\n--- 3. LREE/HREE vs Normalized Depth ---")
    plot_lree_hree_normalized_overlay(df)

    print("\n--- 4. Per-Profile Normalized Depth ---")
    plot_per_profile_normalized(df)

    print("\n--- 5. Combined Analysis ---")
    plot_combined_normalized(df)

    print("\nDone!")


if __name__ == '__main__':
    main()
