#!/usr/bin/env python3
"""
Training Dynamics Visualization with Non-Linear Scaling

This script creates publication-quality plots of training dynamics with custom
non-linear axis scaling that emphasizes the critical middle probability ranges.
Supports LaTeX output, custom styling, and multi-format export.

Based on visualization techniques from various open-source projects.
Analysis methods adapted from: https://github.com/simplescaling/s1

Features:
- Non-linear axis scaling for probability visualization
- LaTeX-compatible output formatting
- Multiple export formats (PNG, PDF, SVG)
- Customizable plot styling and colors
- Support for multiple data series and comparison plots
"""
# Flattened training dynamics chart, remove subplot titles, middle chart width increased by 16%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.scale import ScaleBase, register_scale
from matplotlib.transforms import Transform
import os
from matplotlib.gridspec import GridSpec

# Custom non-linear transformation to stretch the 0.05-0.1 range
class CustomScale(ScaleBase):
    name = 'custom'

    def __init__(self, axis, **kwargs):
        super().__init__(axis)
        self.min_val = kwargs.get('min_val', -1.5)  # Minimum value
        self.max_val = kwargs.get('max_val', 1.5)   # Maximum value
        self.focus_min = kwargs.get('focus_min', 0.05)  # Focus area lower limit
        self.focus_max = kwargs.get('focus_max', 0.1)   # Focus area upper limit
        self.focus_proportion = kwargs.get('focus_proportion', 1/4)  # Focus area proportion

    def get_transform(self):
        return CustomTransform(self.min_val, self.max_val, 
                              self.focus_min, self.focus_max, 
                              self.focus_proportion)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(ticker.AutoLocator())
        axis.set_major_formatter(ticker.ScalarFormatter())
        axis.set_minor_locator(ticker.AutoMinorLocator())

class CustomTransform(Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True

    def __init__(self, min_val, max_val, focus_min, focus_max, focus_proportion):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.focus_min = focus_min
        self.focus_max = focus_max
        self.focus_proportion = focus_proportion
        
        # 计算变换参数
        self.range_total = max_val - min_val
        self.range_focus = focus_max - focus_min
        self.range_before = focus_min - min_val
        self.range_after = max_val - focus_max
        
        # 计算各段比例
        self.prop_before = (1 - focus_proportion) * (self.range_before / (self.range_before + self.range_after))
        self.prop_after = (1 - focus_proportion) * (self.range_after / (self.range_before + self.range_after))

    def transform_non_affine(self, a):
        # Convert input values to 0-1 range, then apply piecewise function
        a = np.asarray(a)
        transformed = np.zeros_like(a, dtype=float)
        
        # Below minimum value
        mask_below = a < self.min_val
        transformed[mask_below] = 0
        
        # Before focus area
        mask_before = (a >= self.min_val) & (a < self.focus_min)
        if np.any(mask_before):
            transformed[mask_before] = self.prop_before * (a[mask_before] - self.min_val) / self.range_before
        
        # Inside focus area
        mask_focus = (a >= self.focus_min) & (a <= self.focus_max)
        if np.any(mask_focus):
            transformed[mask_focus] = self.prop_before + self.focus_proportion * (a[mask_focus] - self.focus_min) / self.range_focus
        
        # After focus area
        mask_after = (a > self.focus_max) & (a <= self.max_val)
        if np.any(mask_after):
            transformed[mask_after] = self.prop_before + self.focus_proportion + self.prop_after * (a[mask_after] - self.focus_max) / self.range_after
        
        # Above maximum value
        mask_above = a > self.max_val
        transformed[mask_above] = 1
        
        return transformed

    def inverted(self):
        return CustomTransformInverted(self.min_val, self.max_val, 
                                      self.focus_min, self.focus_max, 
                                      self.focus_proportion)

class CustomTransformInverted(Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True

    def __init__(self, min_val, max_val, focus_min, focus_max, focus_proportion):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.focus_min = focus_min
        self.focus_max = focus_max
        self.focus_proportion = focus_proportion
        
        # 计算变换参数
        self.range_total = max_val - min_val
        self.range_focus = focus_max - focus_min
        self.range_before = focus_min - min_val
        self.range_after = max_val - focus_max
        
        # 计算各段比例
        self.prop_before = (1 - focus_proportion) * (self.range_before / (self.range_before + self.range_after))
        self.prop_after = (1 - focus_proportion) * (self.range_after / (self.range_before + self.range_after))

    def transform_non_affine(self, a):
        # Convert values from 0-1 range back to original range
        a = np.asarray(a)
        transformed = np.zeros_like(a, dtype=float)
        
        # In the front section
        mask_before = a < self.prop_before
        if np.any(mask_before):
            transformed[mask_before] = self.min_val + a[mask_before] * self.range_before / self.prop_before
        
        # 在焦点区域
        mask_focus = (a >= self.prop_before) & (a <= self.prop_before + self.focus_proportion)
        if np.any(mask_focus):
            transformed[mask_focus] = self.focus_min + (a[mask_focus] - self.prop_before) * self.range_focus / self.focus_proportion
        
        # In the rear section
        mask_after = (a > self.prop_before + self.focus_proportion) & (a <= 1)
        if np.any(mask_after):
            transformed[mask_after] = self.focus_max + (a[mask_after] - self.prop_before - self.focus_proportion) * self.range_after / self.prop_after
            
        # Handle out-of-range values
        mask_below = a < 0
        transformed[mask_below] = self.min_val
        
        mask_above = a > 1
        transformed[mask_above] = self.max_val
            
        return transformed

# Register custom scale
register_scale(CustomScale)

# Set Matplotlib's math rendering options and fonts
plt.rcParams['mathtext.fontset'] = 'cm'  # Use Computer Modern font
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['mathtext.it'] = 'serif:italic'
plt.rcParams['mathtext.bf'] = 'serif:bold'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']  # Set Times New Roman
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9

# Create DataFrame from user-provided data
epochs = [1, 2, 3, 4, 5, 6]

# First table data
log_prob_data = {
    'epoch': epochs,
    'Train-F-test-F-y+': [-0.726265809, -0.477814342, -0.322174493, -0.228698501, -0.204687596, -0.201142387],
    'Train-F-test-F-y-': [-1.048971948, -1.125721737, -1.233810733, -1.361633812, -1.579368646, -1.614169914],
    'Train-R-test-R-y+': [-0.764269345, -0.572485764, -0.424912472, -0.347806595, -0.322360264, -0.318942272],
    'Train-R-test-R-y-': [-1.151226815, -1.213307731, -1.333115265, -1.378712537, -1.538816092, -1.561183284],
    'Train-F-test-R-y+': [-1.104215421, -1.164985557, -1.286286335, -1.414295098, -1.649853552, -1.686610591],
    'Train-F-test-R-y-': [-0.867447725, -0.609804335, -0.444267686, -0.342466086, -0.324543993, -0.322790251],
    'Train-R-test-F-y+': [-1.00125926, -1.045710187, -1.161271579, -1.199994041, -1.345268636, -1.36486483],
    'Train-R-test-F-y-': [-0.746699754, -0.565545181, -0.438525051, -0.364714835, -0.353691235, -0.352271307]
}

# Second table data
differ_data = {
    'epoch': epochs,
    'Train-F-test-F-differ': [0.32270614, 0.647907395, 0.91163624, 1.13293531, 1.37468105, 1.413027527],
    'Train-R-test-R-differ': [0.38695747, 0.640821966, 0.908202793, 1.030905942, 1.216455828, 1.242241012],
    'Train-F-test-R-differ': [-0.236767696, -0.555181222, -0.842018649, -1.071829012, -1.325309559, -1.36382034],
    'Train-R-test-F-differ': [-0.254559506, -0.480165006, -0.722746528, -0.835279206, -0.991577401, -1.012593524],
    # Add new data
    'Train-M-test-M-SFT': [0.060925197, 0.064402346, 0.071438487, 0.071737445, 0.078639228, 0.080954835],
    'Train-M-test-M-DPO': [8.11E-02, 8.16E-02, 8.37E-02, 8.67E-02, 9.01E-02, 9.20E-02]
}

# Third table data
sft_dpo_data = {
    'epoch': epochs + list(range(1, 8)),  # SFT 1-6 + DPO 1-7
    'Train-M-test-M-y+': [-0.803001133, -0.611815022, -0.448964911, -0.365061035, -0.339910908, -0.339680546] + 
                       [-3.40E-01, -3.41E-01, -3.41E-01, -3.40E-01, -3.41E-01, -3.42E-01, -3.43E-01],
    'Train-M-test-M-y-': [-0.86392633, -0.676217368, -0.520403398, -0.43679848, -0.418550136, -0.420635381] + 
                       [-4.21E-01, -4.22E-01, -4.25E-01, -4.27E-01, -4.32E-01, -4.34E-01, -4.36E-01]
}

# Create DataFrame
log_prob_df = pd.DataFrame(log_prob_data)
differ_df = pd.DataFrame(differ_data)
sft_dpo_df = pd.DataFrame(sft_dpo_data)

# Define LaTeX format legend labels
latex_labels = {
    'Train-F-test-F-y+': r'$\mathcal{D}_f, \mathcal{T}_f, y^+$',
    'Train-F-test-F-y-': r'$\mathcal{D}_f, \mathcal{T}_f, y^-$',
    'Train-R-test-R-y+': r'$\mathcal{D}_r, \mathcal{T}_r, y^+$',
    'Train-R-test-R-y-': r'$\mathcal{D}_r, \mathcal{T}_r, y^-$',
    'Train-F-test-R-y+': r'$\mathcal{D}_f, \mathcal{T}_r, y^+$',
    'Train-F-test-R-y-': r'$\mathcal{D}_f, \mathcal{T}_r, y^-$',
    'Train-R-test-F-y+': r'$\mathcal{D}_r, \mathcal{T}_f, y^+$',
    'Train-R-test-F-y-': r'$\mathcal{D}_r, \mathcal{T}_f, y^-$',
    'Train-F-test-F-differ': r'$\mathcal{D}_f, \mathcal{T}_f, \Delta$',
    'Train-R-test-R-differ': r'$\mathcal{D}_r, \mathcal{T}_r, \Delta$',
    'Train-F-test-R-differ': r'$\mathcal{D}_f, \mathcal{T}_r, \Delta$',
    'Train-R-test-F-differ': r'$\mathcal{D}_r, \mathcal{T}_f, \Delta$',
    'Train-M-test-M-y+': r'$\mathcal{D}_m, \mathcal{T}_m, y^+$',
    'Train-M-test-M-y-': r'$\mathcal{D}_m, \mathcal{T}_m, y^-$',
    'Train-M-test-M-y+ (SFT)': r'$\mathcal{D}_m, \mathcal{T}_m, y^+$ (SFT)',
    'Train-M-test-M-y- (SFT)': r'$\mathcal{D}_m, \mathcal{T}_m, y^-$ (SFT)',
    'Train-M-test-M-y+ (DPO)': r'$\text{DPO } y^+$',
    'Train-M-test-M-y- (DPO)': r'$\text{DPO } y^-$',
    # Add new labels
    'Train-M-test-M-SFT': r'$\mathcal{D}_m, \mathcal{T}_m, \Delta$ (SFT)',
    'Train-M-test-M-DPO': r'$\mathcal{D}_m, \mathcal{T}_m, \Delta$ (DPO)'
}

# Create figure - use GridSpec to control subplot width
# Reduce figure width by 6%
fig = plt.figure(figsize=(18*0.94, 4.5))  # Original width was 18, now reduced by 6%

# Set grid, increase middle figure width by 16%, but we've already swapped the order of figures 2 and 3
gs = GridSpec(1, 3, width_ratios=[1, 1, 1.16])  # Third figure width ratio set to 1.16, increased by 16%
axes = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 2]), plt.subplot(gs[0, 1])]  # Swapped order of 2 and 3

# Color configuration
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'x']

# Figure 1: Training epoch vs. Average log probability
ax = axes[0]
# Add (a) label
ax.text(-0.5, -0.1, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
for i, col in enumerate(['Train-F-test-F-y+', 'Train-F-test-F-y-', 'Train-R-test-R-y+', 'Train-R-test-R-y-']):
    ax.plot(log_prob_df['epoch'], log_prob_df[col], marker=markers[i], label=latex_labels[col], color=colors[i], linewidth=1.8, markersize=8)
ax.set_xlabel('Training Epoch', fontweight='bold')
ax.set_ylabel('Average Log Probability', fontweight='bold')
# 移除标题
ax.set_xticks(log_prob_df['epoch'])
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='best', frameon=True, fontsize=9)

# Figure 3: SFT+DPO training epoch vs. Average log probability (originally figure 3, now figure 2)
ax = axes[1]
# Add (b) label
ax.text(-0.5, -0.1, '(b)', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

# Find the minimum and maximum values for all data in the second figure
all_values = []
for col in differ_df.columns[1:]:  # Skip epoch column
    all_values.extend(differ_df[col].values)
min_val = min(all_values)
max_val = max(all_values)

# Use custom non-linear axis
ax.set_yscale('custom', min_val=-1.5, max_val=1.5, 
             focus_min=0.05, focus_max=0.1, focus_proportion=1/4)

# Draw all lines separately, highlight lines in the 0.05-0.1 range
# 首先绘制常规线条
other_lines = ['Train-F-test-F-differ', 'Train-R-test-R-differ', 'Train-F-test-R-differ', 'Train-R-test-F-differ']
for i, col in enumerate(other_lines):
    ax.plot(differ_df['epoch'], differ_df[col], marker=markers[i], label=latex_labels[col], 
           color=colors[i], linewidth=1.5, markersize=7)

# 然后绘制重点线条（使用更粗的线宽和更大的标记）
focus_lines = ['Train-M-test-M-SFT', 'Train-M-test-M-DPO']
for i, col in enumerate(focus_lines):
    ax.plot(differ_df['epoch'], differ_df[col], marker=markers[i+4], label=latex_labels[col], 
           color=colors[i+4], linewidth=2.2, markersize=9)

# 添加一个矩形区域标记0.05-0.1区间
ax.axhspan(0.05, 0.1, alpha=0.1, color='gray')

# 设置Y轴标签和网格等
ax.set_xlabel('Training Epoch', fontweight='bold')
ax.set_ylabel('Probability Difference', fontweight='bold')
# 移除标题
ax.set_xticks(differ_df['epoch'])

# 设置固定的Y轴范围
ax.set_ylim(-1.5, 1.5)

# 自定义Y轴刻度，避免重叠
yticks = [-1.5, -1.0, -0.5, 0.0, 0.05, 0.1, 0.5, 1.0, 1.5]  # 基本刻度包含关键区域边界
yticks.sort()
ax.set_yticks(yticks)

# 调整刻度标签，避免重叠
def custom_formatter(x, pos):
    # 格式化Y轴标签
    if abs(x) < 0.0001:  # 接近0的值
        return ''  # 不显示0处的标签
    elif x == 0.05:
        return '0.05'
    elif x == 0.1:
        return '0.1'
    else:
        return f'{x:.1f}'
        
ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
ax.grid(True, linestyle='--', alpha=0.6)

# 图例 - 设置为2列3行并置于左下角
ax.legend(loc='lower left', frameon=True, fontsize=8, ncol=2, columnspacing=1.0, handletextpad=0.5)

# 图2: Training epoch vs. Probability difference - 使用非线性坐标使得0.05-0.1区间占据四分之一 (原来是图2，现在是图3)
ax = axes[2]
# 添加(c)标签
ax.text(-0.5, -0.1, '(c)', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

# 未使用自定义坐标，而是通过其他方式模拟非线性效果

epochs_sft = sft_dpo_df['epoch'][:6]
epochs_dpo = sft_dpo_df['epoch'][6:]
y_plus_sft = sft_dpo_df['Train-M-test-M-y+'][:6]
y_plus_dpo = sft_dpo_df['Train-M-test-M-y+'][6:]
y_minus_sft = sft_dpo_df['Train-M-test-M-y-'][:6]
y_minus_dpo = sft_dpo_df['Train-M-test-M-y-'][6:]

# 创建所有的epoch点
# SFT阶段
ax.plot(epochs_sft, y_plus_sft, marker='o', label=latex_labels['Train-M-test-M-y+ (SFT)'], color=colors[0], linewidth=1.8, markersize=8)
ax.plot(epochs_sft, y_minus_sft, marker='s', label=latex_labels['Train-M-test-M-y- (SFT)'], color=colors[1], linewidth=1.8, markersize=8)

# 创建完整曲线，包含从第6个epoch到第7个epoch的连接
all_epochs = np.arange(1, 14)  # 1刓7全部epoch

# 创建完整的y+和y-值数组
all_y_plus = list(y_plus_sft) + list(y_plus_dpo)
all_y_minus = list(y_minus_sft) + list(y_minus_dpo)

# 绘制SFT阶段曲线（实线）
ax.plot(all_epochs[:6], all_y_plus[:6], marker='o', color=colors[0], 
      linewidth=1.8, markersize=8, label=latex_labels['Train-M-test-M-y+ (SFT)'])
ax.plot(all_epochs[:6], all_y_minus[:6], marker='s', color=colors[1], 
      linewidth=1.8, markersize=8, label=latex_labels['Train-M-test-M-y- (SFT)'])

# 绘制DPO阶段曲线（虚线）
ax.plot(all_epochs[6:], all_y_plus[6:], marker='o', linestyle='--', color=colors[0], 
      linewidth=1.8, markersize=8, label=latex_labels['Train-M-test-M-y+ (DPO)'])
ax.plot(all_epochs[6:], all_y_minus[6:], marker='s', linestyle='--', color=colors[1], 
      linewidth=1.8, markersize=8, label=latex_labels['Train-M-test-M-y- (DPO)'])

# 添加阴影区域突出-0.4到-0.6区间
ax.axhspan(-0.6, -0.4, alpha=0.1, color='gray')

# 添加垂直线表示SFT到DPO的过渡
ax.axvline(x=6.5, color='gray', linestyle=':', alpha=0.7)
ax.text(6.5, -0.65, r'SFT$\rightarrow$DPO', rotation=90, verticalalignment='center')

ax.set_xlabel('Training Epoch', fontweight='bold')
ax.set_ylabel('Average Log Probability', fontweight='bold')
# 移除标题
ax.set_xticks(list(range(1, 14)))

# 自定义Y轴刻度，避免重叠
yticks = [-1.5, -1.2, -0.9, -0.6, -0.5, -0.4, -0.3, -0.0]  # 基本刻度包含关键区域边界
yticks.sort()
ax.set_yticks(yticks)

# 调整刻度标签，避免重叠
def custom_formatter_right(x, pos):
    # 格式化Y轴标签
    if x == -0.6:
        return '-0.6'
    elif x == -0.4:
        return '-0.4'
    elif abs(x) < 0.001:  # 非常接近0
        return '0'
    else:
        return f'{x:.1f}'  # 保留一位小数
        
ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter_right))

# 设置Y轴范围为-1.5到0
ax.set_ylim(-1.5, 0)

# 使用自定义位置是服立于这个区间的标签更显的
# 创建放大区域的标记
ax.annotate('', xy=(13.5, -0.4), xytext=(13.5, -0.6),
           arrowprops=dict(arrowstyle='<->', color='#444444', lw=1.5))
ax.text(13.7, -0.5, 'Key\nArea', ha='left', va='center', color='#444444')

# 空出图表标签区域的空间
fig.subplots_adjust(right=0.98)

ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='best', frameon=True, fontsize=9)

# 调整布局
plt.tight_layout()

# 保存图表
output_dir = "/hpc2hdd/home/tzhu619/Judy-demi/s1/LDF/result"
output_file = os.path.join(output_dir, "training_dynamics_modified.png")
plt.savefig(output_file, dpi=400, bbox_inches='tight')

# 生成LaTeX代码
latex_code = r"""
\begin{figure*}[htbp]
    \centering
    \includegraphics[width=\textwidth]{training_dynamics_modified.png}
    \caption{Training dynamics analysis. (a) Average log probability of different reasoning paths across training epochs; 
    (b) SFT followed by DPO training dynamics showing model's preference development; 
    (c) Probability differences with non-linear scale allowing 0.05-0.1 range to occupy 1/4 of the plot height.}
    \label{fig:training-dynamics-modified}
\end{figure*}
"""

# 保存LaTeX代码
latex_file = os.path.join(output_dir, "latex_figure_code_modified.tex")
with open(latex_file, "w") as f:
    f.write(latex_code)

print(f"Figure saved to {output_file}")
print(f"LaTeX code saved to {latex_file}")
print("Note: Figure width reduced by 6%, subplots 2 and 3 swapped, and subplot labels (a,b,c) added.")
print("      Third plot uses a non-linear Y-axis scaling to make the 0.05-0.1 range occupy 1/4 of the plot height.")
