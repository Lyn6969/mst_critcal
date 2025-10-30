% plot_multi_sensitivity_results - 多初发敏感性可视化脚本
%
% 功能：读取 run_delta_c_m_vs_1_scan_parallel 生成的数据，根据同一批
%       级联规模结果绘制：
%   1. 多种初发个体数量下的平均级联规模
%   2. 1 vs m 的敏感性曲线
%   3. m vs (m+1) 的边际敏感性曲线（通过相邻 c_m 差分计算）
% 输出三张单图到项目根目录下的 pic 文件夹。

clear; clc; close all;

%% 配置区域 ---------------------------------------------------------------
% 图片尺寸与样式（参考 plot_delta_c_scan_results）
FIG_WIDTH = 400;
FIG_HEIGHT = 200;

LINE_WIDTH = 2.2;
AXIS_LINE_WIDTH = 1.5;

FONT_NAME = 'Arial';
LABEL_FONT_SIZE = 12;
LABEL_FONT_WEIGHT = 'Bold';
TICK_FONT_SIZE = 13;
TICK_FONT_WEIGHT = 'Bold';
LEGEND_FONT_SIZE = 12;

GRID_ON = 'off';
GRID_ALPHA = 0;
GRID_LINE_STYLE = '--';
TICK_DIR = 'in';

INFO_LABEL = 'Info. agents(m)';
COLORBAR_TICK_FONT_SIZE = 19;
COLORBAR_TICK_FONT_WEIGHT = 'Bold';
COLORBAR_LABEL_FONT_SIZE = 23;
COLORBAR_LABEL_FONT_WEIGHT = 'Bold';

% 数据文件（请根据实际结果路径修改）
mat_file_m_vs_1 = fullfile('mst_critcal', 'data', 'experiments', 'delta_c_m_vs_1_scan', ...
    '20251029_233333', 'data.mat');

% 输出文件名
figure_names = { ...
    'cascade_multi', ...
    'delta_c_1_vs_m', ...
    'delta_c_m_vs_mplus1', ...
    'colorbar_reference' ...
};

%% 载入数据 ---------------------------------------------------------------
% 判定项目根目录
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(fileparts(script_dir)));

% 图像输出目录
pic_dir = fullfile(project_root, 'mst_critcal', 'pic');
if ~exist(pic_dir, 'dir')
    mkdir(pic_dir);
end

% 数据路径转为绝对路径
mat_path_m_vs_1 = fullfile(project_root, mat_file_m_vs_1);

if ~isfile(mat_path_m_vs_1)
    error('未找到 1 vs m 数据文件，请更新路径：%s', mat_path_m_vs_1);
end
data_m_vs_1 = load(mat_path_m_vs_1, 'results');
res_m_vs_1 = data_m_vs_1.results;

% 基础数据
thresholds = res_m_vs_1.cj_thresholds(:);
c_mean_raw = res_m_vs_1.c_mean;
delta_c_1m_raw = res_m_vs_1.delta_c;

% 负值裁剪为 0，保证纵轴从 0 起
c_mean = max(c_mean_raw, 0);
delta_c_1m = max(delta_c_1m_raw, 0);

% 使用同一组数据计算 Δc_{m→m+1}
if size(c_mean, 2) < 2
    error('级联规模数据不足以计算 m vs (m+1) 敏感性。');
end
delta_c_mm1_raw = c_mean(:, 2:end) - c_mean(:, 1:end-1);
delta_c_mm1 = max(delta_c_mm1_raw, 0);

total_pulses = numel(res_m_vs_1.pulse_counts);
if total_pulses < 1
    error('结果中缺少 pulse_counts 信息，无法绘制图像。');
end

%% 颜色与映射 -------------------------------------------------------------
% 蓝色色系，从深到浅
function_colors = create_blue_gradient(max(10, total_pulses));

%% 图1：平均级联规模 ------------------------------------------------------
fig1 = figure('Position', [100, 100, FIG_WIDTH, FIG_HEIGHT], 'Color', 'white');
ax1 = axes('Parent', fig1);
hold(ax1, 'on');

num_curves = size(c_mean, 2);
for idx = 1:num_curves
    color_idx = min(max(idx, 1), size(function_colors, 1));
    plot(ax1, thresholds, c_mean(:, idx), 'LineWidth', LINE_WIDTH, ...
        'Color', function_colors(color_idx, :), ...
        'DisplayName', sprintf('c_{%d}', idx));
end

configure_axes(ax1, 'M_T', 'Avg. Cascade Size', GRID_ON, GRID_ALPHA, GRID_LINE_STYLE, TICK_DIR);

% 设置纵轴范围
set_axis_limits(ax1, thresholds, c_mean);

fig1_path = fullfile(pic_dir, sprintf('%s.pdf', figure_names{1}));
save_figure(fig1, fig1_path);
fprintf('图1已保存至: %s (色标：%s 暂未显示)\n', fig1_path, INFO_LABEL);

%% 图2：Δc (1 vs m) -------------------------------------------------------
fig2 = figure('Position', [120, 120, FIG_WIDTH, FIG_HEIGHT], 'Color', 'white');
ax2 = axes('Parent', fig2);
hold(ax2, 'on');

m_values = res_m_vs_1.pulse_counts(2:end);
for idx = 1:size(delta_c_1m, 2)
    color_idx = min(max(m_values(idx), 1), size(function_colors, 1));
    plot(ax2, thresholds, delta_c_1m(:, idx), 'LineWidth', LINE_WIDTH, ...
        'Color', function_colors(color_idx, :), ...
        'DisplayName', sprintf('\\Delta c_{%d-1}', m_values(idx)));
end

configure_axes(ax2, 'M_T', '1 vs m Sensitivity', GRID_ON, GRID_ALPHA, GRID_LINE_STYLE, TICK_DIR);
set_axis_limits(ax2, thresholds, delta_c_1m);

fig2_path = fullfile(pic_dir, sprintf('%s.pdf', figure_names{2}));
save_figure(fig2, fig2_path);
fprintf('图2已保存至: %s (色标：%s 暂未显示)\n', fig2_path, INFO_LABEL);

%% 图3：Δc (m vs m+1) -----------------------------------------------------
fig3 = figure('Position', [140, 140, FIG_WIDTH, FIG_HEIGHT], 'Color', 'white');
ax3 = axes('Parent', fig3);
hold(ax3, 'on');

m_base_full = res_m_vs_1.pulse_counts(:);
if numel(m_base_full) < 2
    error('结果中缺少足够的 pulse_counts 信息以计算 m vs (m+1) 敏感性。');
end
m_base = m_base_full(1:end-1);
for idx = 1:numel(m_base)
    color_idx = min(max(m_base(idx) + 1, 1), size(function_colors, 1));
    plot(ax3, thresholds, delta_c_mm1(:, idx), 'LineWidth', LINE_WIDTH, ...
        'Color', function_colors(color_idx, :), ...
        'DisplayName', sprintf('\\Delta c_{%d\\rightarrow%d}', m_base(idx), m_base(idx) + 1));
end

% 在最大值所在的 M_T 处绘制红色虚线
[peak_value, peak_linear_idx] = max(delta_c_mm1(:));
if isfinite(peak_value)
    [peak_row, ~] = ind2sub(size(delta_c_mm1), peak_linear_idx);
    peak_threshold = thresholds(peak_row);
    xline(ax3, peak_threshold, '--', 'Color', [0.85 0 0], 'LineWidth', 2.5);
end

configure_axes(ax3, 'M_T', 'm vs (m+1) Sensitivity', GRID_ON, GRID_ALPHA, GRID_LINE_STYLE, TICK_DIR);
set_axis_limits(ax3, thresholds, delta_c_mm1);

fig3_path = fullfile(pic_dir, sprintf('%s.pdf', figure_names{3}));
save_figure(fig3, fig3_path);
fprintf('图3已保存至: %s (色标：%s 暂未显示)\n', fig3_path, INFO_LABEL);

%% 单独输出色标 -----------------------------------------------------------
cb_levels = size(function_colors, 1);
fig_cb_width = 720;
fig_cb_height = 180;
fig_cb = figure('Position', [100, 100, fig_cb_width, fig_cb_height], 'Color', 'white');

ax_cb = axes('Parent', fig_cb, 'Position', [0.06 0.35 0.88 0.25], 'Visible', 'off');
colormap(ax_cb, function_colors);
clim(ax_cb, [1, cb_levels]);
cb = colorbar('peer', ax_cb, 'southoutside', 'Ticks', 1:cb_levels, ...
    'TickLabels', string(1:cb_levels), 'LineWidth', 1.5);
cb.TickLabelInterpreter = 'tex';
cb.FontName = FONT_NAME;
cb.FontSize = COLORBAR_TICK_FONT_SIZE;
cb.FontWeight = COLORBAR_TICK_FONT_WEIGHT;
cb.Label.String = '';
cb.Label.Visible = 'off';
cb.Title.String = INFO_LABEL;
cb.Title.FontName = FONT_NAME;
cb.Title.FontSize = COLORBAR_LABEL_FONT_SIZE;
cb.Title.FontWeight = COLORBAR_LABEL_FONT_WEIGHT;
cb.Title.HorizontalAlignment = 'center';
cb.AxisLocation = 'out';
set(cb, 'Box', 'on');

% 调整色条宽度与位置
cb_pos = cb.Position;
cb_pos(1) = 0.06;
cb_pos(3) = 0.88;
cb_pos(2) = 0.28;
cb_pos(4) = max(0.12, cb_pos(4));
cb.Position = cb_pos;

fig_cb_path = fullfile(pic_dir, sprintf('%s.pdf', figure_names{4}));
exportgraphics(fig_cb, fig_cb_path, 'ContentType', 'vector');
fprintf('色标图已保存至: %s\n', fig_cb_path);
close(fig_cb);

%% 辅助函数 ---------------------------------------------------------------
function colors = create_blue_gradient(n)
% create_blue_gradient 生成从深蓝到浅蓝的颜色表
    deep_blue = [0, 45, 160] / 255;
    colors = zeros(n, 3);
    for ii = 1:n
        t = (ii - 1) / max(n - 1, 1);
        lighten = 0.8 * t;
        colors(ii, :) = deep_blue * (1 - lighten) + lighten;
    end
    colors = min(max(colors, 0), 1);
end

function configure_axes(ax, x_label, y_label, grid_on, grid_alpha, grid_style, tick_dir)
% configure_axes 统一轴样式
    ax.FontName = 'Arial';
    ax.FontSize = 13;
    ax.FontWeight = 'Bold';
    ax.LineWidth = 1.5;
    grid(ax, grid_on);
    ax.GridAlpha = grid_alpha;
    ax.GridLineStyle = grid_style;
    ax.TickDir = tick_dir;
    box(ax, 'on');

    xlabel(ax, x_label, 'Interpreter', 'tex', 'FontName', 'Arial', ...
        'FontSize', 12, 'FontWeight', 'Bold');
    ylabel(ax, y_label, 'Interpreter', 'tex', 'FontName', 'Arial', ...
        'FontSize', 12, 'FontWeight', 'Bold');
end

function set_axis_limits(ax, x_data, y_data)
% set_axis_limits 根据数据自适应坐标范围
    finite_mask = isfinite(y_data);
    if ~any(finite_mask, 'all')
        return;
    end
    x_min = min(x_data(:), [], 'omitnan');
    x_max = max(x_data(:), [], 'omitnan');
    y_max = max(y_data(finite_mask), [], 'omitnan');

    if x_min < x_max
        xlim(ax, [x_min, x_max]);
    end
    y_min = 0;
    if y_max <= 0
        y_max = 1;
    else
        margin = 0.05 * y_max;
        y_max = y_max + margin;
    end
    ylim(ax, [y_min, y_max]);
end

function save_figure(fig_handle, output_file)
% save_figure 使用矢量格式保存图像
    exportgraphics(fig_handle, output_file, 'ContentType', 'vector');
end
