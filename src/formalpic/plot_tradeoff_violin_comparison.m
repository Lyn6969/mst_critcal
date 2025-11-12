% plot_tradeoff_violin_comparison - Fixed vs Adaptive violin comparisons
%
% 功能：
%   - 读取 cj_tradeoff_adaptive_shared_seed_* 实验结果
%   - 分别绘制响应性与持久性的小提琴图（M_T=1/3/5 vs Adaptive）
%   - 图像风格与 formalpic 目录下其它论文图保持一致
%
% 输出：
%   - pic/tradeoff_violin_responsivity.pdf
%   - pic/tradeoff_violin_persistence.pdf

clear; clc; close all;

%% -------------------- 图像样式配置 --------------------
FIG_WIDTH = 400;
FIG_HEIGHT = 300;
FONT_NAME = 'Arial';
LABEL_FONT_SIZE = 12;
LABEL_FONT_WEIGHT = 'Bold';
TICK_FONT_SIZE = 13;
TICK_FONT_WEIGHT = 'Bold';
AXIS_LINE_WIDTH = 1.5;
TICK_DIR = 'in';
VIOLIN_HALF_WIDTH = 0.32;
VIOLIN_ALPHA = 0.65;

COLOR_MT1 = [0.78, 0.88, 0.96];
COLOR_MT3 = [0.36, 0.62, 0.88];
COLOR_MT5 = [0.13, 0.31, 0.66];
COLOR_ADAPTIVE = [0.85, 0.20, 0.20];

%% -------------------- 数据路径配置 --------------------
mat_file = 'cj_tradeoff_adaptive_shared_seed_20251105_221310.mat'; % TODO: 根据需要修改

script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(script_dir));  % formalpic 必须仅向上两级

candidates = {
    fullfile(project_root, 'src', 'sim', 'experiments', 'results', mat_file), ...
    fullfile(project_root, 'results', 'tradeoff', mat_file)
};
mat_path = '';
for k = 1:numel(candidates)
    if isfile(candidates{k})
        mat_path = candidates{k};
        break;
    end
end
if isempty(mat_path)
    error('未找到数据文件：%s', strjoin(candidates, ' , '));
end

%% -------------------- 加载数据 --------------------
data = load(mat_path, 'summary');
summary = data.summary;

fixed = summary.results.fixed;
adaptive = summary.results.adaptive;
cj_thresholds = fixed.cj_thresholds(:);

% 精确匹配 M_T = 1, 3, 5
idx_map = containers.Map({'1.0','3.0','5.0'}, {NaN,NaN,NaN});
targets = [1.0, 3.0, 5.0];
for t = targets
    idx = find(abs(cj_thresholds - t) < 1e-6, 1);
    if isempty(idx)
        error('未找到 M_T = %.1f 的数据点。', t);
    end
    idx_map(sprintf('%.1f', t)) = idx;
end

% 响应性数据
R_data = {
    fixed.R_raw(idx_map('1.0'), :)', ...
    fixed.R_raw(idx_map('3.0'), :)', ...
    fixed.R_raw(idx_map('5.0'), :)', ...
    adaptive.R_raw(:)
};

% 原始持久性数据
P_mt1_raw = fixed.P_raw(idx_map('1.0'), :)';
P_mt3_raw = fixed.P_raw(idx_map('3.0'), :)';
P_mt5_raw = fixed.P_raw(idx_map('5.0'), :)';
P_adaptive_raw = adaptive.P_raw(:);

if ~isfield(summary, 'P_min') || ~isfield(summary, 'P_range') || summary.P_range <= 0
    error('summary 中缺少 P_min/P_range 或 P_range<=0，无法进行持久性归一化。');
end

P_mt1 = (P_mt1_raw - summary.P_min) / summary.P_range;
P_mt3 = (P_mt3_raw - summary.P_min) / summary.P_range;
P_mt5 = (P_mt5_raw - summary.P_min) / summary.P_range;
P_adaptive = (P_adaptive_raw - summary.P_min) / summary.P_range;

% 为保证小提琴图刻度在 [0,1]，对归一化结果进行截断（均值法计算的 P_min/P_max 可能小于样本极值）
P_mt1 = min(max(P_mt1, 0), 1);
P_mt3 = min(max(P_mt3, 0), 1);
P_mt5 = min(max(P_mt5, 0), 1);
P_adaptive = min(max(P_adaptive, 0), 1);

P_data = {P_mt1, P_mt3, P_mt5, P_adaptive};

labels = {'M_T=1', 'M_T=3', 'M_T=5', 'Adaptive'};
colors = {COLOR_MT1, COLOR_MT3, COLOR_MT5, COLOR_ADAPTIVE};

%% -------------------- 绘制响应性 --------------------
fig_r = figure('Position', [140, 140, FIG_WIDTH, FIG_HEIGHT], 'Color', 'white');
ax_r = axes('Parent', fig_r);
plot_violin_group(ax_r, R_data, labels, colors, VIOLIN_HALF_WIDTH, VIOLIN_ALPHA);
format_axes(ax_r, 'Responsivity', FONT_NAME, LABEL_FONT_SIZE, LABEL_FONT_WEIGHT, ...
    TICK_FONT_SIZE, TICK_FONT_WEIGHT, AXIS_LINE_WIDTH, TICK_DIR);
ylim(ax_r, [0, 1]);
title(ax_r, 'Responsivity Distribution', 'FontName', FONT_NAME, ...
    'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);

%% -------------------- 绘制持久性 --------------------
fig_p = figure('Position', [200, 200, FIG_WIDTH, FIG_HEIGHT], 'Color', 'white');
ax_p = axes('Parent', fig_p);
plot_violin_group(ax_p, P_data, labels, colors, VIOLIN_HALF_WIDTH, VIOLIN_ALPHA);
format_axes(ax_p, 'Normalized Persistence', FONT_NAME, LABEL_FONT_SIZE, LABEL_FONT_WEIGHT, ...
    TICK_FONT_SIZE, TICK_FONT_WEIGHT, AXIS_LINE_WIDTH, TICK_DIR);
ylim(ax_p, [0, 1]);
title(ax_p, 'Persistence Distribution', 'FontName', FONT_NAME, ...
    'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);

%% -------------------- 导出 --------------------
pic_dir = fullfile(project_root, 'pic');
if ~exist(pic_dir, 'dir'); mkdir(pic_dir); end

output_r = fullfile(pic_dir, 'tradeoff_violin_responsivity.pdf');
output_p = fullfile(pic_dir, 'tradeoff_violin_persistence.pdf');
exportgraphics(fig_r, output_r, 'ContentType', 'vector');
exportgraphics(fig_p, output_p, 'ContentType', 'vector');
fprintf('Responsivity violin saved: %s\n', output_r);
fprintf('Persistence violin saved: %s\n', output_p);

%% ========================================================================
% 绘图辅助函数
%% ========================================================================
function plot_violin_group(ax, data_cell, labels, colors, half_width, face_alpha)
    hold(ax, 'on');
    n_groups = numel(data_cell);

    for i = 1:n_groups
        data = data_cell{i};
        data = data(~isnan(data));
        if isempty(data)
            continue;
        end

        [f, xi] = ksdensity(data, 'NumPoints', 200);
        f_norm = f / max(f) * half_width;

        patch(ax, [i - f_norm, fliplr(i + f_norm)], [xi, fliplr(xi)], colors{i}, ...
            'EdgeColor', 'none', 'FaceAlpha', face_alpha);

        med = median(data);
        q1 = quantile(data, 0.25);
        q3 = quantile(data, 0.75);
        dmin = min(data);
        dmax = max(data);

        plot(ax, [i-0.08, i+0.08], [med, med], 'k-', 'LineWidth', 2.2);
        rectangle(ax, 'Position', [i-0.08, q1, 0.16, q3-q1], ...
            'EdgeColor', [0.3 0.3 0.3], 'LineWidth', 1.2, 'FaceColor', 'none');

        plot(ax, [i, i], [q3, dmax], 'k-', 'LineWidth', 1);
        plot(ax, [i, i], [dmin, q1], 'k-', 'LineWidth', 1);
        plot(ax, [i-0.035, i+0.035], [dmax, dmax], 'k-', 'LineWidth', 1);
        plot(ax, [i-0.035, i+0.035], [dmin, dmin], 'k-', 'LineWidth', 1);
    end

    xlim(ax, [0.5, n_groups + 0.5]);
    set(ax, 'XTick', 1:n_groups, 'XTickLabel', labels);
    hold(ax, 'off');
end

function format_axes(ax, ylabel_str, font_name, label_size, label_weight, ...
    tick_size, tick_weight, axis_line_width, tick_dir)
    ylabel(ax, ylabel_str, 'FontName', font_name, 'FontSize', label_size, ...
        'FontWeight', label_weight);
    ax.FontName = font_name;
    ax.FontSize = tick_size;
    ax.FontWeight = tick_weight;
    ax.LineWidth = axis_line_width;
    ax.TickDir = tick_dir;
    box(ax, 'on');
    grid(ax, 'off');
end
