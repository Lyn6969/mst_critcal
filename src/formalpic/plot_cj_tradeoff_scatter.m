% plot_cj_tradeoff_scatter
% =========================================================================
% 功能：
%   - 读取 run_cj_tradeoff_adaptive_scan_shared_seed 生成的 summary 数据，
%     以论文风格绘制固定阈值的 R-P 散点分布与 散点对比，并突出自适应阈值结果。
% 使用说明：
%   - 修改 mat_file & mat_dir 以指向实际生成的 MAT 结果文件。
% =========================================================================

clear; clc; close all;

%% -------------------- 图像样式配置 --------------------
FIG_SIZE = [150, 150, 600,600];      % 方正布局，便于排版
FONT_NAME = 'Arial';
LABEL_FONT_SIZE = 15;
LABEL_FONT_WEIGHT = 'Bold';
TICK_FONT_SIZE = 14;
TICK_FONT_WEIGHT = 'Bold';
AXIS_LINE_WIDTH = 1.5;
TICK_DIR = 'in';
SCATTER_SIZE = 46;
ADAPTIVE_MARKER_SIZE = 280;
FRONT_LINE_WIDTH = 3.0;
CURVE_LINE_WIDTH = 4;
FRONT_COLORMAP = turbo(256);
GREY_COLOR = [0.7 0.7 0.7];
COLORBAR_LINE_WIDTH = 1.5;
GRID_LINE_STYLE = '--';
GRID_LINE_WIDTH = 1.5;
GRID_ALPHA = 0.1;
ENABLE_SMOOTH_CURVE = true;   % 需要时设置为 true，可绘制渐变平滑曲线

%% -------------------- 数据路径配置 --------------------
% TODO: 根据实际输出文件更新时间戳
mat_dir = fullfile('results', 'tradeoff');
mat_file = 'cj_tradeoff_adaptive_shared_seed_20251112_234029.mat';

script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(script_dir));
mat_path = fullfile(project_root, mat_dir, mat_file);

if ~isfile(mat_path)
    error('未找到数据文件：%s', mat_path);
end

data = load(mat_path, 'summary');
summary = data.summary;

fixed = summary.results.fixed;
adaptive = summary.results.adaptive;
cj_vals = summary.cj_thresholds_fixed(:);

R_fixed = fixed.R_mean(:);
if ~isfield(fixed, 'P_mean_norm') || isempty(fixed.P_mean_norm)
    error('输入 summary 中缺少固定阈值的 P\_mean\_norm，请确保使用 run\_cj\_tradeoff\_adaptive\_scan\_shared\_seed 的原始输出。');
end
P_fixed = fixed.P_mean_norm(:);

mask_valid = ~isnan(R_fixed) & ~isnan(P_fixed);
R_fixed = R_fixed(mask_valid);
P_fixed = P_fixed(mask_valid);
cj_vals = cj_vals(mask_valid);

if isempty(R_fixed)
    error('固定阈值结果为空，请检查输入数据。');
end

%% -------------------- 计算 Pareto 前沿 --------------------
% 使用 paretofront 判定非支配解（需最大化 R 与 P）
if exist('paretofront', 'file') == 2
    front_mask = paretofront([-R_fixed, -P_fixed]);
else
    front_mask = local_paretofront([-R_fixed, -P_fixed]);
end
R_front = R_fixed(front_mask);
P_front = P_fixed(front_mask);
cj_front = cj_vals(front_mask);

% 按 R 值排序并平滑插值，生成光滑前沿
[R_sorted, order] = sort(R_front);
% 阈值范围（用于色彩映射）
cj_min = min(cj_vals);
cj_max = max(cj_vals);

%% -------------------- 绘图 --------------------
fig = figure('Color', 'white', 'Position', FIG_SIZE);
ax = axes('Parent', fig); hold(ax, 'on');

if ENABLE_SMOOTH_CURVE && numel(R_fixed) >= 4
    [cj_sorted, order] = sort(cj_vals);
    R_sorted = R_fixed(order);
    P_sorted = P_fixed(order);
    cj_dense = linspace(cj_sorted(1), cj_sorted(end), 400);

    smooth_param = 1;
    R_spline = csaps(cj_sorted, R_sorted, smooth_param);
    P_spline = csaps(cj_sorted, P_sorted, smooth_param);
    R_dense = fnval(R_spline, cj_dense);
    P_dense = fnval(P_spline, cj_dense);

    cj_norm = (cj_dense - cj_min) ./ max(cj_max - cj_min, eps);
    color_idx = max(1, min(size(FRONT_COLORMAP,1), round(cj_norm * (size(FRONT_COLORMAP,1)-1)) + 1));
    for idx = 1:numel(cj_dense)-1
        line(ax, R_dense(idx:idx+1), P_dense(idx:idx+1), 'Color', FRONT_COLORMAP(color_idx(idx), :), ...
            'LineWidth', CURVE_LINE_WIDTH, 'HandleVisibility', 'off', 'LineJoin', 'round');
    end
else
    % 固定阈值散点
    scatter(ax, R_fixed, P_fixed, SCATTER_SIZE, cj_vals, 'filled', 'MarkerFaceAlpha', 0.8, ...
        'MarkerEdgeColor', [0.4 0.4 0.4], 'DisplayName', 'Fixed thresholds');
end

% 自适应阈值标记
R_adapt = adaptive.R_mean;
if ~isfield(adaptive, 'P_mean_norm') || isempty(adaptive.P_mean_norm)
    error('输入 summary 中缺少自适应阈值的 P\_mean\_norm，请确保使用 run\_cj\_tradeoff\_adaptive\_scan\_shared\_seed 的原始输出。');
end
P_adapt = adaptive.P_mean_norm;

% 最近邻索引（按持久性匹配）
if ~exist('idx_ref', 'var')
    [~, idx_ref] = min(abs(fixed.P_mean_norm - P_adapt));
end
scatter(ax, R_adapt, P_adapt, ADAPTIVE_MARKER_SIZE, [0.85 0.2 0.2], 'filled', ...
    'Marker', 'p', 'MarkerEdgeColor', [0.45 0 0]);
text(ax, R_adapt-0.08, P_adapt -0.06, 'Adaptive M_T', 'FontName', FONT_NAME, ...
    'FontSize', LABEL_FONT_SIZE, 'FontWeight', 'Bold', 'HorizontalAlignment', 'left');

R_ref = fixed.R_mean(idx_ref);
P_ref = fixed.P_mean_norm(idx_ref);
if ~ENABLE_SMOOTH_CURVE || numel(R_fixed) < 4
    scatter(ax, R_ref, P_ref, SCATTER_SIZE * 1.2, [0.2 0.2 0.2], 'filled', ...
        'Marker', 'o', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceAlpha', 0.9);
end
resp_gain_vis = R_adapt / R_ref;
h_gain_line = plot(ax, [R_ref, R_adapt], [P_ref, P_ref], '--', 'Color', [0.15 0.15 0.15], ...
    'LineWidth', 1.3, 'HandleVisibility', 'off');
uistack(h_gain_line, 'bottom');
text(ax, (R_ref + R_adapt)/2, P_ref + 0.025, sprintf('+%.1f%%', (resp_gain_vis - 1)*100), ...
    'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE - 1, 'FontWeight', 'Bold', ...
    'HorizontalAlignment', 'center');

% 坐标与样式
ax.FontName = FONT_NAME;
ax.FontSize = TICK_FONT_SIZE;
ax.FontWeight = TICK_FONT_WEIGHT;
ax.LineWidth = AXIS_LINE_WIDTH;
ax.TickDir = TICK_DIR;
box(ax, 'on');
ax.GridLineStyle = GRID_LINE_STYLE;
ax.GridColor = [0 0 0];
ax.GridAlpha = GRID_ALPHA;
ax.LineWidth = AXIS_LINE_WIDTH;
ax.Layer = 'top';
ax.XGrid = 'on';
ax.YGrid = 'on';
xlabel(ax, 'Responsivity', 'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);
ylabel(ax, 'Norm. Persistence', 'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);

axis(ax, 'square');
xlim(ax, [0, 1.05]);
ylim(ax, [0, 1.05]);
xticks(ax, 0:0.2:1);
yticks(ax, 0:0.2:1);

% 色条（映射固定阈值大小）
colormap(ax, FRONT_COLORMAP);
clim(ax, [cj_min, cj_max]);
cb = colorbar(ax);
cb.Label.String = 'Fixed M_T';
cb.Label.FontName = FONT_NAME;
cb.Label.FontSize = LABEL_FONT_SIZE;
cb.FontName = FONT_NAME;
cb.FontSize = TICK_FONT_SIZE;
cb.Ticks = linspace(floor(cj_min), ceil(cj_max), min(6, ceil(cj_max)-floor(cj_min)+1));
cb.LineWidth = COLORBAR_LINE_WIDTH;

legend(ax, 'off');

%% -------------------- 导出 --------------------
pic_dir = fullfile(project_root, 'pic');
if ~exist(pic_dir, 'dir'); mkdir(pic_dir); end

if isfield(summary.results, 'adaptive') && isfield(summary.results.adaptive, 'saliency_threshold')
    eta_label = strrep(sprintf('eta_%0.3f', sqrt(2 * summary.base_params_resp.angleNoiseIntensity)), '.', 'p');
else
    eta_label = 'eta_unknown';
end
base_name = sprintf('cj_tradeoff_scatter_%s', eta_label);
pdf_path = fullfile(pic_dir, [base_name, '.pdf']);

exportgraphics(fig, pdf_path, 'ContentType', 'vector');

% 最近邻性能增益统计
R_adapt = adaptive.R_mean;
P_adapt = adaptive.P_mean_norm;
[~, idx_ref] = min(abs(fixed.P_mean_norm - P_adapt));
R_ref = fixed.R_mean(idx_ref);
P_ref = fixed.P_mean_norm(idx_ref);
resp_gain = R_adapt / R_ref;
pers_gain = P_adapt / P_ref;
pers_delta = P_adapt - P_ref;
fprintf('Responsivity gain (adaptive vs nearest fixed): %.2f%%\n', (resp_gain-1)*100);
fprintf('Persistence gain (adaptive vs nearest fixed): %.2f%%\n', (pers_gain-1)*100);
fprintf('Persistence difference (adaptive - fixed): %.3f\n', pers_delta);
fprintf('Scatter 图已保存：\n  PDF: %s\n', pdf_path);

%% -------------------- 辅助函数 --------------------
function mask = local_paretofront(points)
    n = size(points, 1);
    mask = true(n, 1);
    for i = 1:n
        if ~mask(i)
            continue;
        end
        dominates = all(bsxfun(@ge, points, points(i, :)), 2) & any(bsxfun(@gt, points, points(i, :)), 2);
        mask(dominates) = false;
    end
end
