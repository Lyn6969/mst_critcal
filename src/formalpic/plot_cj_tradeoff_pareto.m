% plot_cj_tradeoff_pareto
% =========================================================================
% 功能：
%   - 读取 run_cj_tradeoff_adaptive_scan_shared_seed 生成的 summary 数据，
%     以论文风格绘制固定阈值的 R-P 散点分布与 Pareto 前沿，并突出自适应阈值结果。
% 使用说明：
%   - 修改 mat_file & mat_dir 以指向实际生成的 MAT 结果文件。
% =========================================================================

clear; clc; close all;

%% -------------------- 图像样式配置 --------------------
FIG_SIZE = [150, 150, 520, 520];      % 方正布局，便于排版
FONT_NAME = 'Arial';
LABEL_FONT_SIZE = 12;
LABEL_FONT_WEIGHT = 'Bold';
TICK_FONT_SIZE = 13;
TICK_FONT_WEIGHT = 'Bold';
AXIS_LINE_WIDTH = 1.5;
TICK_DIR = 'in';
SCATTER_SIZE = 46;
ADAPTIVE_MARKER_SIZE = 150;
FRONT_LINE_WIDTH = 3.0;
FRONT_COLORMAP = turbo(256);
GREY_COLOR = [0.7 0.7 0.7];
COLORBAR_LINE_WIDTH = 1.5;
GRID_LINE_STYLE = '--';
GRID_LINE_WIDTH = 1.0;
GRID_ALPHA = 0.1;

%% -------------------- 数据路径配置 --------------------
% TODO: 根据实际输出文件更新时间戳
mat_dir = fullfile('results', 'tradeoff');
mat_file = 'cj_tradeoff_adaptive_shared_seed_20251112_181943.mat';

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
P_sorted = P_front(order);
cj_sorted = cj_front(order);

if numel(R_sorted) >= 4
    R_dense = linspace(R_sorted(1), R_sorted(end), 160);
    P_dense = interp1(R_sorted, P_sorted, R_dense, 'pchip');
    CJ_dense = interp1(R_sorted, cj_sorted, R_dense, 'pchip');
else
    R_dense = R_sorted;
    P_dense = P_sorted;
    CJ_dense = cj_sorted;
end

% 根据阈值大小映射颜色（最小值→蓝，最大值→黄）
cj_min = min(cj_vals);
cj_max = max(cj_vals);
cj_norm = (CJ_dense - cj_min) ./ max(cj_max - cj_min, eps);
color_idx = max(1, min(size(FRONT_COLORMAP, 1), round(cj_norm * (size(FRONT_COLORMAP,1)-1)) + 1));
front_colors = FRONT_COLORMAP(color_idx, :);

%% -------------------- 绘图 --------------------
fig = figure('Color', 'white', 'Position', FIG_SIZE);
ax = axes('Parent', fig); hold(ax, 'on');

% 固定阈值散点
scatter(ax, R_fixed, P_fixed, SCATTER_SIZE, cj_vals, 'filled', 'MarkerFaceAlpha', 0.8, ...
    'MarkerEdgeColor', [0.4 0.4 0.4], 'DisplayName', 'Fixed thresholds');

% 自适应阈值标记
R_adapt = adaptive.R_mean;
if ~isfield(adaptive, 'P_mean_norm') || isempty(adaptive.P_mean_norm)
    error('输入 summary 中缺少自适应阈值的 P\_mean\_norm，请确保使用 run\_cj\_tradeoff\_adaptive\_scan\_shared\_seed 的原始输出。');
end
P_adapt = adaptive.P_mean_norm;
scatter(ax, R_adapt, P_adapt, ADAPTIVE_MARKER_SIZE, [0.85 0.2 0.2], 'filled', ...
    'Marker', 'p', 'MarkerEdgeColor', [0.45 0 0]);

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
xlim(ax, [0, 1]);
ylim(ax, [0, 1]);
xticks(ax, 0:0.2:1);
yticks(ax, 0:0.2:1);

% 色条（映射固定阈值大小）
colormap(ax, FRONT_COLORMAP);
caxis(ax, [cj_min, cj_max]);
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

base_name = sprintf('cj_tradeoff_pareto_%s', summary.timestamp);
pdf_path = fullfile(pic_dir, [base_name, '.pdf']);
png_path = fullfile(pic_dir, [base_name, '.png']);

exportgraphics(fig, pdf_path, 'ContentType', 'vector');
exportgraphics(fig, png_path, 'Resolution', 300);

fprintf('Pareto 图已保存：\n  PDF: %s\n  PNG: %s\n', pdf_path, png_path);

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
