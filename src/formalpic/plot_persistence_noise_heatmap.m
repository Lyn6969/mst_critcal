% plot_persistence_noise_heatmap - 持久性噪声扫描热力图绘制
%
% 功能说明：
%   从 run_persistence_scan 生成的结果文件中读取持久性数据，
%   绘制运动显著性阈值 M_T × 真实噪声幅度 \eta 的热力图。
%   颜色表示平均持久性 P，数值越大表示群体越稳定。

clear; clc; close all;

%% -------------------- 图像样式配置 --------------------
FIG_WIDTH = 500;          % 图像宽度（像素）
FIG_HEIGHT = 400;         % 图像高度（像素）

FONT_NAME = 'Arial';      % 全局字体
LABEL_FONT_SIZE = 12;     % 坐标轴标签字体大小
LABEL_FONT_WEIGHT = 'Bold';
TICK_FONT_SIZE = 13;      % 刻度字体大小
TICK_FONT_WEIGHT = 'Bold';
COLORBAR_FONT_SIZE = 12;  % 色阶条字体大小
AXIS_LINE_WIDTH = 1.5;    % 坐标轴线宽
COLORBAR_LINE_WIDTH = 1.5;
TICK_DIR = 'in';          % 刻度方向

CONTOUR_LEVELS = 40;      % 等高线层数，保证热力图平滑

%% -------------------- 数据路径配置 --------------------
mat_file = 'persistence_results.mat';
mat_dir = fullfile('data', 'experiments', 'persistence_scan', ...
    '20251105_221002');  % TODO：按实际时间戳修改目录

script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(script_dir));

mat_path_abs = fullfile(project_root, mat_dir, mat_file);

if ~isfile(mat_path_abs)
    error('未找到结果文件，请检查路径: %s', mat_path_abs);
end

data = load(mat_path_abs, 'results');
results = data.results;

%% -------------------- 数据提取与转换 --------------------
cj_thresholds = results.cj_thresholds(:)';
noise_levels = results.noise_levels(:);          % D_\theta

if isfield(results, 'P_mean') && ~isempty(results.P_mean)
    P_matrix = results.P_mean;
else
    D_mean = results.D_mean;
    if isfield(results, 'config') && isfield(results.config, 'min_diffusion')
        min_diffusion = results.config.min_diffusion;
    else
        min_diffusion = 1e-3;
    end
    P_matrix = 1 ./ sqrt(max(D_mean, min_diffusion));
end

P_matrix = max(P_matrix, 0);  % 截断负值

eta_levels = sqrt(2 * noise_levels);

[num_eta, num_cj] = size(P_matrix);
if num_eta ~= numel(eta_levels) || num_cj ~= numel(cj_thresholds)
    error('数据维度不匹配，请核实结果文件。');
end

if isfield(results, 'base_params') && isfield(results.base_params, 'N')
    N_particles = results.base_params.N;
else
    N_particles = 200;
end

%% -------------------- 输出目录配置 --------------------
pic_dir = fullfile(project_root, 'pic');
if ~exist(pic_dir, 'dir')
    mkdir(pic_dir);
end

output_name = sprintf('persistence_noise_heatmap_N%d.pdf', N_particles);
output_path = fullfile(pic_dir, output_name);
output_name_norm = sprintf('persistence_noise_heatmap_normalized_N%d.pdf', N_particles);
output_path_norm = fullfile(pic_dir, output_name_norm);

%% -------------------- 绘制热力图 --------------------
fig = figure('Position', [200, 200, FIG_WIDTH, FIG_HEIGHT], 'Color', 'white');
set(fig, 'Renderer', 'painters');
ax = axes('Parent', fig);

contourf(ax, cj_thresholds, eta_levels, P_matrix, CONTOUR_LEVELS, 'LineColor', 'none');
set(ax, 'YDir', 'normal');
colormap(ax, 'turbo');
box(ax, 'on');
axis(ax, 'tight');

ax.FontName = FONT_NAME;
ax.FontSize = TICK_FONT_SIZE;
ax.FontWeight = TICK_FONT_WEIGHT;
ax.LineWidth = AXIS_LINE_WIDTH;
ax.TickDir = TICK_DIR;

xlabel(ax, 'M_T', 'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);
ylabel(ax, '\eta', 'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);

cb = colorbar(ax);
cb.Label.String = 'Persistence P';
cb.Label.FontName = FONT_NAME;
cb.Label.FontSize = COLORBAR_FONT_SIZE;
cb.Label.FontWeight = LABEL_FONT_WEIGHT;
cb.FontName = FONT_NAME;
cb.FontSize = TICK_FONT_SIZE;
cb.LineWidth = COLORBAR_LINE_WIDTH;

exportgraphics(fig, output_path, 'ContentType', 'vector');

fprintf('持久性噪声热力图已保存至: %s\n', output_path);
fprintf('参数范围: M_T ∈ [%.1f, %.1f], \eta ∈ [%.3f, %.3f]\n', ...
    min(cj_thresholds), max(cj_thresholds), min(eta_levels), max(eta_levels));
fprintf('P 范围: [%.3f, %.3f]\n', ...
    min(P_matrix(:), [], 'omitnan'), max(P_matrix(:), [], 'omitnan'));

%% -------------------- 归一化持久性热力图 --------------------
finite_vals = P_matrix(isfinite(P_matrix));
if isempty(finite_vals)
    P_norm = zeros(size(P_matrix));
else
    p_min = min(finite_vals);
    p_max = max(finite_vals);
    if abs(p_max - p_min) < 1e-12
        P_norm = zeros(size(P_matrix));
    else
        P_norm = (P_matrix - p_min) ./ (p_max - p_min);
    end
end

fig_norm = figure('Position', [220, 220, FIG_WIDTH, FIG_HEIGHT], 'Color', 'white');
set(fig_norm, 'Renderer', 'painters');
ax_norm = axes('Parent', fig_norm);

contourf(ax_norm, cj_thresholds, eta_levels, P_norm, CONTOUR_LEVELS, 'LineColor', 'none');
set(ax_norm, 'YDir', 'normal');
colormap(ax_norm, 'turbo');
box(ax_norm, 'on');
axis(ax_norm, 'tight');
hold(ax_norm, 'on');

% 绘制归一化持久性 0.6 等值线，突出关键敏感带
[C_norm, h_norm] = contour(ax_norm, cj_thresholds, eta_levels, P_norm, [0.6 0.6], ...
    'LineColor', [0 0 0], 'LineWidth', 1.8, 'LineStyle', '--');
if ~isempty(h_norm) && ishghandle(h_norm)
    clabel(C_norm, h_norm, 'FontName', FONT_NAME, 'FontSize', 11, ...
        'FontWeight', 'Bold', 'BackgroundColor', 'none');
end
hold(ax_norm, 'off');

ax_norm.FontName = FONT_NAME;
ax_norm.FontSize = TICK_FONT_SIZE;
ax_norm.FontWeight = TICK_FONT_WEIGHT;
ax_norm.LineWidth = AXIS_LINE_WIDTH;
ax_norm.TickDir = TICK_DIR;

xlabel(ax_norm, 'M_T', 'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);
ylabel(ax_norm, '\eta', 'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);

cb_norm = colorbar(ax_norm);
cb_norm.Label.String = 'Normalized Persistence';
cb_norm.Label.FontName = FONT_NAME;
cb_norm.Label.FontSize = COLORBAR_FONT_SIZE;
cb_norm.Label.FontWeight = LABEL_FONT_WEIGHT;
cb_norm.FontName = FONT_NAME;
cb_norm.FontSize = TICK_FONT_SIZE;
cb_norm.LineWidth = COLORBAR_LINE_WIDTH;
clim(ax_norm, [0, 1]);

exportgraphics(fig_norm, output_path_norm, 'ContentType', 'vector');

fprintf('归一化持久性热力图已保存至: %s\n', output_path_norm);
