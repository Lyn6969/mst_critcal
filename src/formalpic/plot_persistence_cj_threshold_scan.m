% plot_persistence_cj_threshold_scan - 持久性阈值扫描结果绘图脚本
%
% 功能：读取 run_persistence_cj_threshold_scan 生成的数据，
%       绘制 M_T (cj_threshold) 与归一化持久性之间的折线关系，
%       并以阴影展示 ± 标准误差区域。

clear; clc; close all;

%% -------------------- 图像样式配置 --------------------
FIG_WIDTH = 400;           % 图片宽度（像素）
FIG_HEIGHT = 200;          % 图片高度（像素）
LINE_COLOR = [38, 94, 180] / 255; % 主折线颜色（蓝色）
SHADE_COLOR = LINE_COLOR;  % 阴影颜色
LINE_WIDTH = 2.5;          % 折线线宽
AXIS_LINE_WIDTH = 1.5;     % 坐标轴线宽
FONT_NAME = 'Arial';       % 字体
LABEL_FONT_SIZE = 12;      % 轴标签字体
LABEL_FONT_WEIGHT = 'Bold';
TICK_FONT_SIZE = 13;       % 刻度字体
TICK_FONT_WEIGHT = 'Bold';
TICK_DIR = 'in';
SHADE_ALPHA = 0.25;        % 阴影透明度

%% -------------------- 数据路径配置 --------------------
% TODO: 将下列文件名替换为实际的时间戳文件
results_dir = fullfile('results', 'persistence');
mat_file = 'persistence_cj_scan_20251105_214654.mat';

script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(script_dir));

mat_path_abs = fullfile(project_root, results_dir, mat_file);
if ~isfile(mat_path_abs)
    error('未找到结果文件，请检查路径: %s', mat_path_abs);
end

data = load(mat_path_abs, 'results');
results = data.results;

if isfield(results, 'parameters') && isfield(results.parameters, 'angleNoiseIntensity')
    D_theta = results.parameters.angleNoiseIntensity;
else
    D_theta = NaN;
end
eta_value = sqrt(2 * D_theta);

%% -------------------- 数据整理 --------------------
cj_thresholds = results.cj_thresholds(:);
P_mean = results.P_mean(:);
P_sem = results.P_sem(:);

upper_bound = P_mean + P_sem;
lower_bound = max(P_mean - P_sem, 0);

%% -------------------- 输出目录配置 --------------------
pic_dir = fullfile(project_root, 'pic');
if ~exist(pic_dir, 'dir')
    mkdir(pic_dir);
end

agent_count = NaN;
if isfield(results, 'parameters') && isfield(results.parameters, 'N')
    agent_count = results.parameters.N;
end
if isnan(agent_count)
    output_name = 'persistence_vs_mt.pdf';
else
    output_name = sprintf('persistence_vs_mt_N%d.pdf', agent_count);
end
output_path = fullfile(pic_dir, output_name);

%% -------------------- 绘制折线图 --------------------
fig = figure('Position', [140, 140, FIG_WIDTH, FIG_HEIGHT], 'Color', 'white');
ax = axes('Parent', fig);
hold(ax, 'on');

fill(ax, [cj_thresholds; flip(cj_thresholds)], ...
    [upper_bound; flip(lower_bound)], ...
    SHADE_COLOR, 'FaceAlpha', SHADE_ALPHA, 'EdgeColor', 'none');

plot(ax, cj_thresholds, P_mean, '-', 'LineWidth', LINE_WIDTH, 'Color', LINE_COLOR);

ax.FontName = FONT_NAME;
ax.FontSize = TICK_FONT_SIZE;
ax.FontWeight = TICK_FONT_WEIGHT;
ax.LineWidth = AXIS_LINE_WIDTH;
ax.TickDir = TICK_DIR;
box(ax, 'on');
grid(ax, 'off');

xlabel(ax, 'M_T', 'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);
ylabel(ax, 'Persistence', 'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);

xlim(ax, [min(cj_thresholds), max(cj_thresholds)]);
ylim(ax, [0, max(upper_bound) * 1.05]);

if ~isnan(eta_value)
    text(ax, 'Units', 'normalized', 'Position', [0.97, 0.08], ...
        'String', sprintf('\\eta = %.1f', eta_value), ...
        'HorizontalAlignment', 'right', 'FontName', FONT_NAME, ...
        'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);
end

hold(ax, 'off');

%% -------------------- 导出与信息输出 --------------------
exportgraphics(fig, output_path, 'ContentType', 'vector');
fprintf('Persistence figure saved to: %s\n', output_path);
if ~isnan(D_theta)
    fprintf('噪声强度 D_theta = %.1f, 对应 eta = %.1f\n', D_theta, eta_value);
end
