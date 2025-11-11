% plot_responsiveness_cj_threshold_scan - 响应性阈值扫描结果绘图脚本
%
% 功能：读取 run_responsiveness_cj_threshold_scan 生成的结果文件，
%       绘制运动显著性阈值 M_T 与 Responsivity (R_mean) 的关系折线图，
%       并以阴影表示 ±标准误差区域。

% 清理环境
clear; clc; close all;

%% -------------------- 图像样式配置 --------------------
FIG_WIDTH = 400;           % 图片宽度（像素）
FIG_HEIGHT = 200;          % 图片高度（像素）
LINE_COLOR = [0.85 0.1 0.1]; % 主折线颜色（红色）
SHADE_COLOR = LINE_COLOR; % 阴影颜色与主色一致
LINE_WIDTH = 2.5;          % 折线线宽
AXIS_LINE_WIDTH = 1.5;     % 坐标轴线宽
FONT_NAME = 'Arial';       % 字体名称
LABEL_FONT_SIZE = 12;      % 坐标轴标签字体大小
LABEL_FONT_WEIGHT = 'Bold';% 坐标轴标签字体粗细
TICK_FONT_SIZE = 13;       % 刻度字体大小
TICK_FONT_WEIGHT = 'Bold'; % 刻度字体粗细
TICK_DIR = 'in';           % 刻度方向
SHADE_ALPHA = 0.25;        % 阴影透明度

%% -------------------- 数据路径配置 --------------------
% TODO: 根据实际结果文件更新时间戳
results_dir = fullfile('results', 'responsiveness');
mat_file = 'responsiveness_cj_scan_20251111_173907_eta_0p300.mat';

% 计算项目根目录路径
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(script_dir));

% 构建数据文件绝对路径
mat_path_abs = fullfile(project_root, results_dir, mat_file);
if ~isfile(mat_path_abs)
    error('未找到结果文件，请检查路径: %s', mat_path_abs);
end

% 读取结果结构体
data = load(mat_path_abs, 'results');
results = data.results;

% 噪声强度转换为 \eta = sqrt(2 D_\theta)
if isfield(results, 'base_parameters') && isfield(results.base_parameters, 'angleNoiseIntensity')
    eta_value = sqrt(2 * results.base_parameters.angleNoiseIntensity);
else
    eta_value = NaN;
end

%% -------------------- 数据整理 --------------------
cj_thresholds = results.cj_thresholds(:); % M_T 阈值
R_mean = results.R_mean(:);              % 平均响应性
R_sem = results.R_sem(:);                % 标准误差

if numel(cj_thresholds) ~= numel(R_mean)
    error('阈值数量与均值数据长度不一致。');
end

upper_bound = R_mean + R_sem;
lower_bound = R_mean - R_sem;
lower_bound = max(lower_bound, 0); % 响应性下界不为负

%% -------------------- 输出目录配置 --------------------
pic_dir = fullfile(project_root, 'pic');
if ~exist(pic_dir, 'dir')
    mkdir(pic_dir);
end

if isnan(eta_value)
    output_name = sprintf('responsiveness_vs_mt_N%d.pdf', results.base_parameters.N);
else
    eta_tag = strrep(sprintf('eta_%0.3f', eta_value), '.', 'p');
    output_name = sprintf('responsiveness_vs_mt_N%d_%s.pdf', results.base_parameters.N, eta_tag);
end
output_path = fullfile(pic_dir, output_name);

%% -------------------- 绘制折线图 --------------------
fig = figure('Position', [160, 160, FIG_WIDTH, FIG_HEIGHT], 'Color', 'white');
ax = axes('Parent', fig);
hold(ax, 'on');

% 阴影区域
fill(ax, [cj_thresholds; flip(cj_thresholds)], ...
    [upper_bound; flip(lower_bound)], ...
    SHADE_COLOR, 'FaceAlpha', SHADE_ALPHA, 'EdgeColor', 'none');

% 平均响应性折线
plot(ax, cj_thresholds, R_mean, '-', 'LineWidth', LINE_WIDTH, ...
    'Color', LINE_COLOR);

% 坐标轴与字体
ax.FontName = FONT_NAME;
ax.FontSize = TICK_FONT_SIZE;
ax.FontWeight = TICK_FONT_WEIGHT;
ax.LineWidth = AXIS_LINE_WIDTH;
ax.TickDir = TICK_DIR;
box(ax, 'on');
grid(ax, 'off');

% 轴标签
xlabel(ax, 'M_T', 'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);
ylabel(ax, 'Responsivity', 'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);

% 适度留白
xlim(ax, [min(cj_thresholds), max(cj_thresholds)]);
ylim(ax, [0, 1]);
yticks(ax, 0:0.2:1);

hold(ax, 'off');

% -------------------- 导出图像 --------------------
exportgraphics(fig, output_path, 'ContentType', 'vector');
fprintf('Responsiveness figure saved to: %s\n', output_path);
if ~isnan(eta_value)
    fprintf('\n噪声强度 D_theta = %.1f, 对应 eta = %.1f\n', ...
        results.base_parameters.angleNoiseIntensity, eta_value);
end
