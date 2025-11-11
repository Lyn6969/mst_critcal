% plot_responsiveness_persistence_compare - 响应性/持久性双轴对比图
%
% 功能概述：
%   专业的MATLAB可视化工具，用于绘制响应性（Responsivity）和持久性（Persistence）的
%   双轴对比图，验证群集运动模型中的持久性-响应性权衡关系。
%
% 输入数据：
%   - 响应性数据：results/responsiveness/ 目录下的 .mat 文件
%     * 包含：cj_thresholds, R_mean, R_sem, base_parameters.angleNoiseIntensity
%   - 持久性数据：results/persistence/ 目录下的 .mat 文件
%     * 包含：cj_thresholds, P_mean, P_sem, parameters.angleNoiseIntensity
%
% 输出结果：
%   - 高质量PDF双轴对比图：pic/responsiveness_persistence_dual_axis_eta_xxx.pdf
%   - 左轴：响应性曲线（红色）+ 误差填充
%   - 右轴：归一化持久性曲线（蓝色）+ 误差填充
%   - 图上显示噪声参数标签：η = √(2D_θ)

% 注意事项：
%   - 确保两个数据文件的阈值采样完全一致
%   - 文件名中的时间戳需要根据实际实验结果调整
%   - 需要MATLAB R2020a+版本支持exportgraphics函数
%
% 作者：Lyn6969
% 日期：2025年

clear; clc; close all;

%% -------------------- 样式配置 --------------------
FIG_WIDTH = 400;
FIG_HEIGHT = 200;
R_COLOR = [0.85 0.1 0.1];      % 响应性颜色
P_COLOR = [38, 94, 180] / 255;  % 持久性颜色
SHADE_ALPHA = 0.25;
LINE_WIDTH = 2.4;
AXIS_LINE_WIDTH = 1.5;
FONT_NAME = 'Arial';
LABEL_FONT_SIZE = 12;
LABEL_FONT_WEIGHT = 'Bold';
TICK_FONT_SIZE = 13;
TICK_FONT_WEIGHT = 'Bold';
TICK_DIR = 'in';

%% -------------------- 数据路径 --------------------
% TODO: 修改为实际文件名
resp_dir = fullfile('results', 'responsiveness');
resp_file = 'responsiveness_cj_scan_20251111_173429_eta_0p100.mat';

pers_dir = fullfile('results', 'persistence');
pers_file = 'persistence_cj_scan_20251111_203019_eta_0p100.mat';

script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(script_dir));

resp_path = fullfile(project_root, resp_dir, resp_file);
pers_path = fullfile(project_root, pers_dir, pers_file);

if ~isfile(resp_path)
    error('未找到响应性结果：%s', resp_path);
end
if ~isfile(pers_path)
    error('未找到持久性结果：%s', pers_path);
end

resp_data = load(resp_path, 'results');
pers_data = load(pers_path, 'results');
resp_results = resp_data.results;
pers_results = pers_data.results;

%% -------------------- 数据整理 --------------------
% 提取响应性数据：阈值、均值和标准误
cj_resp = resp_results.cj_thresholds(:);
R_mean = resp_results.R_mean(:);
R_sem = resp_results.R_sem(:);

cj_pers = pers_results.cj_thresholds(:);
P_mean = pers_results.P_mean(:);
P_sem = pers_results.P_sem(:);

% 验证阈值采样一致性，确保数据可对比
if numel(cj_resp) ~= numel(cj_pers) || any(abs(cj_resp - cj_pers) > 1e-9)
    error('响应性与持久性阈值采样不一致。');
end

% 持久性数据归一化到[0,1]范围，便于与响应性对比
valid_idx = ~isnan(P_mean);
if numel(unique(P_mean(valid_idx))) <= 1
    warning('持久性数据变化不足，归一化退化。');
    P_norm = zeros(size(P_mean));
    P_norm_sem = zeros(size(P_sem));
else
    P_min = min(P_mean(valid_idx));
    P_max = max(P_mean(valid_idx));
    P_norm = (P_mean - P_min) / (P_max - P_min);
    P_norm_sem = P_sem / (P_max - P_min);
end

% 计算误差范围：响应性和持久性的均值±标准误
upper_R = R_mean + R_sem;
lower_R = max(R_mean - R_sem, 0);
upper_P = min(P_norm + P_norm_sem, 1);
lower_P = max(P_norm - P_norm_sem, 0);

% 提取噪声参数：计算真实噪声强度 η = √(2D_θ)
eta_resp = NaN;
eta_pers = NaN;
if isfield(resp_results, 'base_parameters') && isfield(resp_results.base_parameters, 'angleNoiseIntensity')
    eta_resp = sqrt(2 * resp_results.base_parameters.angleNoiseIntensity);
end
if isfield(pers_results, 'parameters') && isfield(pers_results.parameters, 'angleNoiseIntensity')
    eta_pers = sqrt(2 * pers_results.parameters.angleNoiseIntensity);
end

%% -------------------- 输出目录 --------------------
pic_dir = fullfile(project_root, 'pic');
if ~exist(pic_dir, 'dir'); mkdir(pic_dir); end

% 生成动态文件名：包含噪声标签以便区分不同参数下的结果
eta_value = NaN;
if ~isnan(eta_resp)
    eta_value = eta_resp;
elseif ~isnan(eta_pers)
    eta_value = eta_pers;
end

if ~isnan(eta_value)
    eta_tag = strrep(sprintf('eta_%0.3f', eta_value), '.', 'p');
    filename = sprintf('responsiveness_persistence_dual_axis_%s.pdf', eta_tag);
else
    filename = 'responsiveness_persistence_dual_axis.pdf';
end

output_path = fullfile(pic_dir, filename);

%% -------------------- 绘图 --------------------
fig = figure('Position', [180, 160, FIG_WIDTH, FIG_HEIGHT], 'Color', 'white');
ax = axes('Parent', fig);
hold(ax, 'on');

% 左轴：绘制响应性曲线和误差填充
yyaxis left;
fill([cj_resp; flip(cj_resp)], [upper_R; flip(lower_R)], R_COLOR, ...
    'FaceAlpha', SHADE_ALPHA, 'EdgeColor', 'none');
plot(cj_resp, R_mean, '-', 'Color', R_COLOR, 'LineWidth', LINE_WIDTH);
ylabel('Responsivity', 'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);
ax.YColor = R_COLOR;
ylim([0, 1.05]);
yticks(ax, 0:0.2:1);
% 右轴：绘制归一化持久性曲线和误差填充
yyaxis right;
fill([cj_resp; flip(cj_resp)], [upper_P; flip(lower_P)], P_COLOR, ...
    'FaceAlpha', SHADE_ALPHA, 'EdgeColor', 'none');
plot(cj_resp, P_norm, '-', 'Color', P_COLOR, 'LineWidth', LINE_WIDTH);
ylabel('Norm. Persistence', 'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);
ylim([0, 1.05]);
yticks(ax, 0:0.2:1);
ax.YColor = P_COLOR;

yyaxis left;
ax.FontName = FONT_NAME;
ax.FontSize = TICK_FONT_SIZE;
ax.FontWeight = TICK_FONT_WEIGHT;
ax.LineWidth = AXIS_LINE_WIDTH;
ax.TickDir = TICK_DIR;
box(ax, 'on');
grid(ax, 'off');
xlabel('M_T', 'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);
xlim([min(cj_resp), max(cj_resp)]);


% 在图上添加噪声参数标签，显示当前实验条件
hold(ax, 'off');

%% -------------------- 导出 --------------------
% 导出高质量PDF并显示实验参数信息
exportgraphics(fig, output_path, 'ContentType', 'vector');
fprintf('Dual-axis figure saved to: %s\n', output_path);
if ~isnan(eta_resp)
    fprintf('响应性噪声: D_theta = %.1f, eta = %.1f\n', resp_results.base_parameters.angleNoiseIntensity, eta_resp);
end
if ~isnan(eta_pers)
    fprintf('持久性噪声: D_theta = %.1f, eta = %.1f\n', pers_results.parameters.angleNoiseIntensity, eta_pers);
end
