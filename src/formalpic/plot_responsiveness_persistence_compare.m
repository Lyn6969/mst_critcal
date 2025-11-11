% plot_responsiveness_persistence_compare - 响应性/持久性双轴对比图

clear; clc; close all;

%% -------------------- 样式配置 --------------------
FIG_WIDTH = 500;
FIG_HEIGHT = 280;
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
resp_file = 'responsiveness_cj_scan_20251105_211706.mat';

pers_dir = fullfile('results', 'persistence');
pers_file = 'persistence_cj_scan_20251105_214654.mat';

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
cj_resp = resp_results.cj_thresholds(:);
R_mean = resp_results.R_mean(:);
R_sem = resp_results.R_sem(:);

cj_pers = pers_results.cj_thresholds(:);
P_mean = pers_results.P_mean(:);
P_sem = pers_results.P_sem(:);

if numel(cj_resp) ~= numel(cj_pers) || any(abs(cj_resp - cj_pers) > 1e-9)
    error('响应性与持久性阈值采样不一致。');
end

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

upper_R = R_mean + R_sem;
lower_R = max(R_mean - R_sem, 0);
upper_P = min(P_norm + P_norm_sem, 1);
lower_P = max(P_norm - P_norm_sem, 0);

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
output_path = fullfile(pic_dir, 'responsiveness_persistence_dual_axis.pdf');

%% -------------------- 绘图 --------------------
fig = figure('Position', [180, 160, FIG_WIDTH, FIG_HEIGHT], 'Color', 'white');
ax = axes('Parent', fig);
hold(ax, 'on');

yyaxis left;
fill([cj_resp; flip(cj_resp)], [upper_R; flip(lower_R)], R_COLOR, ...
    'FaceAlpha', SHADE_ALPHA, 'EdgeColor', 'none');
plot(cj_resp, R_mean, '-', 'Color', R_COLOR, 'LineWidth', LINE_WIDTH);
ylabel('Responsivity', 'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);
ax.YColor = R_COLOR;

yyaxis right;
fill([cj_resp; flip(cj_resp)], [upper_P; flip(lower_P)], P_COLOR, ...
    'FaceAlpha', SHADE_ALPHA, 'EdgeColor', 'none');
plot(cj_resp, P_norm, '-', 'Color', P_COLOR, 'LineWidth', LINE_WIDTH);
ylabel('Normalized Persistence', 'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);
ylim([0, 1.05]);
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

eta_label = NaN;
if ~isnan(eta_resp)
    eta_label = eta_resp;
elseif ~isnan(eta_pers)
    eta_label = eta_pers;
end
if ~isnan(eta_label)
    text(ax, 'Units', 'normalized', 'Position', [0.97, 0.08], ...
        'String', sprintf('\\eta = %.1f', eta_label), ...
        'HorizontalAlignment', 'right', 'FontName', FONT_NAME, ...
        'FontSize', LABEL_FONT_SIZE-1, 'FontWeight', LABEL_FONT_WEIGHT);
end

hold(ax, 'off');

%% -------------------- 导出 --------------------
exportgraphics(fig, output_path, 'ContentType', 'vector');
fprintf('Dual-axis figure saved to: %s\n', output_path);
if ~isnan(eta_resp)
    fprintf('响应性噪声: D_theta = %.1f, eta = %.1f\n', resp_results.base_parameters.angleNoiseIntensity, eta_resp);
end
if ~isnan(eta_pers)
    fprintf('持久性噪声: D_theta = %.1f, eta = %.1f\n', pers_results.parameters.angleNoiseIntensity, eta_pers);
end
