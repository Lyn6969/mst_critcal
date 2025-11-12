% plot_tradeoff_fscore_contours - Fβ 等值线背景 + R–P 散点 + 自适应点
%
% 功能：
%   - 在 R–P 平面绘制 Fβ 等值线（背景热区），叠加固定阈值扫描得到的 (R_mean, P_mean_norm)
%     散点及其连线，并以五角星高亮“自适应阈值”点，直观展示其优势。
%
% 使用说明：
%   1) 手动指定输入文件名（仅文件名或相对/绝对路径均可）：
%        mat_filename = 'cj_tradeoff_adaptive_shared_seed_20251111_005228.mat';
%   2) 运行本脚本，图像将保存到 `<项目根>/pic/` 目录。
%
% 路径规范（formalpic 必须遵守）：
%   - 仅向上两级获取项目根目录：
%       script_dir  = fileparts(mfilename('fullpath'));
%       project_root = fileparts(fileparts(script_dir));
%
% 风格规范：
%   - 字体、线宽、坐标轴风格参考 src/formalpic/plot_delta_c_scan_results.m
%   - 导出矢量 PDF：exportgraphics(..., 'ContentType', 'vector')
%
% 作者：系统生成（依据仓库规范）
% 版本：MATLAB R2025a

clear; clc; close all;

%% -------------------- 基本配置 --------------------
% 手动指定输入文件名（你可以直接修改为目标文件名或绝对路径）
mat_filename = 'cj_tradeoff_adaptive_shared_seed_20251105_221310.mat';  % TODO: 按需修改

% Fβ 权衡参数（β=1 为对称偏好；β>1 偏好“持久性 P”，β<1 偏好“响应性 R”）
beta = 1.0;

% 画面风格（参考 formalpic 规范）
FIG_WIDTH = 500;           % 图片宽度（像素）
FIG_HEIGHT = 500;          % 图片高度（像素）
LINE_WIDTH = 2.4;          % 曲线线宽
AXIS_LINE_WIDTH = 1.5;     % 坐标轴框线线宽
MARKER_SIZE_FIXED = 42;    % 固定阈值散点大小
MARKER_SIZE_ADAPT = 140;   % 自适应五角星大小
TICK_DIR = 'in';           % 刻度朝向
FONT_NAME = 'Arial';       % 字体
LABEL_FONT_SIZE = 12;      % 坐标轴标签字号
LABEL_FONT_WEIGHT = 'Bold';% 坐标轴标签粗细
TICK_FONT_SIZE = 13;       % 刻度字号
TICK_FONT_WEIGHT = 'Bold'; % 刻度粗细
LEGEND_LOCATION = 'southeast';

% 颜色配置
FIXED_COLOR = [0.35 0.35 0.35];      % 固定阈值散点/连线颜色（灰）
TRACK_COLOR = [0.45 0.45 0.45];      % 固定阈值轨迹线颜色（浅灰）
ADAPT_COLOR = [0.85 0.20 0.20];      % 自适应五角星颜色（红）
ADAPT_EDGE = [0.50 0.00 0.00];       % 自适应五角星边框
CONTOUR_LINE = [0.15 0.15 0.15];     % 等值线颜色（可选）

%% -------------------- 路径与数据加载 --------------------
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(script_dir));  % 仅向上两级（严禁越级）

% 兼容：支持绝对/相对路径。
% 若仅提供文件名，优先在 <项目根>/src/sim/experiments/results/ 下寻找，
% 若不存在再回退到 <项目根>/results/tradeoff/（兼容旧目录结构）。
if isfile(mat_filename)
    input_path = mat_filename;
else
    candidate1 = fullfile(project_root, 'src', 'sim', 'experiments', 'results', mat_filename);
    candidate2 = fullfile(project_root, 'results', 'tradeoff', mat_filename);
    if isfile(candidate1)
        input_path = candidate1;
    elseif isfile(candidate2)
        input_path = candidate2;
    else
        error('未找到输入文件：%s 或 %s', candidate1, candidate2);
    end
end

S = load(input_path, 'summary');
if ~isfield(S, 'summary')
    error('文件中未找到变量 summary：%s', input_path);
end
summary = S.summary;

if ~isfield(summary, 'results') || ~isfield(summary.results, 'fixed') || ~isfield(summary.results, 'adaptive')
    error('summary 结构不完整，缺少 results.fixed 或 results.adaptive。');
end
fixed = summary.results.fixed;
adapt = summary.results.adaptive;

% 取固定阈值点：R 均值与 P 归一化均值（剔除 NaN）
R_fixed = fixed.R_mean(:);
P_fixed = [];
if isfield(fixed, 'P_mean_norm')
    P_fixed = fixed.P_mean_norm(:);
elseif isfield(fixed, 'P_mean') && isfield(summary, 'P_range') && summary.P_range > 0
    P_fixed = (fixed.P_mean(:) - summary.P_min) / summary.P_range;
else
    error('未找到归一化持久性：results.fixed.P_mean_norm 或（P_mean + summary.P_min/P_range）。');
end
valid = ~(isnan(R_fixed) | isnan(P_fixed));
R_fixed = R_fixed(valid);
P_fixed = P_fixed(valid);
thr_fixed = [];
if isfield(fixed, 'cj_thresholds')
    thr_fixed = fixed.cj_thresholds(:);
    thr_fixed = thr_fixed(valid);
end

% 取自适应点（单点）
if ~isscalar(adapt.R_mean) || ~isscalar(adapt.P_mean) && ~isfield(adapt, 'P_mean_norm')
    % 理论上自适应模式只有一个“参数点”，若存在多值，这里取均值
end
R_adapt = adapt.R_mean(1);
if isfield(adapt, 'P_mean_norm')
    P_adapt = adapt.P_mean_norm(1);
elseif isfield(adapt, 'P_mean') && isfield(summary, 'P_range') && summary.P_range > 0
    P_adapt = (adapt.P_mean(1) - summary.P_min) / summary.P_range;
else
    error('未找到自适应的归一化持久性：results.adaptive.P_mean_norm 或（P_mean + summary.P_min/P_range）。');
end

% 误差（可选显示）
R_sem_adapt = NaN; P_sem_adapt = NaN;
if isfield(adapt, 'R_sem'), R_sem_adapt = adapt.R_sem(1); end
if isfield(adapt, 'P_sem_norm')
    P_sem_adapt = adapt.P_sem_norm(1);
elseif isfield(adapt, 'P_sem') && isfield(summary, 'P_range') && summary.P_range > 0
    P_sem_adapt = adapt.P_sem(1) / summary.P_range;
end

%% -------------------- 背景：Fβ 等值线 --------------------
% Fβ 定义（避免除零）
fbeta = @(R, P, b) (1 + b.^2) .* (R .* P) ./ max(b.^2 .* R + P, eps);

% 网格
grid_n = 201;
r_vec = linspace(0, 1, grid_n);
p_vec = linspace(0, 1, grid_n);
[RR, PP] = meshgrid(r_vec, p_vec);
FF = fbeta(RR, PP, beta);

%% -------------------- 绘图 --------------------
fig = figure('Position', [100 100 FIG_WIDTH FIG_HEIGHT], 'Color', 'white');
ax = axes('Parent', fig); hold(ax, 'on');

% 等值颜色背景
contourf(ax, r_vec, p_vec, FF, 16, 'LineStyle', 'none');
colormap(ax, parula);
caxis([0 1]);   % Fβ ∈ [0,1]
cb = colorbar(ax);
cb.Label.String = sprintf('F_\\beta (\\beta = %.2f)', beta);

% 可选：叠加等值线以增强层次
contour(ax, r_vec, p_vec, FF, 0.6:0.05:0.95, 'LineColor', CONTOUR_LINE, 'LineWidth', 0.6, 'LineStyle', '-');

% 固定阈值轨迹（按阈值升序连接）
if ~isempty(thr_fixed) && numel(thr_fixed) == numel(R_fixed)
    [~, ord] = sort(thr_fixed, 'ascend');
else
    ord = 1:numel(R_fixed);
end
plot(ax, R_fixed(ord), P_fixed(ord), '-', 'Color', TRACK_COLOR, 'LineWidth', 1.2, ...
    'DisplayName', 'Fixed Threshold Track');

% 固定阈值散点
scatter(ax, R_fixed, P_fixed, MARKER_SIZE_FIXED, FIXED_COLOR, 'filled', ...
    'DisplayName', 'Fixed Threshold');

% 自适应点（五角星）与误差十字
plot(ax, R_adapt, P_adapt, 'p', 'MarkerSize', sqrt(MARKER_SIZE_ADAPT), ...
    'MarkerFaceColor', ADAPT_COLOR, 'MarkerEdgeColor', ADAPT_EDGE, 'LineWidth', 1.5, ...
    'DisplayName', 'Adaptive Threshold');
if ~isnan(R_sem_adapt) && ~isnan(P_sem_adapt) && R_sem_adapt>=0 && P_sem_adapt>=0
    line(ax, [R_adapt - R_sem_adapt, R_adapt + R_sem_adapt], [P_adapt, P_adapt], ...
        'Color', ADAPT_EDGE, 'LineWidth', 1.2);
    line(ax, [R_adapt, R_adapt], [P_adapt - P_sem_adapt, P_adapt + P_sem_adapt], ...
        'Color', ADAPT_EDGE, 'LineWidth', 1.2);
end

% 坐标轴与注释（中文）
ax.FontName = FONT_NAME;
ax.FontSize = TICK_FONT_SIZE;
ax.FontWeight = TICK_FONT_WEIGHT;
ax.LineWidth = AXIS_LINE_WIDTH;
ax.TickDir = TICK_DIR;
box(ax, 'on');
grid(ax, 'off');
xlim(ax, [0 1.05]);
ylim(ax, [0 1.05]);
xticks(ax, 0:0.2:1);
yticks(ax, 0:0.2:1);
xlabel(ax, 'Responsiveness R', 'Interpreter', 'tex', 'FontName', FONT_NAME, ...
    'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);
ylabel(ax, 'Normalized Persistence P_{norm}', 'Interpreter', 'tex', 'FontName', FONT_NAME, ...
    'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);
legend(ax, 'Location', LEGEND_LOCATION, 'Box', 'on');
title(ax, 'R–P Trade-off on F_{\beta} Contours', 'FontWeight', 'bold');

%% -------------------- 指标量化与角标 --------------------
% 计算自适应与固定阈值的 Fβ 指标与占优比例
F_adapt = fbeta(R_adapt, P_adapt, beta);
F_fixed = fbeta(R_fixed, P_fixed, beta);
[F_fixed_best, idx_best] = max(F_fixed, [], 'omitnan');

dominated = (R_adapt >= R_fixed) & (P_adapt >= P_fixed) & ...
            ((R_adapt > R_fixed) | (P_adapt > P_fixed));
dom_ratio = sum(dominated) / max(numel(R_fixed), 1);

%% -------------------- 导出 --------------------
pic_dir = fullfile(project_root, 'pic');
if ~exist(pic_dir, 'dir'); mkdir(pic_dir); end
beta_tag = strrep(sprintf('%.2f', beta), '.', 'p');
ts = '';
if isfield(summary, 'timestamp')
    ts = ['_' char(summary.timestamp)];
end
output_pdf = fullfile(pic_dir, sprintf('tradeoff_fscore_contours_beta_%s%s.pdf', beta_tag, ts));
exportgraphics(fig, output_pdf, 'ContentType', 'vector');

fprintf('图像已导出：%s\n', output_pdf);
fprintf('输入：%s\n', input_path);
