% plot_delta_c_scan_results - Delta C 参数扫描结果可视化
%
% 功能：从实验数据生成级联规模对比图和 Delta C 分析图
% 输出：两个 PDF 文件（cascade 和 delta）

clear; clc; close all;

%% 配置
% 图片尺寸
FIG_WIDTH = 400;
FIG_HEIGHT = 200;

% 线条粗细
LINE_WIDTH = 2.5;              % 数据曲线线宽
AXIS_LINE_WIDTH = 1.5;         % 坐标轴框线线宽
MARKER_LINE_WIDTH = 1.2;     % 峰值标记线宽

% 文字样式
FONT_NAME = 'Arial';        % 字体名称（全局）
LABEL_FONT_SIZE = 12;           % 坐标轴标签字体大小（xlabel, ylabel）
LABEL_FONT_WEIGHT = 'Bold';     % 坐标轴标签字体粗细 ('normal' 或 'bold')
TICK_FONT_SIZE = 13;            % 坐标轴刻度字体大小
TICK_FONT_WEIGHT = 'Bold';    % 坐标轴刻度字体粗细 ('normal' 或 'bold')
LEGEND_FONT_SIZE = 15;          % 图例字体大小

% 坐标轴刻度设置
TICK_DIR = 'in';                % 刻度方向 ('in' 或 'out')

% c1-c2 填充区域设置
FILL_ALPHA = 0.8;               % c1-c2 之间填充区域的透明度 (0-1)
FILL_COLOR = [0.5 0.5 0.5];     % 填充颜色（灰色）

% 网格设置
GRID_ON = 'off';             % 网格开关 ('on' 或 'off')
GRID_ALPHA = 0.1;           % 网格透明度 (0-1)
GRID_LINE_STYLE = '--';     % 网格线型 ('-' 实线, '--' 虚线, ':' 点线, '-.' 点划线)

% 图例设置
LEGEND_LOCATION = 'northeast';  % 图例位置
LEGEND_BOX = 'on';             % 图例边框 ('on' 或 'off')

%% 加载数据
mat_file = 'data.mat';
mat_path = 'mst_critcal\data\experiments\delta_c_m_vs_1_scan\20251030_204718\';

% 获取项目根目录（用于转换为绝对路径）
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(fileparts(script_dir)));

% 转换为绝对路径
mat_path_abs = fullfile(project_root, mat_path);
data = load(fullfile(mat_path_abs, mat_file), 'results');
results = data.results;

% 提取数据（转为行向量）
thresholds = results.cj_thresholds(:)';

% 级联规模
if isfield(results, 'c_mean')
    c_mean = results.c_mean;
else
    error('结果中缺少 c_mean 字段，无法绘图。');
end

% 1 vs m 数据（c1 与 c2）
if size(c_mean, 2) < 2
    error('c_mean 列数少于 2，无法生成 c1/c2 对比。');
end
c1_mean = c_mean(:, 1)';
c2_mean = c_mean(:, 2)';

% Δc = c2 - c1
delta_c = c2_mean - c1_mean;

%% 设置输出路径
pic_dir = fullfile(project_root, 'mst_critcal', 'pic');
if ~exist(pic_dir, 'dir')
    mkdir(pic_dir);
end

[~, mat_name, ~] = fileparts(fullfile(mat_path_abs, mat_file));
cascade_pdf = fullfile(pic_dir, sprintf('%s_cascade.pdf', mat_name));
delta_pdf = fullfile(pic_dir, sprintf('%s_delta.pdf', mat_name));

%% 先计算最优阈值（Δc 的峰值位置）
[max_delta_c, max_idx] = max(delta_c);
optimal_cj = thresholds(max_idx);

%% 图1：级联规模对比
fig1 = figure('Position', [100, 100, FIG_WIDTH, FIG_HEIGHT], 'Color', 'white');

ax1 = axes('Parent', fig1);
hold(ax1, 'on');

% c1-c2 之间的填充区域（灰色阴影）
fill([thresholds fliplr(thresholds)], [c1_mean fliplr(c2_mean)], FILL_COLOR, ...
    'FaceAlpha', FILL_ALPHA, 'EdgeColor', 'none', 'HandleVisibility', 'off');

% c1 曲线（蓝色实线）
plot(ax1, thresholds, c1_mean, 'Color', [0 0 1], 'LineWidth', LINE_WIDTH, 'DisplayName', 'c_1');

% c2 曲线（蓝色虚线）
plot(ax1, thresholds, c2_mean, '--', 'Color', [0 0 1], 'LineWidth', LINE_WIDTH, 'DisplayName', 'c_2');

% 先设置坐标轴属性（包括刻度字体粗细）
ax1.FontName = FONT_NAME;
ax1.FontSize = TICK_FONT_SIZE;
ax1.FontWeight = TICK_FONT_WEIGHT;
ax1.LineWidth = AXIS_LINE_WIDTH;
grid(ax1, GRID_ON);
ax1.GridAlpha = GRID_ALPHA;
ax1.GridLineStyle = GRID_LINE_STYLE;
box(ax1, 'on');
ax1.TickDir = TICK_DIR;

% 后设置标签（会覆盖之前的 FontWeight 设置）
xlabel(ax1, 'M_T', 'Interpreter', 'tex', 'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);
ylabel(ax1, 'Avg. Cascade Size', 'Interpreter', 'tex', 'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);
legend(ax1, 'Location', LEGEND_LOCATION, 'Box', LEGEND_BOX, 'FontSize', LEGEND_FONT_SIZE);

% Δc 峰值位置红色虚线（与图2保持一致，不显示在图例中）
xline(ax1, optimal_cj, '--', 'Color', [0.85 0 0], 'LineWidth', 2, 'HandleVisibility', 'off');

exportgraphics(fig1, cascade_pdf, 'ContentType', 'vector');
fprintf('级联规模图已保存至: %s\n', cascade_pdf);

%% 图2：Delta C 分析

fig2 = figure('Position', [100, 100, FIG_WIDTH, FIG_HEIGHT], 'Color', 'white');

ax2 = axes('Parent', fig2);
hold(ax2, 'on');

% Delta C 曲线
delta_color = [38 94 180] / 255;
plot(ax2, thresholds, delta_c, 'Color', delta_color, 'LineWidth', LINE_WIDTH);
% 峰值位置的红色虚线（不显示在图例中）
xline(ax2, optimal_cj, '--', 'Color', [0.85 0 0], 'LineWidth', 2, 'HandleVisibility', 'off');

% 先设置坐标轴属性（包括刻度字体粗细）
ax2.FontName = FONT_NAME;
ax2.FontSize = TICK_FONT_SIZE;
ax2.FontWeight = TICK_FONT_WEIGHT;
ax2.LineWidth = AXIS_LINE_WIDTH;
grid(ax2, GRID_ON);
ax2.GridAlpha = GRID_ALPHA;
ax2.GridLineStyle = GRID_LINE_STYLE;
box(ax2, 'on');
ax2.TickDir = TICK_DIR;

% 后设置标签（会覆盖之前的 FontWeight 设置）
xlabel(ax2, 'M_T', 'Interpreter', 'tex', 'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);
ylabel(ax2, '\Delta c = c_2 - c_1', 'Interpreter', 'tex', 'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);

exportgraphics(fig2, delta_pdf, 'ContentType', 'vector');
fprintf('Δc 图已保存至: %s\n', delta_pdf);
