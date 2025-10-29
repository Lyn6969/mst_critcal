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
GRID_ON = 'on';             % 网格开关 ('on' 或 'off')
GRID_ALPHA = 0.1;           % 网格透明度 (0-1)
GRID_LINE_STYLE = '--';     % 网格线型 ('-' 实线, '--' 虚线, ':' 点线, '-.' 点划线)

% 图例设置
LEGEND_LOCATION = 'northeast';  % 图例位置
LEGEND_BOX = 'on';             % 图例边框 ('on' 或 'off')

%% 加载数据
mat_file = 'data.mat';
mat_path = 'data\experiments\delta_c_scan\N200_noise0.000_20251023_202713\';

% 获取项目根目录（用于转换为绝对路径）
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(fileparts(script_dir)));

% 转换为绝对路径
mat_path_abs = fullfile(project_root, mat_path);
data = load(fullfile(mat_path_abs, mat_file), 'results');
results = data.results;

% 提取数据（转为行向量）
thresholds = results.cj_thresholds(:)';
c1_mean = results.c1_mean(:)';
c2_mean = results.c2_mean(:)';
c1_sem = results.c1_sem(:)';
c2_sem = results.c2_sem(:)';
delta_c = results.delta_c(:)';

%% 设置输出路径
data_root = fullfile(project_root, 'data');
results_root = fullfile(project_root, 'results');

if startsWith(mat_path_abs, data_root)
    relative_path = mat_path_abs(length(data_root)+1:end);
    if startsWith(relative_path, filesep)
        relative_path = relative_path(2:end);
    end
    output_dir = fullfile(results_root, relative_path);
else
    output_dir = results_root;
end

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

[~, mat_name, ~] = fileparts(fullfile(mat_path_abs, mat_file));
cascade_pdf = fullfile(output_dir, sprintf('%s_cascade.pdf', mat_name));
delta_pdf = fullfile(output_dir, sprintf('%s_delta.pdf', mat_name));

%% 图1：级联规模对比
fig1 = figure('Position', [100, 100, FIG_WIDTH, FIG_HEIGHT], 'Color', 'white');

ax1 = axes('Parent', fig1);
hold(ax1, 'on');

% c1-c2 之间的填充区域（灰色阴影）
fill([thresholds fliplr(thresholds)], [c1_mean fliplr(c2_mean)], FILL_COLOR, ...
    'FaceAlpha', FILL_ALPHA, 'EdgeColor', 'none', 'HandleVisibility', 'off');

% c1 曲线（红色）
plot(ax1, thresholds, c1_mean, 'Color', [1 0 0], 'LineWidth', LINE_WIDTH, 'DisplayName', 'c_1');

% c2 曲线（蓝色）
plot(ax1, thresholds, c2_mean, 'Color', [0 0 1], 'LineWidth', LINE_WIDTH, 'DisplayName', 'c_2');

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

exportgraphics(fig1, cascade_pdf, 'ContentType', 'vector');
fprintf('级联规模图已保存至: %s\n', cascade_pdf);

%% 图2：Delta C 分析
[max_delta_c, max_idx] = max(delta_c);
optimal_cj = thresholds(max_idx);

fig2 = figure('Position', [100, 100, FIG_WIDTH, FIG_HEIGHT], 'Color', 'white');

ax2 = axes('Parent', fig2);
hold(ax2, 'on');

% Delta C 曲线
plot(ax2, thresholds, delta_c, 'Color', [45 45 45]/255, 'LineWidth', LINE_WIDTH);
plot(ax2, optimal_cj, max_delta_c, 'o', 'MarkerSize', 7, ...
    'MarkerFaceColor', [228 26 28]/255, 'MarkerEdgeColor', 'k', 'LineWidth', MARKER_LINE_WIDTH);

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
