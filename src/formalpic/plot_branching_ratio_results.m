% plot_branching_ratio_results - 分支比扫描结果可视化脚本
%
% 功能说明：
%   从 run_branching_ratio_scan_parallel 生成的结果文件中读取平均分支比，
%   绘制随 M_T 变化的折线图，并在分支比为 1 的位置绘制参考虚线。
%   图片输出到项目根目录同级的 pic 目录，格式参考 plot_delta_c_scan_results。

clear; clc; close all;

%% -------------------- 图像样式配置 --------------------
FIG_WIDTH = 400;
FIG_HEIGHT = 200;
LINE_WIDTH = 2.5;
AXIS_LINE_WIDTH = 1.5;

FONT_NAME = 'Arial';
LABEL_FONT_SIZE = 12;
LABEL_FONT_WEIGHT = 'Bold';
TICK_FONT_SIZE = 13;
TICK_FONT_WEIGHT = 'Bold';
LEGEND_FONT_SIZE = 12;

GRID_ON = 'off';
GRID_ALPHA = 0.1;
GRID_LINE_STYLE = '--';
TICK_DIR = 'in';

CASCADE_LINE_COLOR = [38, 94, 180] / 255;
THRESHOLD_LINE_COLOR = [0.95, 0.75, 0.1];

%% -------------------- 数据路径配置 --------------------
% TODO: 按实际结果路径修改以下目录与文件名
mat_file = 'data.mat';
mat_dir = fullfile('mst_critcal', 'data', 'experiments', 'branching_ratio_scan', ...
    '20251016_231610');  % 示例目录，请根据实际时间戳修改

script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(fileparts(script_dir)));

mat_path_abs = fullfile(project_root, mat_dir, mat_file);
if ~isfile(mat_path_abs)
    error('未找到结果文件，请检查路径: %s', mat_path_abs);
end

data = load(mat_path_abs, 'results');
results = data.results;

%% -------------------- 数据提取与整理 --------------------
thresholds = results.cj_thresholds(:);
b_mean = results.b_mean(:);

if numel(thresholds) ~= numel(b_mean)
    error('阈值数量与平均分支比数量不一致。');
end

%% -------------------- 图像输出目录 --------------------
pic_dir = fullfile(project_root, 'pic');
if ~exist(pic_dir, 'dir')
    mkdir(pic_dir);
end

output_name = sprintf('branching_ratio_mean_N%d.pdf', results.parameters.N);
output_path = fullfile(pic_dir, output_name);

%% -------------------- 绘制平均分支比折线图 --------------------
fig = figure('Position', [120, 120, FIG_WIDTH, FIG_HEIGHT], 'Color', 'white');
ax = axes('Parent', fig);
hold(ax, 'on');

plot(ax, thresholds, b_mean, 'LineWidth', LINE_WIDTH, 'Color', CASCADE_LINE_COLOR);
yline(ax, 1.0, '--', 'Color', THRESHOLD_LINE_COLOR, 'LineWidth', 2.0, 'HandleVisibility', 'off');

ax.FontName = FONT_NAME;
ax.FontSize = TICK_FONT_SIZE;
ax.FontWeight = TICK_FONT_WEIGHT;
ax.LineWidth = AXIS_LINE_WIDTH;
ax.TickDir = TICK_DIR;
grid(ax, GRID_ON);
ax.GridAlpha = GRID_ALPHA;
ax.GridLineStyle = GRID_LINE_STYLE;
box(ax, 'on');

xlabel(ax, 'M_T', 'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);
ylabel(ax, 'Mean Branching Ratio', 'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);

ylim_upper = max(max(b_mean) * 1.1, 1.1);
ylim(ax, [0, ylim_upper]);

hold(ax, 'off');

exportgraphics(fig, output_path, 'ContentType', 'vector');
fprintf('Branching ratio figure saved to: %s\n', output_path);

