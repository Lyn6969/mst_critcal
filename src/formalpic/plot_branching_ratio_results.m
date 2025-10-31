% plot_branching_ratio_results - 分支比扫描结果可视化脚本
%
% 功能说明：
%   从 run_branching_ratio_scan_parallel 生成的结果文件中读取平均分支比，
%   绘制随 M_T 变化的折线图，并在分支比为 1 的位置绘制参考虚线。
%   图片输出到项目根目录同级的 pic 目录，格式参考 plot_delta_c_scan_results。

% 清理工作空间、命令窗口和关闭所有图形窗口
clear; clc; close all;

%% -------------------- 图像样式配置 --------------------
% 图片尺寸设置（单位：像素）
FIG_WIDTH = 400;      % 图片宽度
FIG_HEIGHT = 200;     % 图片高度

% 线条粗细设置
LINE_WIDTH = 2.5;        % 数据曲线线宽
AXIS_LINE_WIDTH = 1.5;   % 坐标轴框线线宽

% 字体样式设置
FONT_NAME = 'Arial';           % 字体名称（全局）
LABEL_FONT_SIZE = 12;           % 坐标轴标签字体大小（xlabel, ylabel）
LABEL_FONT_WEIGHT = 'Bold';     % 坐标轴标签字体粗细 ('normal' 或 'bold')
TICK_FONT_SIZE = 13;            % 坐标轴刻度字体大小
TICK_FONT_WEIGHT = 'Bold';      % 坐标轴刻度字体粗细 ('normal' 或 'bold')
LEGEND_FONT_SIZE = 12;          % 图例字体大小

% 网格和刻度设置
GRID_ON = 'off';             % 网格开关 ('on' 或 'off')
GRID_ALPHA = 0.1;           % 网格透明度 (0-1)
GRID_LINE_STYLE = '--';     % 网格线型 ('-' 实线, '--' 虚线, ':' 点线, '-.' 点划线)
TICK_DIR = 'in';            % 刻度方向 ('in' 或 'out')

% 颜色设置（RGB值，范围0-1）
CASCADE_LINE_COLOR = [38, 94, 180] / 255;      % 级联曲线颜色（蓝色）
THRESHOLD_LINE_COLOR = [0.95, 0.75, 0.1];     % 阈值参考线颜色（黄色）

%% -------------------- 数据路径配置 --------------------
% TODO: 按实际结果路径修改以下目录与文件名
mat_file = 'data.mat';  % 数据文件名
mat_dir = fullfile('mst_critcal', 'data', 'experiments', 'branching_ratio_scan', ...
    '20251016_231610');  % 示例目录，请根据实际时间戳修改

% 获取脚本所在目录的绝对路径
script_dir = fileparts(mfilename('fullpath'));
% 获取项目根目录（向上三级目录）
project_root = fileparts(fileparts(fileparts(script_dir)));

% 构建数据文件的完整绝对路径
mat_path_abs = fullfile(project_root, mat_dir, mat_file);
% 检查文件是否存在
if ~isfile(mat_path_abs)
    error('未找到结果文件，请检查路径: %s', mat_path_abs);
end

% 加载数据文件中的results变量
data = load(mat_path_abs, 'results');
results = data.results;

%% -------------------- 数据提取与整理 --------------------
% 提取阈值数据并转为列向量
thresholds = results.cj_thresholds(:);
% 提取平均分支比数据并转为列向量
b_mean = results.b_mean(:);

% 检查数据维度是否一致
if numel(thresholds) ~= numel(b_mean)
    error('阈值数量与平均分支比数量不一致。');
end

%% -------------------- 图像输出目录 --------------------
% 设置图片输出目录
pic_dir = fullfile(project_root, 'pic');
% 如果目录不存在则创建
if ~exist(pic_dir, 'dir')
    mkdir(pic_dir);
end

% 生成输出文件名，包含粒子数量N参数
output_name = sprintf('branching_ratio_mean_N%d.pdf', results.parameters.N);
output_path = fullfile(pic_dir, output_name);

%% -------------------- 绘制平均分支比折线图 --------------------
% 创建图形窗口，设置位置、大小和背景色
fig = figure('Position', [120, 120, FIG_WIDTH, FIG_HEIGHT], 'Color', 'white');
% 创建坐标轴对象
ax = axes('Parent', fig);
% 保持当前图形，允许在同一坐标轴上绘制多条曲线
hold(ax, 'on');

% 绘制平均分支比曲线
plot(ax, thresholds, b_mean, 'LineWidth', LINE_WIDTH, 'Color', CASCADE_LINE_COLOR);
% 在y=1处绘制水平参考虚线（分支比临界值）
yline(ax, 1.0, '--', 'Color', THRESHOLD_LINE_COLOR, 'LineWidth', 2.0, 'HandleVisibility', 'off');

% 设置坐标轴字体属性
ax.FontName = FONT_NAME;
ax.FontSize = TICK_FONT_SIZE;
ax.FontWeight = TICK_FONT_WEIGHT;
ax.LineWidth = AXIS_LINE_WIDTH;
ax.TickDir = TICK_DIR;
% 设置网格属性
grid(ax, GRID_ON);
ax.GridAlpha = GRID_ALPHA;
ax.GridLineStyle = GRID_LINE_STYLE;
% 显示坐标轴边框
box(ax, 'on');

% 设置坐标轴标签
xlabel(ax, 'M_T', 'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);
ylabel(ax, 'Mean Branching Ratio', 'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);

% 设置y轴范围：下限为0，上限为最大值的1.1倍或1.1中的较大值
ylim_upper = max(max(b_mean) * 1.1, 1.1);
ylim(ax, [0, ylim_upper]);

% 释放坐标轴保持状态
hold(ax, 'off');

% 导出图形为PDF矢量格式
exportgraphics(fig, output_path, 'ContentType', 'vector');
% 输出保存路径信息
fprintf('Branching ratio figure saved to: %s\n', output_path);

