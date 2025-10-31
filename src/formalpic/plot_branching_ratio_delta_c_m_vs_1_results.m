% plot_branching_ratio_delta_c_m_vs_1_results - 从 Delta_c 扫描结果提取分支比并绘图
%
% 功能说明：
%   读取 run_delta_c_m_vs_1_scan_parallel.m 生成的数据文件，提取在指定初发个体数
%   下的平均分支比 (branching_mean)，绘制随 M_T 变化的折线图，并在分支比等于 1
%   的位置绘制虚线参考线。输出格式参考 plot_delta_c_scan_results。
%
% 输入参数：
%   无（直接从配置的数据路径读取）
%
% 输出：
%   在 pic/ 目录下生成 PDF 格式的分支比图像文件
%
% 依赖文件：
%   - data/experiments/delta_c_m_vs_1_scan/[时间戳]/data.mat
%   - 需要包含 branching_mean, branching_sem, pulse_counts, cj_thresholds 等字段
%
% 作者：自动生成
% 日期：2025年

% 清理工作空间、命令窗口和关闭所有图形窗口
clear; clc; close all;

%% -------------------- 图像样式配置 --------------------
% 图片尺寸设置（单位：像素）
FIG_WIDTH = 400;      % 图片宽度
FIG_HEIGHT = 200;     % 图片高度

% 线条粗细设置
LINE_WIDTH = 2.5;         % 数据曲线线宽
AXIS_LINE_WIDTH = 1.5;    % 坐标轴框线线宽

% 字体样式设置
FONT_NAME = 'Arial';           % 字体名称（全局）
LABEL_FONT_SIZE = 12;          % 坐标轴标签字体大小（xlabel, ylabel）
LABEL_FONT_WEIGHT = 'Bold';    % 坐标轴标签字体粗细 ('normal' 或 'bold')
TICK_FONT_SIZE = 13;           % 坐标轴刻度字体大小
TICK_FONT_WEIGHT = 'Bold';     % 坐标轴刻度字体粗细 ('normal' 或 'bold')
LEGEND_FONT_SIZE = 12;         % 图例字体大小

% 网格和刻度设置
GRID_ON = 'off';              % 网格开关 ('on' 或 'off')
GRID_ALPHA = 0.1;             % 网格透明度 (0-1)
GRID_LINE_STYLE = '--';       % 网格线型 ('-' 实线, '--' 虚线, ':' 点线, '-.' 点划线)
TICK_DIR = 'in';              % 刻度方向 ('in' 或 'out')

% 颜色设置
BR_COLOR = [38, 94, 180] / 255;              % 分支比曲线颜色（蓝色）
THRESHOLD_LINE_COLOR = [0.95, 0.75, 0.1];    % 阈值参考线颜色（金黄色）

%% -------------------- 数据路径配置 --------------------
% 数据文件和目录路径设置
mat_file = 'data.mat';  % 数据文件名
% TODO: 根据实际结果目录更新以下路径
mat_dir = fullfile('mst_critcal', 'data', 'experiments', 'delta_c_m_vs_1_scan', ...
    '20251030_233909');  % 示例目录，请替换为真实时间戳

% 获取脚本所在目录和项目根目录（用于构建绝对路径）
script_dir = fileparts(mfilename('fullpath'));  % 获取当前脚本所在目录
project_root = fileparts(fileparts(fileparts(script_dir)));  % 向上三级获取项目根目录

% 构建数据文件的绝对路径
mat_path_abs = fullfile(project_root, mat_dir, mat_file);
% 检查数据文件是否存在
if ~isfile(mat_path_abs)
    error('结果文件不存在：%s', mat_path_abs);
end

% 加载数据文件中的 results 变量
data = load(mat_path_abs, 'results');
results = data.results;

%% -------------------- 参数选择 --------------------
% 检查结果数据中是否包含必要的分支比字段
if ~isfield(results, 'branching_mean')
    error('结果中缺少 branching_mean 字段，请确认已使用更新后的 run_delta_c_m_vs_1_scan_parallel。');
end

% 提取关键数据变量（转换为列向量以确保一致性）
pulse_counts = results.pulse_counts(:);      % 初发个体数数组
branching_mean = results.branching_mean;     % 平均分支比矩阵（行：阈值，列：初发个体数）
branching_sem = results.branching_sem;       % 分支比标准误差矩阵

% 设置目标初发个体数（默认为1，即单初发个体情况）
TARGET_PULSE = 1;  % 默认使用单初发个体 (c1)
% 在 pulse_counts 数组中查找目标初发个体数对应的索引
target_idx = find(pulse_counts == TARGET_PULSE, 1);
if isempty(target_idx)
    error('结果中未找到 pulse_count = %d 的分支比数据。', TARGET_PULSE);
end

% 提取阈值数据和对应目标初发个体数的分支比数据
thresholds = results.cj_thresholds(:);       % 阈值数组（M_T值）
mean_values = branching_mean(:, target_idx); % 目标初发个体数对应的平均分支比
sem_values = branching_sem(:, target_idx);   % 目标初发个体数对应的分支比标准误差

%% -------------------- 图像输出目录 --------------------
% 设置图片输出目录
pic_dir = fullfile(project_root, 'pic');
% 如果输出目录不存在，则创建该目录
if ~exist(pic_dir, 'dir')
    mkdir(pic_dir);
end

% 生成输出文件名（包含初发个体数和系统规模N信息）
output_name = sprintf('branching_ratio_from_delta_c_m_vs_1_p%d_N%d.pdf', ...
    TARGET_PULSE, results.parameters.N);
output_path = fullfile(pic_dir, output_name);

%% -------------------- 绘图 --------------------
% 创建图形窗口和坐标轴
fig = figure('Position', [150, 150, FIG_WIDTH, FIG_HEIGHT], 'Color', 'white');
ax = axes('Parent', fig);
hold(ax, 'on');  % 保持当前图形，允许叠加绘制

% 绘制分支比曲线
plot(ax, thresholds, mean_values, 'LineWidth', LINE_WIDTH, 'Color', BR_COLOR);

% 如果存在标准误差数据，绘制误差带（半透明填充区域）
if ~all(isnan(sem_values))
    % 构建误差带的x和y坐标（使用flipud实现闭合路径）
    patch_x = [thresholds; flipud(thresholds)];
    patch_y = [mean_values + sem_values; flipud(mean_values - sem_values)];
    % 绘制半透明误差带
    patch(ax, patch_x, patch_y, BR_COLOR, 'FaceAlpha', 0.15, ...
        'EdgeColor', 'none', 'HandleVisibility', 'off');
end

% 在分支比等于1的位置绘制水平参考线（临界值）
yline(ax, 1.0, '--', 'Color', THRESHOLD_LINE_COLOR, 'LineWidth', 2.0, 'HandleVisibility', 'off');

% 设置坐标轴属性
ax.FontName = FONT_NAME;           % 字体名称
ax.FontSize = TICK_FONT_SIZE;      % 刻度字体大小
ax.FontWeight = TICK_FONT_WEIGHT;  % 刻度字体粗细
ax.LineWidth = AXIS_LINE_WIDTH;    % 坐标轴线宽
ax.TickDir = TICK_DIR;             % 刻度方向
grid(ax, GRID_ON);                 % 网格开关
ax.GridAlpha = GRID_ALPHA;         % 网格透明度
ax.GridLineStyle = GRID_LINE_STYLE;% 网格线型
box(ax, 'on');                     % 显示坐标轴边框

% 设置坐标轴标签
xlabel(ax, 'M_T', 'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);
ylabel(ax, sprintf('Mean Branching Ratio (m=%d)', TARGET_PULSE), ...
    'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);

% 设置y轴范围（确保包含所有数据点和参考线）
ylim_upper = max(max(mean_values + abs(sem_values)) * 1.1, 1.1);  % 上限为数据最大值的1.1倍或1.1的较大者
ylim(ax, [0, ylim_upper]);  % y轴范围从0到计算的上限

% 添加图例
legend(ax, {sprintf('m = %d', TARGET_PULSE)}, 'Location', 'northeast', ...
    'FontName', FONT_NAME, 'FontSize', LEGEND_FONT_SIZE, 'Box', 'off');

hold(ax, 'off');  % 结束图形保持状态

% 导出图形为PDF矢量格式
exportgraphics(fig, output_path, 'ContentType', 'vector');
fprintf('Branching ratio plot saved to: %s\n', output_path);
