% plot_responsiveness_noise_heatmap - 响应性噪声扫描热力图绘制
%
% 功能说明：
%   从 run_responsiveness_noise_cj_scan 生成的结果文件中读取响应性数据，
%   绘制运动显著性阈值 × 真实噪声大小的热力图。
%   横轴为运动显著性阈值 M_T，纵轴为真实噪声参数 η = √(2 D_θ)。

% 清理工作空间、命令窗口和关闭所有图形窗口
clear; clc; close all;

%% -------------------- 图像样式配置 --------------------
% 图片尺寸设置（单位：像素）
FIG_WIDTH = 500;      % 图片宽度
FIG_HEIGHT = 400;     % 图片高度

% 字体样式设置
FONT_NAME = 'Arial';           % 字体名称（全局）
LABEL_FONT_SIZE = 12;          % 坐标轴标签字体大小（xlabel, ylabel）
LABEL_FONT_WEIGHT = 'Bold';    % 坐标轴标签字体粗细
TICK_FONT_SIZE = 13;           % 坐标轴刻度字体大小
TICK_FONT_WEIGHT = 'Bold';     % 坐标轴刻度字体粗细
COLORBAR_FONT_SIZE = 12;       % 色阶条字体大小
COLORBAR_LINE_WIDTH = 1.5;     % 色阶条边框线宽

% 坐标轴设置
AXIS_LINE_WIDTH = 1.5;    % 坐标轴框线线宽
TICK_DIR = 'in';          % 刻度方向 ('in' 或 'out')

% 平滑与等高线配置
SMOOTHING_FACTOR = 10;     % 网格细分倍数 (>1 时启用插值平滑)
SMOOTHING_METHOD = 'makima'; % 插值方法（'linear'、'spline'、'makima' 等）
CONTOUR_LEVELS = 40;      % 等高线层数，数值越大视觉越平滑
REFERENCE_LEVELS = [0.8]; % 需要强调的 R 等值线
REFERENCE_LINE_WIDTH = 2;    % 等值线线宽
REFERENCE_LABEL_FONT_SIZE = 12; % 等值线标签字体大小
REFERENCE_LABEL_FONT_WEIGHT = 'Bold';
REFERENCE_LABEL_BG = 'none';    % 等值线标签背景颜色
REFERENCE_LABEL_SPACING = 300;  % 等值线标签间距控制

%% -------------------- 数据路径配置 --------------------
% TODO: 按实际结果路径修改以下目录与文件名
mat_file = 'results.mat';  % 数据文件名
mat_dir = fullfile('data', 'experiments', 'responsiveness_noise_cj_scan', ...
    '20251105_001140');  % 示例目录，请根据实际时间戳修改

% 获取脚本所在目录的绝对路径
script_dir = fileparts(mfilename('fullpath'));
% 获取项目根目录（向上两级目录）
project_root = fileparts(fileparts(script_dir));

% 构建数据文件的完整绝对路径
mat_path_abs = fullfile(project_root, mat_dir, mat_file);

% 检查文件是否存在
if ~isfile(mat_path_abs)
    error('未找到结果文件，请检查路径: %s', mat_path_abs);
end

% 加载数据文件中的results变量
data = load(mat_path_abs, 'results');
results = data.results;

%% -------------------- 数据提取与转换 --------------------
% 提取基础数据
cj_thresholds = results.cj_thresholds;      % 运动显著性阈值 (横轴)
noise_levels = results.noise_levels;        % 原始噪声参数 D_θ
R_mean = max(results.R_mean, 0);            % 响应性均值矩阵 (噪声×阈值，截断为非负)

% 转换噪声参数：η = √(2 D_θ)
eta_levels = sqrt(2 * noise_levels);        % 真实噪声参数 (纵轴)

% 检查数据维度
[num_noise, num_cj] = size(R_mean);
if num_noise ~= numel(eta_levels) || num_cj ~= numel(cj_thresholds)
    error('数据矩阵维度与参数数组不匹配。');
end

% 若需要，使用插值将数据网格细分以减弱“格子感”
if SMOOTHING_FACTOR > 1
    % 细化后的采样点数（确保端点包含在内）
    num_noise_fine = (num_noise - 1) * SMOOTHING_FACTOR + 1;
    num_cj_fine = (num_cj - 1) * SMOOTHING_FACTOR + 1;

    eta_plot = linspace(eta_levels(1), eta_levels(end), num_noise_fine);
    cj_plot = linspace(cj_thresholds(1), cj_thresholds(end), num_cj_fine);

    interpolant = griddedInterpolant({eta_levels, cj_thresholds}, R_mean, ...
        SMOOTHING_METHOD, 'nearest');
    R_plot = interpolant({eta_plot, cj_plot});
else
    eta_plot = eta_levels;
    cj_plot = cj_thresholds;
    R_plot = R_mean;
end

R_plot(R_plot < 0) = 0;

% 获取粒子数量（用于文件命名）
if isfield(results, 'resp_params') && isfield(results.resp_params, 'N')
    N_particles = results.resp_params.N;
else
    N_particles = 200;  % 默认值
end

%% -------------------- 图像输出目录 --------------------
% 设置图片输出目录
pic_dir = fullfile(project_root, 'pic');
% 如果目录不存在则创建
if ~exist(pic_dir, 'dir')
    mkdir(pic_dir);
end

% 生成输出文件名，包含粒子数量N参数
output_name = sprintf('responsiveness_noise_heatmap_N%d.pdf', N_particles);
output_path = fullfile(pic_dir, output_name);

%% -------------------- 绘制响应性热力图 --------------------
% 创建图形窗口，设置位置、大小和背景色
fig = figure('Position', [200, 200, FIG_WIDTH, FIG_HEIGHT], 'Color', 'white');
set(fig, 'Renderer', 'painters');
% 创建坐标轴对象
ax = axes('Parent', fig);

% 绘制平滑热力图（使用等高线填充并隐藏等高线轮廓）
contourf(ax, cj_plot, eta_plot, R_plot, CONTOUR_LEVELS, 'LineColor', 'none');
set(ax, 'YDir', 'normal');
axis(ax, 'tight');
ax.Layer = 'top';
clim(ax, [min(R_plot(:), [], 'omitnan'), max(R_plot(:), [], 'omitnan')]);

% 设置坐标轴字体属性
ax.FontName = FONT_NAME;
ax.FontSize = TICK_FONT_SIZE;
ax.FontWeight = TICK_FONT_WEIGHT;
ax.LineWidth = AXIS_LINE_WIDTH;
ax.TickDir = TICK_DIR;
% 显示坐标轴边框
box(ax, 'on');

% 设置坐标轴标签
xlabel(ax, 'M_T', 'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);
ylabel(ax, '\eta', 'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);

% 添加色阶条
cb = colorbar(ax);
cb.Label.String = 'Responsivity';
cb.Label.FontName = FONT_NAME;
cb.Label.FontSize = COLORBAR_FONT_SIZE;
cb.Label.FontWeight = LABEL_FONT_WEIGHT;
cb.FontName = FONT_NAME;
cb.FontSize = TICK_FONT_SIZE;
cb.LineWidth = COLORBAR_LINE_WIDTH;

% 设置色彩映射
colormap(ax, 'turbo');

% 在热力图上叠加指定 R 水平的等值线

hold(ax, 'on');
for ref_idx = 1:numel(REFERENCE_LEVELS)
    level = REFERENCE_LEVELS(ref_idx);
    [C, h_ref] = contour(ax, cj_plot, eta_plot, R_plot, [level level], ...
        'LineColor', [0 0 0], 'LineWidth', REFERENCE_LINE_WIDTH, 'LineStyle', '--');
    if ~isempty(h_ref) && ishghandle(h_ref)
        clabel(C, h_ref, 'LabelSpacing', REFERENCE_LABEL_SPACING, ...
            'FontName', FONT_NAME, 'FontSize', REFERENCE_LABEL_FONT_SIZE, ...
            'FontWeight', REFERENCE_LABEL_FONT_WEIGHT, 'Color', [0 0 0], ...
            'BackgroundColor', REFERENCE_LABEL_BG);
    end
end
hold(ax, 'off');

% 导出图形为PDF矢量格式
exportgraphics(fig, output_path, 'ContentType', 'vector');
% 输出保存路径信息
fprintf('响应性噪声热力图已保存至: %s\n', output_path);

% 输出数据统计信息
fprintf('数据统计:\n');
fprintf('  运动显著性阈值范围: [%.1f, %.1f]\n', min(cj_thresholds), max(cj_thresholds));
fprintf('  真实噪声η范围: [%.3f, %.3f]\n', min(eta_levels), max(eta_levels));
fprintf('  响应性R范围: [%.3f, %.3f]\n', min(R_mean(:), [], 'omitnan'), max(R_mean(:), [], 'omitnan'));

