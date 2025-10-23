% plot_delta_c_scan_results 脚本
%
% 功能描述：
% 本脚本用于可视化 delta_c 参数扫描实验的结果，从 run_delta_c_parameter_scan_parallel
% 生成的 MAT 文件中读取数据并生成科研级质量图表。
%
% 输入：
%   - 包含实验结果的 .mat 文件（通过文件选择对话框选择）
%   - 文件必须包含 results 结构体，内含实验数据和参数
%
% 输出：
%   - 级联规模图：显示不同 c_j 阈值下的平均级联规模
%   - Delta C 图：显示不同 c_j 阈值下的 Δc = c_2 - c_1 差值
%   - 两个 PDF 文件保存在 MAT 文件所在目录的 results 子目录中：
%     * <MAT所在目录>/results/<mat_name>_cascade.pdf
%     * <MAT所在目录>/results/<mat_name>_delta.pdf
%
% 作者：自动生成脚本
% 日期：2025年

% 清理工作空间和命令行窗口
clear; clc;

% ==================== 图片尺寸配置 ====================
% 设置输出图片的尺寸（单位：像素）
FIG_WIDTH = 800;   % 图片宽度（像素）
FIG_HEIGHT = 600;  % 图片高度（像素）

% ==================== 文件选择和数据加载 ====================
% 弹出文件选择对话框，让用户选择实验数据文件
[mat_file, mat_path] = uigetfile('*.mat', ...
    '选择 run\_delta\_c\_parameter\_scan\_parallel 生成的 data.mat');
% 检查用户是否取消了文件选择
if isequal(mat_file, 0)
    fprintf('未选择文件，脚本结束。\n');
    return;
end

% 构建完整的文件路径
results_mat_file = fullfile(mat_path, mat_file);

% 从 MAT 文件中加载 results 结构体
data = load(results_mat_file, 'results');
% 验证文件中是否包含所需的 results 结构体
if ~isfield(data, 'results')
    error('文件中缺少 results 结构体: %s', results_mat_file);
end
results = data.results;

% ==================== 数据完整性验证 ====================
% 定义实验结果必须包含的字段列表
required_fields = {'cj_thresholds','c1_mean','c1_sem','c2_mean','c2_sem', ...
    'delta_c','parameters','num_runs','parallel_workers'};
% 逐个检查必需字段是否存在
for f = required_fields
    if ~isfield(results, f{1})
        error('results 缺少必须字段: %s', f{1});
    end
end

% ==================== 数据提取和预处理 ====================
% 从 results 结构体中提取数据并转换为行向量格式
% 确保所有数据都是行向量以便后续绘图
thresholds = results.cj_thresholds(:)';  % c_j 阈值数组
c1_mean = results.c1_mean(:)';          % c_1 的平均值
c2_mean = results.c2_mean(:)';          % c_2 的平均值
c1_sem = results.c1_sem(:)';            % c_1 的标准误差
c2_sem = results.c2_sem(:)';            % c_2 的标准误差
delta_c = results.delta_c(:)';          % Δc = c_2 - c_1 差值
% 提取文件名（不含扩展名）用于输出文件命名
[~, mat_name, ~] = fileparts(results_mat_file);

% ==================== 输出目录设置 ====================
% 获取当前脚本所在目录（src/sim/analysis）
script_dir = fileparts(mfilename('fullpath'));
% 获取项目根目录（向上三级：analysis -> sim -> src -> project_root）
project_root = fileparts(fileparts(fileparts(script_dir)));
% 定义 data 和 results 根目录（与项目根目录平级）
data_root = fullfile(project_root, 'data');
results_root = fullfile(project_root, 'results');

% 获取 MAT 文件相对于 data 目录的相对路径
if startsWith(mat_path, data_root)
    % 提取相对路径（去除 data_root 前缀）
    relative_path_from_data = mat_path(length(data_root)+1:end);
    % 移除开头的文件分隔符
    if startsWith(relative_path_from_data, filesep)
        relative_path_from_data = relative_path_from_data(2:end);
    end
    % 构建对应的 results 子目录路径
    output_dir = fullfile(results_root, relative_path_from_data);
else
    % 如果 MAT 文件不在 data 目录下，直接输出到 results 根目录
    output_dir = results_root;
end
% 检查输出目录是否存在，若不存在则创建
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
% 构建输出 PDF 文件的完整路径
cascade_pdf = fullfile(output_dir, sprintf('%s_cascade.pdf', mat_name));
delta_pdf = fullfile(output_dir, sprintf('%s_delta.pdf', mat_name));

% ==================== 级联规模图绘制 ====================
% 创建级联规模图的画布，包含实验参数信息
cascade_fig = create_canvas(sprintf('级联规模 (N=%d, runs=%d)', ...
    results.parameters.N, results.num_runs), FIG_WIDTH, FIG_HEIGHT);
% 创建坐标轴对象
ax1 = axes('Parent', cascade_fig);
% 设置坐标轴保持模式，允许多次绘图
hold(ax1, 'on');
% 绘制 c_1 的带误差阴影的曲线（绿色）
shaded_line(ax1, thresholds, c1_mean, c1_sem, [27 158 119]/255, 'c_1');
% 绘制 c_2 的带误差阴影的曲线（橙色）
shaded_line(ax1, thresholds, c2_mean, c2_sem, [217 95 2]/255, 'c_2');
% 设置 x 轴标签
xlabel(ax1, 'c_j threshold', 'FontWeight', 'bold');
% 设置 y 轴标签
ylabel(ax1, '平均级联规模', 'FontWeight', 'bold');
% 添加图例（位置：左上角，无边框）
legend(ax1, 'Location', 'northwest', 'Box', 'off', 'FontSize', 10);
% 应用统一的坐标轴样式
apply_axes_style(ax1);
% 导出高质量矢量 PDF 文件
exportgraphics(cascade_fig, cascade_pdf, 'ContentType', 'vector');
% 输出保存信息
fprintf('级联规模图已保存至: %s\n', cascade_pdf);

% ==================== Delta C 分析和绘制 ====================
% 找到 Δc 的最大值及其对应的索引
[max_delta_c, max_idx] = max(delta_c);
% 获取最大 Δc 对应的 c_j 阈值
optimal_cj = thresholds(max_idx);

% 创建 Delta C 图的画布
delta_fig = create_canvas(sprintf('\\Delta c (N=%d, runs=%d)', ...
    results.parameters.N, results.num_runs), FIG_WIDTH, FIG_HEIGHT);
% 创建坐标轴对象
ax2 = axes('Parent', delta_fig);
% 设置坐标轴保持模式
hold(ax2, 'on');
% 绘制 Δc 曲线（深灰色，线宽为2）
plot(ax2, thresholds, delta_c, 'Color', [45 45 45]/255, 'LineWidth', 2);
% 在最大值点标记红色圆圈
plot(ax2, optimal_cj, max_delta_c, 'o', 'MarkerSize', 7, ...
    'MarkerFaceColor', [228 26 28]/255, 'MarkerEdgeColor', 'k', 'LineWidth', 1.2);
% 在最大值点添加文本标注，显示峰值和对应的阈值
text(optimal_cj, max_delta_c, sprintf(' 峰值 %.3f @ %.2f', max_delta_c, optimal_cj), ...
    'Parent', ax2, 'FontSize', 11, 'FontWeight', 'bold', ...
    'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');
% 设置 x 轴标签
xlabel(ax2, 'c_j threshold', 'FontWeight', 'bold');
% 设置 y 轴标签（显示数学公式）
ylabel(ax2, '\Delta c = c_2 - c_1', 'FontWeight', 'bold');
% 应用统一的坐标轴样式
apply_axes_style(ax2);
% 导出高质量矢量 PDF 文件
exportgraphics(delta_fig, delta_pdf, 'ContentType', 'vector');
% 输出保存信息
fprintf('Δc 图已保存至: %s\n', delta_pdf);

%% ==================== 辅助函数定义 ====================

% create_canvas - 创建标准化画布
%
% 功能描述：
%   创建具有自定义尺寸和样式的图形画布，适用于科研论文图表
%
% 输入参数：
%   title_text - 字符串，图表标题
%   width - 数值，图片宽度（像素），默认 800
%   height - 数值，图片高度（像素），默认 600
%
% 输出参数：
%   fig - 图形对象句柄
%
% 详细说明：
%   - 画布尺寸可自定义，单位为像素
%   - 白色背景，自动纸张位置模式
%   - 在顶部添加标题文本框
function fig = create_canvas(title_text, width, height)
    % 设置默认参数
    if nargin < 2
        width = 800;   % 默认宽度 800 像素
    end
    if nargin < 3
        height = 600;  % 默认高度 600 像素
    end

    % 创建图形窗口，设置位置和尺寸（像素单位）
    fig = figure('Position', [100, 100, width, height], ...
        'Color', 'white', 'PaperPositionMode', 'auto');

    % 在图形顶部添加标题文本框
    annotation(fig, 'textbox', [0 0.93 1 0.06], 'String', title_text, ...
        'HorizontalAlignment', 'center', 'FontSize', 13, ...
        'FontWeight', 'bold', 'LineStyle', 'none');
end

% shaded_line - 绘制带误差阴影的曲线
%
% 功能描述：
%   绘制带有标准误差阴影区域的曲线，用于可视化数据的不确定性范围
%
% 输入参数：
%   ax - 坐标轴对象句柄
%   x - x 轴数据向量
%   mean_vals - 平均值向量
%   sem_vals - 标准误差向量
%   color_rgb - RGB 颜色向量（范围 [0,1]）
%   label_str - 图例标签字符串
%
% 详细说明：
%   - 使用半透明填充区域显示误差范围
%   - 误差区域不显示在图例中
%   - 主曲线使用指定颜色和线宽绘制
function shaded_line(ax, x, mean_vals, sem_vals, color_rgb, label_str)
    % 确保所有输入向量都是行向量格式
    x = x(:)'; mean_vals = mean_vals(:)'; sem_vals = sem_vals(:)';
    % 计算误差范围的上界和下界
    upper = mean_vals + sem_vals;  % 上界：平均值 + 标准误差
    lower = mean_vals - sem_vals;  % 下界：平均值 - 标准误差
    % 绘制误差阴影区域（半透明填充）
    patch(ax, [x fliplr(x)], [upper fliplr(lower)], color_rgb, ...
        'FaceAlpha', 0.18, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    % 绘制主曲线
    plot(ax, x, mean_vals, 'Color', color_rgb, 'LineWidth', 2, ...
        'DisplayName', label_str);
end

% apply_axes_style - 应用统一的坐标轴样式
%
% 功能描述：
%   为坐标轴应用标准化的科研图表样式，确保所有图表风格一致
%
% 输入参数：
%   ax - 坐标轴对象句柄
%
% 详细说明：
%   - 字体：Arial，11号
%   - 网格线：开启，透明度 0.25
%   - 坐标轴边框：开启
%   - 刻度方向：向外（符合科研出版标准）
function apply_axes_style(ax)
    % 设置字体属性
    ax.FontName = 'Arial';      % 使用 Arial 字体
    ax.FontSize = 11;            % 字体大小 11 号
    ax.LineWidth = 1;            % 坐标轴线宽
    % 设置网格属性
    grid(ax, 'off');              % 开启网格
    ax.GridAlpha = 0.25;         % 网格透明度
    % 设置边框和刻度
    box(ax, 'on');               % 开启坐标轴边框
    ax.TickDir = 'in';          % 刻度方向向外
end
