% Delta_c 实验结果分析与可视化脚本
% 加载并分析参数扫描实验结果
% 作者：[Your Name]
% 日期：2025-01-25

clc;
clear;
close all;

%% 1. 加载实验数据
fprintf('=================================================\n');
fprintf('       Delta_c 实验结果分析与可视化\n');
fprintf('=================================================\n\n');

% 获取最新的实验结果文件
data_files = dir('data/delta_c_scan_*.mat');
if isempty(data_files)
    error('未找到实验数据文件！请先运行 run_delta_c_parameter_scan.m');
end

% 选择最新的文件
[~, idx] = max([data_files.datenum]);
latest_file = fullfile('data', data_files(idx).name);

fprintf('加载数据文件: %s\n', latest_file);
load(latest_file, 'results');
fprintf('数据加载完成！\n\n');

% 提取关键变量
cj_thresholds = results.cj_thresholds;
c1_mean = results.c1_mean;
c2_mean = results.c2_mean;
c1_std = results.c1_std;
c2_std = results.c2_std;
c1_sem = results.c1_sem;
c2_sem = results.c2_sem;
delta_c = results.delta_c;
c1_raw = results.c1_raw;
c2_raw = results.c2_raw;

%% 2. 数据分析
fprintf('----------------------------------------\n');
fprintf('              数据分析\n');
fprintf('----------------------------------------\n');


% 基本统计
fprintf('实验参数:\n');
fprintf('  - N = %d (粒子数)\n', results.parameters.N);
fprintf('  - 场地大小 = %.1f × %.1f\n', results.parameters.fieldSize, results.parameters.fieldSize);
fprintf('  - 重复次数 = %d\n', results.num_runs);
fprintf('  - 参数范围: cj ∈ [%.1f, %.1f]\n', min(cj_thresholds), max(cj_thresholds));
fprintf('  - 参数点数: %d\n', length(cj_thresholds));

% 找出关键点
[max_delta_c, max_idx] = max(delta_c);
optimal_cj = cj_thresholds(max_idx);
fprintf('\n关键发现:\n');
fprintf('  - Δc 峰值 = %.4f\n', max_delta_c);
fprintf('  - 最优 cj = %.2f\n', optimal_cj);

% 寻找临界区间
threshold_50 = max_delta_c * 0.5;
above_threshold = delta_c > threshold_50;
if any(above_threshold)
    critical_range = cj_thresholds(above_threshold);
    fprintf('  - 临界区间 (Δc > 50%%峰值): [%.2f, %.2f]\n', ...
        min(critical_range), max(critical_range));
end

% 单调性分析
dc_diff = diff(delta_c);
increasing_ratio = sum(dc_diff > 0) / length(dc_diff);
fprintf('  - Δc 增长区间比例: %.1f%%\n', increasing_ratio * 100);

%% 3. 主要结果图（发表级质量）
fprintf('\n生成主要结果图...\n');

% 设置全局字体大小
set(0, 'DefaultAxesFontSize', 12);
set(0, 'DefaultTextFontSize', 12);
set(0, 'DefaultLineLineWidth', 1.5);

% 创建主图
fig_main = figure('Name', 'Delta_c Analysis - Main Results', ...
    'Position', [100, 100, 1400, 500]);

% --- 子图1: 级联规模对比 ---
subplot(1, 3, 1);
hold on;

% 绘制c1和c2的均值和误差带
h1 = errorbar(cj_thresholds, c1_mean, c1_sem, 'b-', 'LineWidth', 2);
h2 = errorbar(cj_thresholds, c2_mean, c2_sem, 'r-', 'LineWidth', 2);

% 添加阴影区域表示标准差范围
x_fill = [cj_thresholds, fliplr(cj_thresholds)];
y_fill_c1 = [c1_mean' - c1_std', fliplr(c1_mean' + c1_std')];
y_fill_c2 = [c2_mean' - c2_std', fliplr(c2_mean' + c2_std')];

fill(x_fill, y_fill_c1, 'b', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
fill(x_fill, y_fill_c2, 'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none');

xlabel('运动显著性阈值 c_j', 'FontSize', 14);
ylabel('平均级联规模', 'FontSize', 14);
title('(a) 级联规模', 'FontSize', 14, 'FontWeight', 'bold');
legend([h1, h2], {'c_1 (单个初发)', 'c_2 (两个初发)'}, ...
    'Location', 'best', 'FontSize', 12);
grid on;
box on;
xlim([min(cj_thresholds), max(cj_thresholds)]);
ylim([0, max([c2_mean + c2_std; c1_mean + c1_std]) * 1.1]);

% --- 子图2: Delta_c 主结果 ---
subplot(1, 3, 2);
hold on;

% 绘制delta_c主曲线
h_delta = plot(cj_thresholds, delta_c, 'k-', 'LineWidth', 2.5);

% 标记峰值点
plot(optimal_cj, max_delta_c, 'ro', 'MarkerSize', 12, 'LineWidth', 2, ...
    'MarkerFaceColor', 'r');

% 添加峰值标注
text(optimal_cj, max_delta_c * 1.05, ...
    sprintf('峰值: %.3f\ncj = %.2f', max_delta_c, optimal_cj), ...
    'HorizontalAlignment', 'center', 'FontSize', 11);

% 添加临界区域阴影
if exist('critical_range', 'var')
    y_lim = ylim;
    x_crit = [min(critical_range), max(critical_range), ...
              max(critical_range), min(critical_range)];
    y_crit = [y_lim(1), y_lim(1), y_lim(2), y_lim(2)];
    fill(x_crit, y_crit, 'y', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
end

xlabel('运动显著性阈值 c_j', 'FontSize', 14);
ylabel('\Delta c = c_2 - c_1', 'FontSize', 14);
title('(b) 集群敏感性', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
box on;
xlim([min(cj_thresholds), max(cj_thresholds)]);

% 添加零线参考
yline(0, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1);

% --- 子图3: 相对增强 ---
subplot(1, 3, 3);
hold on;

% 计算相对增强（避免除零）
relative_enhancement = zeros(size(c1_mean));
valid_idx = c1_mean > 0.001;  % 避免除零
relative_enhancement(valid_idx) = delta_c(valid_idx) ./ c1_mean(valid_idx);

plot(cj_thresholds, relative_enhancement, 'g-', 'LineWidth', 2.5);

% 标记最大相对增强
[max_rel, max_rel_idx] = max(relative_enhancement);
plot(cj_thresholds(max_rel_idx), max_rel, 'go', ...
    'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor', 'g');

xlabel('运动显著性阈值 c_j', 'FontSize', 14);
ylabel('相对增强 \Delta c / c_1', 'FontSize', 14);
title('(c) 相对集群效应', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
box on;
xlim([min(cj_thresholds), max(cj_thresholds)]);

% 添加零线参考
yline(0, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1);

% 调整子图间距
set(gcf, 'Color', 'white');

% 保存主图
savefig(fig_main, sprintf('data/delta_c_analysis_main_%s.fig', results.timestamp));
print(sprintf('data/delta_c_analysis_main_%s.png', results.timestamp), ...
    '-dpng', '-r300');
fprintf('主要结果图已保存\n');

% 导出主图数据
mainData = struct();
mainData.subplot1 = struct();
mainData.subplot1.title = '级联规模';
mainData.subplot1.xlabel = '运动显著性阈值 c_j';
mainData.subplot1.ylabel = '平均级联规模';
mainData.subplot1.x = cj_thresholds;
mainData.subplot1.c1_mean = c1_mean;
mainData.subplot1.c2_mean = c2_mean;
mainData.subplot1.c1_sem = c1_sem;
mainData.subplot1.c2_sem = c2_sem;
mainData.subplot1.c1_std = c1_std;
mainData.subplot1.c2_std = c2_std;
mainData.subplot1.legend = {'c_1 (单个初发)', 'c_2 (两个初发)'};

mainData.subplot2 = struct();
mainData.subplot2.title = '集群敏感性';
mainData.subplot2.xlabel = '运动显著性阈值 c_j';
mainData.subplot2.ylabel = '\Delta c = c_2 - c_1';
mainData.subplot2.x = cj_thresholds;
mainData.subplot2.delta_c = delta_c;
mainData.subplot2.max_delta_c = max_delta_c;
mainData.subplot2.optimal_cj = optimal_cj;

mainData.subplot3 = struct();
mainData.subplot3.title = '相对集群效应';
mainData.subplot3.xlabel = '运动显著性阈值 c_j';
mainData.subplot3.ylabel = '相对增强 \Delta c / c_1';
mainData.subplot3.x = cj_thresholds;
mainData.subplot3.relative_enhancement = relative_enhancement;
mainData.subplot3.max_relative_enhancement = max_rel;

% 调用数据导出函数
exportFigureData(mainData, 'delta_c_analysis_main', 'Delta_c分析主要结果图');

%% 4. 数据质量分析图
fprintf('生成数据质量分析图...\n');

fig_quality = figure('Name', 'Data Quality Analysis', ...
    'Position', [100, 100, 1200, 800]);

% --- 子图1: 数据分布热图 ---
subplot(2, 3, 1);
imagesc(cj_thresholds, 1:results.num_runs, c1_raw');
colorbar;
xlabel('c_j threshold');
ylabel('实验运行编号');
title('c_1 原始数据热图');
colormap(subplot(2,3,1), 'parula');

% --- 子图2: 数据分布热图 ---
subplot(2, 3, 2);
imagesc(cj_thresholds, 1:results.num_runs, c2_raw');
colorbar;
xlabel('c_j threshold');
ylabel('实验运行编号');
title('c_2 原始数据热图');
colormap(subplot(2,3,2), 'parula');

% --- 子图3: Delta_c分布 ---
subplot(2, 3, 3);
delta_c_raw = c2_raw - c1_raw;
imagesc(cj_thresholds, 1:results.num_runs, delta_c_raw');
colorbar;
xlabel('c_j threshold');
ylabel('实验运行编号');
title('\Delta c 原始数据热图');
colormap(subplot(2,3,3), 'RdBu');

% --- 子图4: 变异系数分析 ---
subplot(2, 3, 4);
cv_c1 = c1_std ./ c1_mean;
cv_c2 = c2_std ./ c2_mean;
hold on;
plot(cj_thresholds, cv_c1, 'b-', 'LineWidth', 2, 'DisplayName', 'CV(c_1)');
plot(cj_thresholds, cv_c2, 'r-', 'LineWidth', 2, 'DisplayName', 'CV(c_2)');
xlabel('c_j threshold');
ylabel('变异系数 (CV)');
title('数据变异性分析');
legend('Location', 'best');
grid on;

% --- 子图5: 箱线图 ---
subplot(2, 3, 5);
% 选择代表性参数点
n_representative = 10;
repr_indices = round(linspace(1, length(cj_thresholds), n_representative));

% 预分配boxplot_data和group_labels以提高性能
% 首先计算所需的总大小
total_data_size = 0;
for i = 1:length(repr_indices)
    idx = repr_indices(i);
    valid_c1 = c1_raw(idx, ~isnan(c1_raw(idx, :)));
    valid_c2 = c2_raw(idx, ~isnan(c2_raw(idx, :)));
    total_data_size = total_data_size + length(valid_c1) + length(valid_c2);
end

% 预分配数组
boxplot_data = zeros(1, total_data_size);
group_labels = zeros(1, total_data_size);

% 填充预分配的数组
current_idx = 1;
for i = 1:length(repr_indices)
    idx = repr_indices(i);
    valid_c1 = c1_raw(idx, ~isnan(c1_raw(idx, :)));
    valid_c2 = c2_raw(idx, ~isnan(c2_raw(idx, :)));
    
    len_c1 = length(valid_c1);
    len_c2 = length(valid_c2);
    
    % 填充boxplot_data
    boxplot_data(current_idx:current_idx+len_c1-1) = valid_c1;
    current_idx = current_idx + len_c1;
    boxplot_data(current_idx:current_idx+len_c2-1) = valid_c2;
    current_idx = current_idx + len_c2;
    
    % 填充group_labels
    group_labels(current_idx-len_c1-len_c2:current_idx-len_c2-1) = ones(1, len_c1) * i;
    group_labels(current_idx-len_c2:current_idx-1) = ones(1, len_c2) * (i + 0.3);
end

boxplot(boxplot_data, group_labels, 'Colors', 'br');
set(gca, 'XTick', 1:n_representative);
set(gca, 'XTickLabel', arrayfun(@(x) sprintf('%.1f', x), ...
    cj_thresholds(repr_indices), 'UniformOutput', false));
xlabel('c_j threshold');
ylabel('级联规模');
title('数据分布箱线图');
xtickangle(45);

% --- 子图6: 收敛性分析 ---
subplot(2, 3, 6);
% 计算累积平均值来检查收敛性
cumulative_mean_c1 = zeros(length(cj_thresholds), results.num_runs);
cumulative_mean_c2 = zeros(length(cj_thresholds), results.num_runs);

for i = 1:results.num_runs
    cumulative_mean_c1(:, i) = mean(c1_raw(:, 1:i), 2, 'omitnan');
    cumulative_mean_c2(:, i) = mean(c2_raw(:, 1:i), 2, 'omitnan');
end

% 选择几个代表性参数点展示
repr_param_idx = round(linspace(1, length(cj_thresholds), 5));
hold on;
colors = lines(5);
for i = 1:length(repr_param_idx)
    idx = repr_param_idx(i);
    plot(1:results.num_runs, cumulative_mean_c1(idx, :), '-', ...
        'Color', colors(i, :), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('c_1 (cj=%.1f)', cj_thresholds(idx)));
end
xlabel('实验运行次数');
ylabel('累积平均值');
title('收敛性分析');
legend('Location', 'best', 'FontSize', 9);
grid on;

% 保存数据质量图
savefig(fig_quality, sprintf('data/delta_c_quality_%s.fig', results.timestamp));
print(sprintf('data/delta_c_quality_%s.png', results.timestamp), ...
    '-dpng', '-r300');
fprintf('数据质量分析图已保存\n');

% 导出数据质量分析数据
qualityData = struct();
qualityData.subplot1 = struct();
qualityData.subplot1.title = 'c_1 原始数据热图';
qualityData.subplot1.xlabel = 'c_j threshold';
qualityData.subplot1.ylabel = '实验运行编号';
qualityData.subplot1.x = cj_thresholds;
qualityData.subplot1.y = 1:results.num_runs;
qualityData.subplot1.data = c1_raw';

qualityData.subplot2 = struct();
qualityData.subplot2.title = 'c_2 原始数据热图';
qualityData.subplot2.xlabel = 'c_j threshold';
qualityData.subplot2.ylabel = '实验运行编号';
qualityData.subplot2.x = cj_thresholds;
qualityData.subplot2.y = 1:results.num_runs;
qualityData.subplot2.data = c2_raw';

qualityData.subplot3 = struct();
qualityData.subplot3.title = '\Delta c 原始数据热图';
qualityData.subplot3.xlabel = 'c_j threshold';
qualityData.subplot3.ylabel = '实验运行编号';
qualityData.subplot3.x = cj_thresholds;
qualityData.subplot3.y = 1:results.num_runs;
qualityData.subplot3.data = delta_c_raw;

qualityData.subplot4 = struct();
qualityData.subplot4.title = '数据变异性分析';
qualityData.subplot4.xlabel = 'c_j threshold';
qualityData.subplot4.ylabel = '变异系数 (CV)';
qualityData.subplot4.x = cj_thresholds;
qualityData.subplot4.cv_c1 = cv_c1;
qualityData.subplot4.cv_c2 = cv_c2;
qualityData.subplot4.legend = {'CV(c_1)', 'CV(c_2)'};

qualityData.subplot5 = struct();
qualityData.subplot5.title = '数据分布箱线图';
qualityData.subplot5.xlabel = 'c_j threshold';
qualityData.subplot5.ylabel = '级联规模';
qualityData.subplot5.representative_indices = repr_indices;
qualityData.subplot5.representative_cj = cj_thresholds(repr_indices);
qualityData.subplot5.boxplot_data = boxplot_data;
qualityData.subplot5.group_labels = group_labels;

qualityData.subplot6 = struct();
qualityData.subplot6.title = '收敛性分析';
qualityData.subplot6.xlabel = '实验运行次数';
qualityData.subplot6.ylabel = '累积平均值';
qualityData.subplot6.x = 1:results.num_runs;
qualityData.subplot6.representative_param_idx = repr_param_idx;
qualityData.subplot6.cumulative_mean_c1 = cumulative_mean_c1;
qualityData.subplot6.cumulative_mean_c2 = cumulative_mean_c2;

% 调用数据导出函数
exportFigureData(qualityData, 'delta_c_quality', 'Delta_c数据质量分析图');

%% 5. 相位图分析
fprintf('生成相位图分析...\n');

fig_phase = figure('Name', 'Phase Diagram Analysis', ...
    'Position', [100, 100, 1000, 600]);

% --- 子图1: c1 vs c2 相位图 ---
subplot(1, 2, 1);
hold on;

% 绘制相位轨迹
scatter(c1_mean, c2_mean, 50, cj_thresholds, 'filled');
colormap('jet');
c = colorbar;
ylabel(c, 'c_j threshold');

% 添加对角线 (c1 = c2)
lim = [0, max([c1_mean; c2_mean]) * 1.1];
plot(lim, lim, 'k--', 'LineWidth', 1.5);

% 标注关键点
plot(c1_mean(max_idx), c2_mean(max_idx), 'ro', ...
    'MarkerSize', 12, 'LineWidth', 2);
text(c1_mean(max_idx), c2_mean(max_idx), ...
    sprintf('  最大Δc\n  cj=%.2f', optimal_cj), ...
    'FontSize', 10);

xlabel('c_1 (单个初发)', 'FontSize', 12);
ylabel('c_2 (两个初发)', 'FontSize', 12);
title('(a) 级联规模相位图', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
axis equal;
xlim(lim);
ylim(lim);

% --- 子图2: 3D相位图 ---
subplot(1, 2, 2);
surf(cj_thresholds, 1:results.num_runs, c2_raw' - c1_raw', ...
    'EdgeColor', 'none', 'FaceAlpha', 0.8);
colormap('RdBu');
c = colorbar;
ylabel(c, '\Delta c');
view(45, 30);
xlabel('c_j threshold');
ylabel('实验运行');
zlabel('\Delta c');
title('(b) Δc 3D分布', 'FontSize', 14, 'FontWeight', 'bold');
grid on;

% 保存相位图
savefig(fig_phase, sprintf('data/delta_c_phase_%s.fig', results.timestamp));
print(sprintf('data/delta_c_phase_%s.png', results.timestamp), ...
    '-dpng', '-r300');
fprintf('相位图分析已保存\n');

% 导出相位图分析数据
phaseData = struct();
phaseData.subplot1 = struct();
phaseData.subplot1.title = '级联规模相位图';
phaseData.subplot1.xlabel = 'c_1 (单个初发)';
phaseData.subplot1.ylabel = 'c_2 (两个初发)';
phaseData.subplot1.c1_mean = c1_mean;
phaseData.subplot1.c2_mean = c2_mean;
phaseData.subplot1.cj_thresholds = cj_thresholds;
phaseData.subplot1.max_idx = max_idx;
phaseData.subplot1.optimal_cj = optimal_cj;
phaseData.subplot1.max_delta_c = max_delta_c;

phaseData.subplot2 = struct();
phaseData.subplot2.title = '\Delta c 3D分布';
phaseData.subplot2.xlabel = 'c_j threshold';
phaseData.subplot2.ylabel = '实验运行';
phaseData.subplot2.zlabel = '\Delta c';
phaseData.subplot2.x = cj_thresholds;
phaseData.subplot2.y = 1:results.num_runs;
phaseData.subplot2.z = c2_raw' - c1_raw';

% 调用数据导出函数
exportFigureData(phaseData, 'delta_c_phase', 'Delta_c相位图分析');

%% 6. 生成报告
fprintf('\n----------------------------------------\n');
fprintf('生成分析报告...\n');

% 创建报告文件
report_filename = sprintf('data/delta_c_report_%s.txt', results.timestamp);
fid = fopen(report_filename, 'w');

fprintf(fid, '=================================================\n');
fprintf(fid, '       Delta_c 参数扫描实验分析报告\n');
fprintf(fid, '=================================================\n\n');
fprintf(fid, '生成时间: %s\n\n', char(datetime("now", "Format", "yyyy-MM-dd HH:mm:ss")));

fprintf(fid, '1. 实验参数\n');
fprintf(fid, '----------------------------------------\n');
fprintf(fid, '  粒子数量 N: %d\n', results.parameters.N);
fprintf(fid, '  场地大小: %.1f × %.1f\n', ...
    results.parameters.fieldSize, results.parameters.fieldSize);
fprintf(fid, '  粒子速度: %.1f\n', results.parameters.v0);
fprintf(fid, '  邻居半径: %.1f\n', results.parameters.radius);
fprintf(fid, '  稳定期步数: %d\n', results.parameters.stabilization_steps);
fprintf(fid, '  仿真总步数: %d\n', results.parameters.T_max);
fprintf(fid, '  参数扫描范围: cj ∈ [%.1f, %.1f], 步长 %.2f\n', ...
    min(cj_thresholds), max(cj_thresholds), cj_thresholds(2) - cj_thresholds(1));
fprintf(fid, '  参数点数: %d\n', length(cj_thresholds));
fprintf(fid, '  每点重复次数: %d\n', results.num_runs);
fprintf(fid, '  总实验次数: %d\n\n', results.total_experiments);

fprintf(fid, '2. 关键结果\n');
fprintf(fid, '----------------------------------------\n');
fprintf(fid, '  Δc 峰值: %.6f\n', max_delta_c);
fprintf(fid, '  最优 cj_threshold: %.3f\n', optimal_cj);
fprintf(fid, '  峰值处 c1: %.6f ± %.6f\n', ...
    c1_mean(max_idx), c1_std(max_idx));
fprintf(fid, '  峰值处 c2: %.6f ± %.6f\n', ...
    c2_mean(max_idx), c2_std(max_idx));

if exist('critical_range', 'var')
    fprintf(fid, '  临界区间 (Δc > 50%%峰值): [%.3f, %.3f]\n', ...
        min(critical_range), max(critical_range));
end

fprintf(fid, '\n3. 统计摘要\n');
fprintf(fid, '----------------------------------------\n');
fprintf(fid, '  c1 全局均值: %.6f\n', mean(c1_mean, 'omitnan'));
fprintf(fid, '  c2 全局均值: %.6f\n', mean(c2_mean, 'omitnan'));
fprintf(fid, '  Δc 均值: %.6f\n', mean(delta_c, 'omitnan'));
fprintf(fid, '  最大相对增强: %.2f%%\n', max_rel * 100);
fprintf(fid, '  实验总耗时: %.2f 小时\n', results.total_time_hours);

fprintf(fid, '\n4. 数据质量\n');
fprintf(fid, '----------------------------------------\n');
total_errors = sum(results.error_count(:));
success_rate = 100 * (1 - total_errors / results.total_experiments);
fprintf(fid, '  成功率: %.2f%%\n', success_rate);
fprintf(fid, '  c1 平均变异系数: %.4f\n', mean(cv_c1, 'omitnan'));
fprintf(fid, '  c2 平均变异系数: %.4f\n', mean(cv_c2, 'omitnan'));

fclose(fid);
fprintf('分析报告已保存至: %s\n', report_filename);

%% 7. 完成
fprintf('\n=================================================\n');
fprintf('             分析完成！\n');
fprintf('=================================================\n');
fprintf('生成的文件:\n');
fprintf('  1. 主结果图: delta_c_analysis_main_%s.png\n', results.timestamp);
fprintf('  2. 数据质量图: delta_c_quality_%s.png\n', results.timestamp);
fprintf('  3. 相位图: delta_c_phase_%s.png\n', results.timestamp);
fprintf('  4. 分析报告: delta_c_report_%s.txt\n', results.timestamp);
fprintf('\n');

%% 数据导出辅助函数
function exportFigureData(data, filename, description)
    % 确保data目录存在
    if ~exist('data', 'dir')
        mkdir('data');
    end
    
    % 创建完整的数据结构
    exportData = struct();
    exportData.metadata = struct();
    exportData.metadata.description = description;
    exportData.metadata.timestamp = results.timestamp;
    exportData.metadata.creationTime = char(datetime("now", "Format", "yyyy-MM-dd HH:mm:ss"));
    exportData.metadata.parameters = results.parameters;
    exportData.metadata.experimentInfo = struct();
    exportData.metadata.experimentInfo.numRuns = results.num_runs;
    exportData.metadata.experimentInfo.totalExperiments = results.total_experiments;
    exportData.metadata.experimentInfo.totalTimeHours = results.total_time_hours;
    exportData.data = data;
    
    % 导出为JSON文件
    jsonFilename = sprintf('data/%s_%s.json', filename, results.timestamp);
    fid = fopen(jsonFilename, 'w', 'n', 'UTF-8');
    if fid == -1
        error('无法创建文件: %s', jsonFilename);
    end
    
    try
        % 使用MATLAB 2025a的jsonencode函数
        jsonStr = jsonencode(exportData, 'PrettyPrint', true);
        fwrite(fid, jsonStr, 'UTF-8');
        fclose(fid);
        fprintf('数据已导出至: %s\n', jsonFilename);
    catch ME
        fclose(fid);
        error('JSON导出失败: %s', ME.message);
    end
end