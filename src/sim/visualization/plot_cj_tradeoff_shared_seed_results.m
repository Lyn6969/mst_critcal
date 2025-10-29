% plot_cj_tradeoff_shared_seed_results
% =========================================================================
% 作用：
%   - 读取 run_cj_tradeoff_adaptive_scan_shared_seed 生成的 MAT 文件。
%   - 基于文件中的 R/P 统计结果，按 P_mean 的 min-max 重新归一化后绘制图像。
%   - 图像保存为与原 MAT 同名的 *_replot.png 便于对比。
%
% 使用方法：
%   1. 修改 mat_file 为目标结果文件路径。
%   2. 运行本脚本即可（无需重新仿真）。
% =========================================================================

clear;
close all;

mat_file = 'results/tradeoff/cj_tradeoff_adaptive_shared_seed_20251029_150144.mat';
if ~isfile(mat_file)
    error('指定的 MAT 文件不存在：%s', mat_file);
end

data = load(mat_file);
if ~isfield(data, 'summary')
    error('MAT 文件缺少 summary 结构。');
end
summary = data.summary;
results = summary.results;

mode_ids = fieldnames(results);
P_means_all = [];
for idx = 1:numel(mode_ids)
    P_means_all = [P_means_all; results.(mode_ids{idx}).P_mean(:)]; %#ok<AGROW>
end
P_means_all = P_means_all(~isnan(P_means_all));
if isempty(P_means_all)
    warning('P_mean 全部为 NaN，归一化退化为常数。');
    P_min = 0;
    P_max = 1;
else
    P_min = min(P_means_all);
    P_max = max(P_means_all);
end
P_range = max(P_max - P_min, eps);

for idx = 1:numel(mode_ids)
    key = mode_ids{idx};
    res = results.(key);
    res.P_mean_norm = (res.P_mean - P_min) / P_range;
    res.P_std_norm = res.P_std / P_range;
    res.P_sem_norm = res.P_sem / P_range;
    results.(key) = res;
end

figure('Name', '固定 vs 自适应阈值（共享随机环境）', 'Color', 'white', 'Position', [160, 120, 960, 600]);
hold on;

if isfield(results, 'fixed')
    fixed_data = results.fixed;
    if isfield(summary, 'cj_thresholds_fixed')
        fixed_cj = summary.cj_thresholds_fixed;
    else
        fixed_cj = fixed_data.cj_thresholds;
    end
    scatter(fixed_data.R_mean, fixed_data.P_mean_norm, 70, fixed_cj, 'filled', 'DisplayName', '固定阈值');
    colormap('turbo');
    cb = colorbar;
    cb.Label.String = '固定阈值';
    plot(fixed_data.R_mean, fixed_data.P_mean_norm, '-', 'Color', [0.3 0.3 0.3], 'LineWidth', 1.2);
end

if isfield(results, 'adaptive')
    adaptive_data = results.adaptive;
    scatter(adaptive_data.R_mean, adaptive_data.P_mean_norm, 160, [0.85 0.2 0.2], 'filled', ...
        'DisplayName', sprintf('自适应阈值 %.3f', summary.adaptive_config.saliency_threshold), ...
        'Marker', 'p', 'MarkerEdgeColor', [0.5 0 0], 'MarkerFaceColor', [0.85 0.2 0.2]);
end

xlabel('响应性 R');
ylabel('归一化持久性 P (基于 P_{mean} min-max)');
title('固定阈值扫描 vs 自适应显著性阈值（共享随机环境）');
legend('Location', 'best');
grid on;

[pathstr, name] = fileparts(mat_file);
if isempty(pathstr)
    pathstr = '.';
end
output_fig = fullfile(pathstr, [name, '_replot.png']);
saveas(gcf, output_fig);
fprintf('重新绘制图像已保存：%s\n', output_fig);
