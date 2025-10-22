% 使用 noise_threshold_scan 结果绘制 c1/c2/Δc 曲线
% 直接运行本脚本，会弹窗选择 data.mat
clc
clear

[file, path] = uigetfile('*.mat', '选择 noise-threshold 扫描结果文件');
if isequal(file, 0)
    fprintf('未选择文件，脚本结束。\n');
    return;
end
results_file = fullfile(path, file);

loaded = load(results_file);
if ~isfield(loaded, 'results')
    error('文件中未找到 results 结构。');
end
R = loaded.results;

cj = R.cj_thresholds(:);
noise_levels = R.noise_levels(:);
c1_mean = squeeze(R.c_mean(:, :, 1));
c2_mean = squeeze(R.c_mean(:, :, 2));
delta_c = R.delta_c;

colors = lines(numel(noise_levels));
marker_list = {'o','s','d','^','v','>','<','p','h','x'};

% 图1：c1
figure('Name','c1 vs c_j','Color','w');
hold on;
for i = 1:numel(noise_levels)
    mk = marker_list{mod(i-1, numel(marker_list)) + 1};
    plot(cj, c1_mean(:, i), '-', 'Color', colors(i,:), ...
        'LineWidth', 1.5, 'DisplayName', sprintf('\\sigma = %.2f', noise_levels(i)));
    plot(cj, c1_mean(:, i), mk, 'Color', colors(i,:), 'MarkerSize', 5, ...
        'HandleVisibility', 'off');
end
xlabel('运动显著性阈值 c_j');
ylabel('c_1 平均级联规模');
title('c_1 vs c_j (不同噪声)');
legend('Location', 'northeastoutside');
grid on;

% 图2：c2
figure('Name','c2 vs c_j','Color','w');
hold on;
for i = 1:numel(noise_levels)
    mk = marker_list{mod(i-1, numel(marker_list)) + 1};
    plot(cj, c2_mean(:, i), '-', 'Color', colors(i,:), ...
        'LineWidth', 1.5, 'DisplayName', sprintf('\\sigma = %.2f', noise_levels(i)));
    plot(cj, c2_mean(:, i), mk, 'Color', colors(i,:), 'MarkerSize', 5, ...
        'HandleVisibility', 'off');
end
xlabel('运动显著性阈值 c_j');
ylabel('c_2 平均级联规模');
title('c_2 vs c_j (不同噪声)');
legend('Location', 'northeastoutside');
grid on;

% 图3：Δc
figure('Name','Δc vs c_j','Color','w');
hold on;
for i = 1:numel(noise_levels)
    mk = marker_list{mod(i-1, numel(marker_list)) + 1};
    plot(cj, delta_c(:, i), '-', 'Color', colors(i,:), ...
        'LineWidth', 1.5, 'DisplayName', sprintf('\\sigma = %.2f', noise_levels(i)));
    plot(cj, delta_c(:, i), mk, 'Color', colors(i,:), 'MarkerSize', 5, ...
        'HandleVisibility', 'off');
end
xlabel('运动显著性阈值 c_j');
ylabel('\Delta c = c_2 - c_1');
title('\Delta c vs c_j (不同噪声)');
legend('Location', 'northeastoutside');
grid on;
