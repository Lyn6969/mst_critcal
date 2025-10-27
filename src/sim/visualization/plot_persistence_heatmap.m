% plot_persistence_heatmap 脚本
% =========================================================================
% 作用：
%   - 从 persistence_heatmat.mat（或用户选择的任意 MAT 文件）中读取
%     持久性扫描结果，并生成两幅热力图：
%       1. 文件提供的平均持久性 P_mean（若缺失则提示并跳过）
%       2. 由平均扩散系数 D_mean 推导的持久性 P_from_D = 1/sqrt(D_mean)
%   - 统一处理极小扩散系数，避免除零造成的异常色阶
%   - 自动使用统一色条，方便比较不同计算方式的差异
%
% 使用方式：
%   - 直接运行脚本，将弹出文件选择框，选择 persistance_heatmat.mat。
%   - 也可以在命令行手动设置 mat_path/min_diffusion 变量后运行。
%
% 作者：李亚男
% 日期：2025 年
% =========================================================================

clc;
clear;
close all;

%% 1. 参数与输入 -----------------------------------------------------------
if ~exist('mat_path', 'var') || isempty(mat_path)
    [file, path] = uigetfile('*.mat', '选择 persistance_heatmat.mat');
    if isequal(file, 0)
        fprintf('用户取消操作。\n');
        return;
    end
    mat_path = fullfile(path, file);
else
    if ~isfile(mat_path)
        error('指定的 MAT 文件不存在：%s', mat_path);
    end
end

if ~exist('min_diffusion', 'var') || isempty(min_diffusion)
    min_diffusion = 1e-3;
end

if ~exist('clip_percentiles', 'var') || isempty(clip_percentiles)
    clip_percentiles = [0.02, 0.98];   % 颜色范围的分位点
end

fprintf('读取数据文件：%s\n', mat_path);

%% 2. 读取并解析数据 -------------------------------------------------------
raw = load(mat_path);
target = detect_result_struct(raw);
if isempty(target)
    error('未找到包含 cj_thresholds / noise_levels / D_mean 的结构体。');
end

cj = target.cj_thresholds(:)';
noise = target.noise_levels(:);
D_mean = target.D_mean;

if ~ismatrix(D_mean)
    error('D_mean 必须为二维矩阵。');
end

has_P_mean = isfield(target, 'P_mean') && ~isempty(target.P_mean);
if has_P_mean
    legacy_P_mean = target.P_mean;
else
    legacy_P_mean = [];
end

% 统一处理极小扩散系数，并以平均扩散系数计算主持久性
D_clamped = max(D_mean, min_diffusion);
P_from_D = 1 ./ sqrt(D_clamped);

%% 3. 绘制热力图 -----------------------------------------------------------
figure('Name', 'Persistence Heatmaps', 'Color', 'white', ...
    'Position', [100, 100, 1200, 520]);

% 设定统一色阶，避免视觉混淆
if has_P_mean
    compare_values = legacy_P_mean(:);
else
    compare_values = [];
end
valid_values = [P_from_D(:); compare_values];
valid_values = valid_values(isfinite(valid_values));
if isempty(valid_values)
    clim = [0, 1];
else
    sorted_vals = sort(valid_values);
    lower_idx = max(1, round(clip_percentiles(1) * numel(sorted_vals)));
    upper_idx = min(numel(sorted_vals), round(clip_percentiles(2) * numel(sorted_vals)));
    clim = [sorted_vals(lower_idx), sorted_vals(upper_idx)];
    if diff(clim) < 1e-6
        clim(2) = clim(1) + 1;
    end
end

subplot(1, 2, 1);
imagesc(cj, noise, P_from_D);
set(gca, 'YDir', 'normal');
colorbar;
caxis(clim);
xlabel('cj\_threshold');
ylabel('角度噪声强度');
title('持久性 (由平均扩散 D\_{mean} 计算)');

subplot(1, 2, 2);
if has_P_mean
    imagesc(cj, noise, legacy_P_mean);
    set(gca, 'YDir', 'normal');
    colorbar;
    caxis(clim);
    xlabel('cj\_threshold');
    ylabel('角度噪声强度');
    title('原文件提供的 P\_{mean} (参考)');
else
    imagesc(cj, noise, D_mean);
    set(gca, 'YDir', 'normal');
    colorbar;
    d_vals = D_mean(isfinite(D_mean));
    if isempty(d_vals)
        caxis([0, 1]);
    else
        caxis([min(d_vals), max(d_vals)]);
    end
    xlabel('cj\_threshold');
    ylabel('角度噪声强度');
    title('平均扩散系数 D\_{mean}');
end

sgtitle('持久性热力图对比', 'FontWeight', 'bold');

fprintf('绘图完成。\n');

fprintf('色阶范围: [%.3f, %.3f] (由 %.1f%% 到 %.1f%% 分位数给定)\n', ...
    clim(1), clim(2), clip_percentiles(1)*100, clip_percentiles(2)*100);

%% ========================================================================
function target = detect_result_struct(raw)
% detect_result_struct 在 MAT 文件中查找包含目标字段的结构体
    fields = fieldnames(raw);
    required = {'cj_thresholds', 'noise_levels', 'D_mean'};
    target = [];
    for i = 1:numel(fields)
        candidate = raw.(fields{i});
        if isstruct(candidate) && all(isfield(candidate, required))
            target = candidate;
            return;
        end
    end
end
