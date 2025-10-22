% run_persistence_scan 持久性扫描实验脚本
%
% 功能概述:
%   1. 遍历运动显著性阈值 (cj_threshold) 与角度噪声强度 (angleNoiseIntensity)
%   2. 基于 ParticleSimulation 模型执行多次仿真
%   3. 计算每组参数下的扩散系数 D_A 以及持久性指标 P = 1/sqrt(D_A)
%   4. 汇总统计结果并写入 data/experiments/persistence_scan/<时间戳>/ 目录
%
% 使用说明:
%   直接运行本脚本。可根据需要调整下方的参数设置与扫描范围。
%
% 作者：李亚男
% 日期：2025年
% MATLAB 2025a 兼容

clc;
clear;
close all;

%% 1. 全局参数设置 ---------------------------------------------------------
fprintf('=================================================\n');
fprintf('     持久性扫描实验 (无领航者)\n');
fprintf('=================================================\n\n');

config = struct();
config.num_runs_per_setting = 10;          % 每组参数的重复次数
config.enable_parallel = true;             % 默认开启并行
config.desired_workers = [];               % 为空则沿用现有并行池设置
config.burn_in_ratio = 0.2;                % 拟合扩散系数时丢弃的前期比例

% 仿真基础参数，可按项目需求调整
base_params = struct();
base_params.N = 200;
base_params.rho = 1;
base_params.v0 = 1;
base_params.angleUpdateParameter = 10;
base_params.angleNoiseIntensity = 0.1;     % 将被扫描覆盖
base_params.T_max = 800;
base_params.dt = 0.1;
base_params.radius = 5;
base_params.deac_threshold = 0.1745;
base_params.cj_threshold = 1.0;            % 将被扫描覆盖
base_params.fieldSize = 50;
base_params.initDirection = pi/4;
base_params.useFixedField = true;

% 扫描范围 (可按需调整步长与范围)
cj_thresholds = 0.0:0.1:8.0;
noise_levels = 0.0:0.1:1.0;

fprintf('扫描的 cj_threshold 个数: %d\n', numel(cj_thresholds));
fprintf('扫描的噪声水平个数: %d\n', numel(noise_levels));
fprintf('每组参数重复次数: %d\n', config.num_runs_per_setting);

% 随机数种子基线，方便结果重现
base_seed = 20250125;

%% 2. 输出目录准备 ---------------------------------------------------------
timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
output_root = fullfile('data', 'experiments', 'persistence_scan', timestamp);
fig_dir = fullfile(output_root, 'figures');
if ~exist(fig_dir, 'dir')
    mkdir(fig_dir);
end
fprintf('结果将写入: %s\n\n', output_root);

%% 3. 结果缓存变量初始化 ----------------------------------------------------
num_cj = numel(cj_thresholds);
num_noise = numel(noise_levels);

D_mean = NaN(num_noise, num_cj);
D_std = NaN(num_noise, num_cj);
P_mean = NaN(num_noise, num_cj);
P_std = NaN(num_noise, num_cj);

raw_D = cell(num_noise, num_cj);
raw_P = cell(num_noise, num_cj);

total_tasks = num_cj * num_noise;
current_task = 0;

pool = [];
if config.enable_parallel
    pool = configure_parallel_pool(config.desired_workers);
    fprintf('并行模式启用: %d workers\n\n', pool.NumWorkers);
else
    fprintf('串行模式执行。\n\n');
end

%% 4. 主循环: 遍历阈值与噪声 ------------------------------------------------
tic;
for cj_idx = 1:num_cj
    cj_value = cj_thresholds(cj_idx);
    fprintf('>>> 开始处理 cj_threshold = %.2f\n', cj_value);
    
    for noise_idx = 1:num_noise
        noise_value = noise_levels(noise_idx);
        current_task = current_task + 1;
        fprintf('    - 噪声水平 %.2f (%d/%d)\n', noise_value, current_task, total_tasks);
        
        [D_runs, P_runs] = evaluate_single_setting( ...
            base_params, cj_value, noise_value, config, base_seed);
        
        raw_D{noise_idx, cj_idx} = D_runs;
        raw_P{noise_idx, cj_idx} = P_runs;
        
        D_mean(noise_idx, cj_idx) = mean(D_runs, 'omitnan');
        D_std(noise_idx, cj_idx) = std(D_runs, 0, 'omitnan');
        P_mean(noise_idx, cj_idx) = mean(P_runs, 'omitnan');
        P_std(noise_idx, cj_idx) = std(P_runs, 0, 'omitnan');
    end
    
    fprintf('<<< 完成 cj_threshold = %.2f 的全部噪声扫描\n\n', cj_value);
end
elapsed_minutes = toc / 60;
fprintf('全部扫描完成，总耗时约 %.2f 分钟\n', elapsed_minutes);

if config.enable_parallel && ~isempty(pool)
    % 保留并行池以便后续任务继续复用；如需释放可在脚本外手动 delete(gcp('nocreate'))
end

%% 5. 结果保存 --------------------------------------------------------------
results = struct();
results.description = 'Persistence scan without external leaders';
results.timestamp = timestamp;
results.generated_at = datetime('now');
results.base_params = base_params;
results.config = config;
results.cj_thresholds = cj_thresholds;
results.noise_levels = noise_levels;
results.D_mean = D_mean;
results.D_std = D_std;
results.P_mean = P_mean;
results.P_std = P_std;
results.raw_D = raw_D;
results.raw_P = raw_P;

output_mat = fullfile(output_root, 'persistence_results.mat');
save(output_mat, 'results', '-v7.3');
fprintf('统计结果已保存至: %s\n', output_mat);

%% 6. 快速可视化 ------------------------------------------------------------
persist_fig = figure('Name', '持久性热力图', ...
    'Position', [100, 100, 1200, 500], 'Color', 'white');

subplot(1, 2, 1);
imagesc(cj_thresholds, noise_levels, P_mean);
set(gca, 'YDir', 'normal');
colorbar;
xlabel('cj\_threshold');
ylabel('角度噪声强度');
title('平均持久性 P (越大表示越稳定)');

subplot(1, 2, 2);
imagesc(cj_thresholds, noise_levels, D_mean);
set(gca, 'YDir', 'normal');
colorbar;
xlabel('cj\_threshold');
ylabel('角度噪声强度');
title('平均扩散系数 D\_A (越小越好)');

heatmap_path = fullfile(fig_dir, 'persistence_heatmap.png');
print(persist_fig, heatmap_path, '-dpng', '-r300');
savefig(persist_fig, fullfile(fig_dir, 'persistence_heatmap.fig'));
fprintf('热力图已保存至: %s\n', heatmap_path);

fprintf('\n所有任务完成。\n');

%% ========================================================================
%% 辅助函数
%% ========================================================================

function [D_runs, P_runs] = evaluate_single_setting(base_params, cj_value, noise_value, config, base_seed)
% evaluate_single_setting 对单一参数组合执行多次仿真，计算 D_A 与 P
    num_runs = config.num_runs_per_setting;
    D_runs = NaN(num_runs, 1);
    P_runs = NaN(num_runs, 1);
    
    if config.enable_parallel
        parfor run_idx = 1:num_runs %#ok<*PFBNS>
            seed = base_seed + run_idx + round(1e3 * cj_value) + round(1e4 * noise_value);
            [D_runs(run_idx), P_runs(run_idx)] = run_single_trial( ...
                base_params, cj_value, noise_value, config, seed);
        end
    else
        for run_idx = 1:num_runs
            seed = base_seed + run_idx + round(1e3 * cj_value) + round(1e4 * noise_value);
            [D_runs(run_idx), P_runs(run_idx)] = run_single_trial( ...
                base_params, cj_value, noise_value, config, seed);
        end
    end
end

function [D_value, P_value] = run_single_trial(base_params, cj_value, noise_value, config, seed)
% run_single_trial 执行单次仿真并估算扩散系数及持久性
    if nargin >= 5 && ~isempty(seed)
        rng(seed);
    end
    
    params = base_params;
    params.cj_threshold = cj_value;
    params.angleNoiseIntensity = noise_value;
    
    simulation = ParticleSimulation(params);
    
    T = simulation.T_max;
    dt = simulation.dt;
    burn_in_index = max(2, floor((T + 1) * config.burn_in_ratio));
    
    com_series = zeros(T + 1, 2);
    com_series(1, :) = mean(simulation.positions, 1);
    
    for step_idx = 1:T
        simulation.step();
        com_series(step_idx + 1, :) = mean(simulation.positions, 1);
    end
    
    displacements = com_series - com_series(1, :);
    msd = sum(displacements.^2, 2);
    time_vec = (0:T)' * dt;
    
    x = time_vec(burn_in_index:end);
    y = msd(burn_in_index:end);
    
    if numel(x) < 2 || all(abs(y - y(1)) < eps)
        D_value = eps;
    else
        coeffs = polyfit(x, y, 1);
        D_est = coeffs(1);
        if D_est < eps
            D_value = eps;
        else
            D_value = D_est;
        end
    end
    
    P_value = 1 / sqrt(D_value);
end

function pool = configure_parallel_pool(desired_workers)
% configure_parallel_pool 确保并行池可用，并按需调整 worker 数量
    if nargin < 1
        desired_workers = [];
    end
    pool = gcp('nocreate');
    if isempty(pool)
        if isempty(desired_workers)
            pool = parpool;
        else
            pool = parpool(desired_workers);
        end
    elseif ~isempty(desired_workers) && pool.NumWorkers ~= desired_workers
        delete(pool);
        if isempty(desired_workers)
            pool = parpool;
        else
            pool = parpool(desired_workers);
        end
    end
end
