% scan_saliency_threshold 扫描自适应显著性阈值的响应性/持久性
% =========================================================================
% 目标：
%   - 复用 visualize_adaptive_threshold 的仿真环境。
%   - 遍历 adaptive_cfg.saliency_threshold ∈ 0.01:0.001:0.05。
%   - 每个阈值运行 50 次，计算响应性 R 与持久性 P 的均值/标准差。
%   - 对 P 进行归一化（除以所有阈值下 P_mean 的最大值），便于比较。
%
% 输出：
%   - 命令行打印结果表格。
%   - 绘制 R 与归一化 P 随阈值变化的曲线。
%   - 可选保存 .mat 与 .png（默认开启）。
%
% 要求：
%   - MATLAB Parallel Computing Toolbox（若未开启 parpool 亦可串行执行）。
% =========================================================================

clc;
clear;
close all;
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..')));
fprintf('=================================================\n');
fprintf('   自适应显著性阈值扫描 (R / P)\n');
fprintf('=================================================\n\n');

%% 1. 共享仿真参数（同 visualize_adaptive_threshold） ---------------------------
params = struct();
params.N = 200;
params.rho = 1;
params.v0 = 1;
params.angleUpdateParameter = 10;
params.angleNoiseIntensity = 0.05;
params.T_max = 600;
params.dt = 0.1;
params.radius = 5;
params.deac_threshold = 0.1745;
params.cj_threshold = 1.5;
params.fieldSize = 50;
params.initDirection = pi/4;
params.useFixedField = true;
params.stabilization_steps = 200;
params.forced_turn_duration = 400;
params.useAdaptiveThreshold = true;

adaptive_cfg = struct();
adaptive_cfg.cj_low = 0.5;
adaptive_cfg.cj_high = 5.0;
adaptive_cfg.include_self = false;
% saliency_threshold 由扫描循环内设置

pers_cfg = struct();
pers_cfg.burn_in_ratio = 0.25;
pers_cfg.min_fit_points = 40;
pers_cfg.min_diffusion = 1e-4;

num_angles = 1;
time_vec = (0:params.T_max)' * params.dt;

threshold_list = 0.01:0.001:0.05;
num_thresholds = numel(threshold_list);
runs_per_threshold = 50;
base_seed = 20250405;

%% 2. 预分配结果容器 ------------------------------------------------------------
R_mean = zeros(num_thresholds, 1);
R_std = zeros(num_thresholds, 1);
P_mean = zeros(num_thresholds, 1);
P_std = zeros(num_thresholds, 1);
trigger_failures = zeros(num_thresholds, 1);

%% 3. 并行扫描 -------------------------------------------------------------------
fprintf('开始扫描 %d 个显著性阈值，每个运行 %d 次...\n', num_thresholds, runs_per_threshold);
dq = parallel.pool.DataQueue;
progressPrinter(0, num_thresholds, threshold_list); % 重置进度
afterEach(dq, @(idx) progressPrinter(idx, num_thresholds, threshold_list));

parfor t_idx = 1:num_thresholds
    sal_thr = threshold_list(t_idx);

    params_local = params;
    cfg_local = adaptive_cfg;
    cfg_local.saliency_threshold = sal_thr;
    params_local.adaptiveThresholdConfig = cfg_local;

    R_vals = NaN(runs_per_threshold, 1);
    P_vals = NaN(runs_per_threshold, 1);
    fail_count = 0;

    for run_idx = 1:runs_per_threshold
        seed = base_seed + t_idx * 1e6 + run_idx * 97;

        [R_val, triggered] = run_single_responsiveness(params_local, num_angles, time_vec, seed);
        if ~triggered || isnan(R_val)
            fail_count = fail_count + 1;
        end
        R_vals(run_idx) = R_val;

        [P_val, ~] = run_single_persistence(params_local, pers_cfg, seed + 17);
        P_vals(run_idx) = P_val;
    end

    R_mean(t_idx) = mean(R_vals, 'omitnan');
    R_std(t_idx) = std(R_vals, 'omitnan');
    P_mean(t_idx) = mean(P_vals, 'omitnan');
    P_std(t_idx) = std(P_vals, 'omitnan');
    trigger_failures(t_idx) = fail_count;

    send(dq, t_idx);
end

%% 4. P 归一化 -------------------------------------------------------------------
max_P = max(P_mean, [], 'omitnan');
if max_P > 0
    P_mean_norm = P_mean / max_P;
else
    P_mean_norm = NaN(size(P_mean));
end

%% 5. 打印结果表 -----------------------------------------------------------------
T = table(threshold_list', R_mean, R_std, P_mean, P_std, P_mean_norm, trigger_failures, ...
    'VariableNames', {'saliency_threshold', 'R_mean', 'R_std', 'P_mean', 'P_std', 'P_mean_norm', 'trigger_fail'});
disp(T);

%% 6. 绘图 ----------------------------------------------------------------------
figure('Color', 'white', 'Position', [180, 120, 720, 540]);
yyaxis left;
errorbar(threshold_list, R_mean, R_std, 'o-', 'LineWidth', 1.4, 'DisplayName', '响应性 R');
ylabel('响应性 R');

yyaxis right;
errorbar(threshold_list, P_mean_norm, P_std / max_P, 's-', 'LineWidth', 1.4, ...
    'DisplayName', '归一化持久性 P');
ylabel('归一化持久性 P');

xlabel('显著性阈值');
title('自适应显著性阈值扫描');
grid on;
legend('Location', 'best');

%% 7. 保存结果 -------------------------------------------------------------------
results_dir = fullfile('results', 'adaptive_threshold_scan');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end
timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
mat_file = fullfile(results_dir, sprintf('saliency_scan_%s.mat', timestamp));
png_file = fullfile(results_dir, sprintf('saliency_scan_%s.png', timestamp));

save(mat_file, 'threshold_list', 'R_mean', 'R_std', 'P_mean', 'P_std', 'P_mean_norm', ...
    'trigger_failures', 'params', 'adaptive_cfg', 'pers_cfg', 'runs_per_threshold', 'base_seed');
print(png_file, '-dpng', '-r200');

fprintf('结果已保存: %s\n', mat_file);
fprintf('图像已保存: %s\n', png_file);

%% ============================================================================
% 辅助函数
% ============================================================================
function progressPrinter(idx, total, threshold_list)
    persistent count;
    if idx == 0
        count = 0;
        fprintf('进度初始化完成。\n');
        return;
    end
    count = count + 1;
    thr_val = threshold_list(idx);
    fprintf('  [%3d/%3d] 完成阈值 %.3f\n', count, total, thr_val);
end

function [R_value, triggered] = run_single_responsiveness(params, num_angles, time_vec, seed)
    rng(seed);
    sim = ParticleSimulationWithExternalPulse(params);
    sim.external_pulse_count = 1;
    sim.resetCascadeTracking();
    sim.initializeParticles();

    V_history = zeros(params.T_max + 1, 2);
    V_history(1, :) = compute_average_velocity(sim.theta, params.v0);

    projection_history = zeros(params.T_max + 1, num_angles);
    triggered = false;
    n_vectors = [];
    t_start = NaN;

    for t = 1:params.T_max
        sim.step();
        V_history(t + 1, :) = compute_average_velocity(sim.theta, params.v0);

        if ~triggered && sim.external_pulse_triggered
            triggered = true;
            t_start = t;
            leader_idx = find(sim.isExternallyActivated, 1, 'first');
            if isempty(leader_idx)
                leader_idx = 1;
            end
            target_theta = sim.external_target_theta(leader_idx);
            phi_list = repmat(target_theta, 1, num_angles);
            n_vectors = [cos(phi_list); sin(phi_list)];
        end

        if triggered
            projection_history(t + 1, :) = V_history(t + 1, :) * n_vectors;
        end
    end

    if ~triggered || isnan(t_start)
        R_value = NaN;
        return;
    end

    v0 = params.v0;
    T_window = params.forced_turn_duration;
    t_end = min(t_start + T_window, params.T_max);

    r_history = NaN(num_angles, 1);
    for angle_idx = 1:num_angles
        proj = projection_history(:, angle_idx);
        integral_value = trapz(time_vec(t_start+1:t_end+1), proj(t_start+1:t_end+1));
        duration = time_vec(t_end+1) - time_vec(t_start+1);
        if duration > 0
            r_history(angle_idx) = integral_value / (v0 * duration);
        end
    end

    R_value = mean(r_history(~isnan(r_history)));
end

function [P_value, D_value] = run_single_persistence(params, cfg, seed)
    rng(seed);
    params_pers = params;
    if isfield(params_pers, 'stabilization_steps')
        params_pers = rmfield(params_pers, 'stabilization_steps');
    end
    if isfield(params_pers, 'forced_turn_duration')
        params_pers = rmfield(params_pers, 'forced_turn_duration');
    end
    sim = ParticleSimulation(params_pers);

    T = sim.T_max;
    dt = sim.dt;
    burn_in_index = max(2, floor((T + 1) * cfg.burn_in_ratio));

    init_pos = sim.positions;
    centroid0 = mean(init_pos, 1);
    offsets0 = init_pos - centroid0;

    msd = zeros(T + 1, 1);
    msd(1) = 0;

    for step_idx = 1:T
        sim.step();
        positions = sim.positions;
        centroid = mean(positions, 1);
        centered = positions - centroid;
        rel_disp = centered - offsets0;
        msd(step_idx + 1) = mean(sum(rel_disp.^2, 2), 'omitnan');
    end

    time_vec_pers = (0:T)' * dt;
    x = time_vec_pers(burn_in_index:end);
    y = msd(burn_in_index:end);

    if numel(x) < max(2, cfg.min_fit_points) || all(abs(y - y(1)) < eps)
        D_value = NaN;
    else
        x_shift = x - x(1);
        y_shift = y - y(1);
        if any(x_shift > 0) && any(abs(y_shift) > eps)
            smooth_window = max(5, floor(numel(y_shift) * 0.1));
            if smooth_window > 1
                y_shift = smoothdata(y_shift, 'movmean', smooth_window);
            end
            slope = lsqnonneg(x_shift(:), y_shift(:));
            if slope <= 0
                D_value = NaN;
            else
                D_value = slope;
            end
        else
            D_value = NaN;
        end
    end

    if isnan(D_value)
        P_value = NaN;
    else
        D_value = max(D_value, cfg.min_diffusion);
        P_value = 1 / sqrt(D_value);
    end
end

function V = compute_average_velocity(theta, v0)
    V = [mean(v0 * cos(theta)), mean(v0 * sin(theta))];
end
