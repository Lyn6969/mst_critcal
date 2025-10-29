% run_cj_tradeoff_adaptive_scan
% =========================================================================
% 目的：
%   - 在固定噪声场景下比较“固定运动显著性阈值”扫描与“自适应显著性阈值”
%     两种机制的响应性 R 与持久性 P 的权衡关系。
%   - 自适应机制采用邻域运动显著性方差作为触发指标，与
%     visualize_adaptive_threshold 中保持一致 (adaptive_cfg.saliency_threshold)。
%
% 输出：
%   - results/tradeoff/ 下的 MAT 文件，包含两种机制的原始数据与统计量
%   - 对比权衡图 (PNG)，展示两条 R-P 曲线
%
% 使用方式：
%   直接运行本脚本即可。可根据需要调整扫描范围、噪声强度和自适应参数。
% =========================================================================

clc;
clear;
close all;

addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..')));

%% 1. 公共仿真参数 ---------------------------------------------------------------
fprintf('=================================================\n');
fprintf('   固定阈值 vs 自适应阈值：响应性-持久性权衡\n');
fprintf('=================================================\n\n');

base_common = struct();
base_common.N = 200;
base_common.rho = 1;
base_common.v0 = 1;
base_common.angleUpdateParameter = 10;
base_common.angleNoiseIntensity = 0.05; % 固定噪声水平
base_common.T_max = 600;
base_common.dt = 0.1;
base_common.radius = 5;
base_common.deac_threshold = 0.1745;
base_common.cj_threshold = 1.5;
base_common.fieldSize = 50;
base_common.initDirection = pi/4;
base_common.useFixedField = true;

resp_params = base_common;
resp_params.stabilization_steps = 200;
resp_params.forced_turn_duration = 200;

pers_params = base_common;
pers_params.T_max = 800;

cj_thresholds_fixed = 0.0:0.1:5.0;
num_runs = 50;
num_angles = 1;
pers_cfg = struct();
pers_cfg.burn_in_ratio = 0.25;
pers_cfg.min_diffusion = 1e-4;
pers_cfg.min_fit_points = 40;

adaptive_cfg = struct();
adaptive_cfg.cj_low = 0.5;
adaptive_cfg.cj_high = 5.0;
adaptive_cfg.saliency_threshold = 0.031;
adaptive_cfg.include_self = false;

modes = {
    struct('id', 'fixed', 'label', '固定阈值扫描', 'useAdaptive', false, ...
           'cfg', [], 'cj_thresholds', cj_thresholds_fixed), ...
    struct('id', 'adaptive', 'label', '自适应阈值', 'useAdaptive', true, ...
           'cfg', adaptive_cfg, 'cj_thresholds', resp_params.cj_threshold)
};

time_vec_resp = (0:resp_params.T_max)' * resp_params.dt;
base_seed = 20250315;

parallel_cfg = struct();
parallel_cfg.desired_workers = [];
pool = configure_parallel_pool(parallel_cfg.desired_workers);
fprintf('并行模式已启用：%d 个 workers\n\n', pool.NumWorkers);

results = struct();

%% 2. 依次运行两种机制 -----------------------------------------------------------
for mode_idx = 1:numel(modes)
    mode = modes{mode_idx};
    fprintf('[%s] 模式：%s\n', mode.id, mode.label);

    mode_results = run_tradeoff_mode(resp_params, pers_params, mode, ...
        num_runs, num_angles, pers_cfg, time_vec_resp, base_seed);

    results.(mode.id) = mode_results;

    fprintf('  完成：平均 R = %.3f, 平均 P = %.3f\n\n', ...
        mean(mode_results.R_mean, 'omitnan'), mean(mode_results.P_mean, 'omitnan'));
end

%% 3. 绘制对比权衡图 ------------------------------------------------------------
results_dir = fullfile('results', 'tradeoff');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end
timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
output_mat = fullfile(results_dir, sprintf('cj_tradeoff_adaptive_%s.mat', timestamp));
output_fig = fullfile(results_dir, sprintf('cj_tradeoff_adaptive_%s.png', timestamp));

figure('Name', '固定 vs 自适应阈值权衡', 'Color', 'white', 'Position', [160, 120, 960, 600]);
hold on;

% 固定阈值散点（颜色表示阈值大小）
fixed_data = results.fixed;
fixed_cj = fixed_data.cj_thresholds;
scatter(fixed_data.R_mean, fixed_data.P_mean, 70, fixed_cj, 'filled', 'DisplayName', '固定阈值');
colormap('turbo');
cb = colorbar;
cb.Label.String = '固定阈值大小';
plot(fixed_data.R_mean, fixed_data.P_mean, '-', 'Color', [0.3 0.3 0.3], 'LineWidth', 1.2);

% 自适应阈值结果（单点）
adaptive_data = results.adaptive;
adaptive_R = adaptive_data.R_mean;
adaptive_P = adaptive_data.P_mean;
adaptive_point = scatter(adaptive_R, adaptive_P, 160, [0.85 0.2 0.2], 'filled', ...
    'DisplayName', sprintf('自适应阈值 %.3f', adaptive_cfg.saliency_threshold));
adaptive_point.Marker = 'p';
adaptive_point.MarkerEdgeColor = [0.5 0 0];
adaptive_point.MarkerFaceColor = [0.85 0.2 0.2];

xlabel('响应性 R');
ylabel('持久性 P');
title('固定阈值扫描 vs 自适应显著性阈值');
legend('Location', 'best');
grid on;

saveas(gcf, output_fig);
fprintf('图像已保存至: %s\n', output_fig);

%% 4. 保存数据 --------------------------------------------------------------------
summary = struct();
summary.description = 'Fixed threshold scan vs adaptive saliency-threshold trade-off';
summary.timestamp = timestamp;
summary.base_params_resp = resp_params;
summary.base_params_pers = pers_params;
summary.adaptive_config = adaptive_cfg;
summary.cj_thresholds_fixed = cj_thresholds_fixed;
summary.num_runs = num_runs;
summary.results = results;
summary.matlab_version = version;
summary.parallel_config = parallel_cfg;

save(output_mat, 'summary', '-v7.3');
fprintf('数据已保存至: %s\n', output_mat);

%% -------------------------------------------------------------------------------
function mode_results = run_tradeoff_mode(resp_params, pers_params, mode, ...
    num_runs, num_angles, pers_cfg, time_vec_resp, base_seed)

    if mode.useAdaptive
        resp_params.useAdaptiveThreshold = true;
        resp_params.adaptiveThresholdConfig = mode.cfg;
        pers_params.useAdaptiveThreshold = true;
        pers_params.adaptiveThresholdConfig = mode.cfg;
    else
        resp_params.useAdaptiveThreshold = false;
        pers_params.useAdaptiveThreshold = false;
    end

    cj_thresholds = mode.cj_thresholds;
    num_params = numel(cj_thresholds);
    R_raw = NaN(num_params, num_runs);
    P_raw = NaN(num_params, num_runs);
    D_raw = NaN(num_params, num_runs);
    trigger_failures = zeros(num_params, 1);

    fprintf('  -> 共扫描 %d 个阈值，每个重复 %d 次。\n', num_params, num_runs);

    total_timer = tic;
    completed = 0;
    start_time = tic;

    progress_queue = parallel.pool.DataQueue;
    afterEach(progress_queue, @(idx) progress_callback(idx));

    parfor param_idx = 1:num_params
        current_cj = cj_thresholds(param_idx);

        resp_local = resp_params;
        pers_local = pers_params;
        resp_local.cj_threshold = current_cj;
        pers_local.cj_threshold = current_cj;

        local_R = NaN(1, num_runs);
        local_P = NaN(1, num_runs);
        local_D = NaN(1, num_runs);
        local_fail = false(1, num_runs);

        for run_idx = 1:num_runs
            seed_base = base_seed + (param_idx - 1) * num_runs + run_idx;

            [R_value, triggered] = run_single_responsiveness_trial(resp_local, num_angles, time_vec_resp, seed_base);
            local_R(run_idx) = R_value;
            local_fail(run_idx) = (~triggered || isnan(R_value));

            [P_value, D_value] = run_single_persistence_trial(pers_local, pers_cfg, seed_base + 10000);
            local_P(run_idx) = P_value;
            local_D(run_idx) = D_value;
        end

        R_raw(param_idx, :) = local_R;
        P_raw(param_idx, :) = local_P;
        D_raw(param_idx, :) = local_D;
        trigger_failures(param_idx) = sum(local_fail);

        send(progress_queue, param_idx);
    end

    fprintf('  模式 [%s] 总耗时 %.1f 分钟\n', mode.id, toc(total_timer) / 60);

    mode_results = struct();
    mode_results.mode = mode;
    mode_results.R_raw = R_raw;
    mode_results.P_raw = P_raw;
    mode_results.D_raw = D_raw;
    mode_results.trigger_failures = trigger_failures;

    mode_results.R_mean = mean(R_raw, 2, 'omitnan');
    mode_results.R_std = std(R_raw, 0, 2, 'omitnan');
    mode_results.R_sem = mode_results.R_std ./ sqrt(num_runs);

    mode_results.P_mean = mean(P_raw, 2, 'omitnan');
    mode_results.P_std = std(P_raw, 0, 2, 'omitnan');
    mode_results.P_sem = mode_results.P_std ./ sqrt(num_runs);

    mode_results.D_mean = mean(D_raw, 2, 'omitnan');
    mode_results.cj_thresholds = cj_thresholds(:);
    if mode.useAdaptive && isfield(mode.cfg, 'saliency_threshold')
        mode_results.saliency_threshold = mode.cfg.saliency_threshold;
    end

    function progress_callback(param_idx_completed)
        completed = completed + 1;
        elapsed = toc(start_time);
        avg_time = elapsed / completed;
        remaining = avg_time * max(num_params - completed, 0);
        fprintf('    [%.0f%%] (%d/%d) 阈值 %.3f 完成 | 已用 %.1f 分钟 | 预计剩余 %.1f 分钟\n', ...
            100 * completed / num_params, completed, num_params, cj_thresholds(param_idx_completed), ...
            elapsed / 60, remaining / 60);
    end
end

function pool = configure_parallel_pool(desired_workers)
    pool = gcp('nocreate');
    if isempty(pool)
        if isempty(desired_workers)
            pool = parpool;
        else
            pool = parpool(desired_workers);
        end
    elseif ~isempty(desired_workers) && pool.NumWorkers ~= desired_workers
        delete(pool);
        pool = parpool(desired_workers);
    end
end

function [R_value, triggered] = run_single_responsiveness_trial(params, num_angles, time_vec, seed)
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
            if num_angles <= 1
                phi_list = target_theta;
            else
                phi_offsets = linspace(0, pi, num_angles);
                phi_list = target_theta + phi_offsets;
            end
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

function [P_value, D_value] = run_single_persistence_trial(params, cfg, seed)
    rng(seed);
    sim = ParticleSimulation(params);

    T = sim.T_max;
    dt = sim.dt;
    burn_in_index = max(2, floor((T + 1) * cfg.burn_in_ratio));

    initial_positions = sim.positions;
    centroid0 = mean(initial_positions, 1);
    offsets0 = initial_positions - centroid0;

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

    time_vec = (0:T)' * dt;
    x = time_vec(burn_in_index:end);
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
