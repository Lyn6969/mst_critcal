% run_cj_tradeoff_adaptive_scan
% =========================================================================
% 目的：
%   - 在固定噪声场景下比较“固定运动显著性阈值”与“自适应阈值”两种机制，
%     度量响应性 R 与持久性 P 的权衡关系。
%   - 自适应机制采用局部序参量来调节粒子的实时阈值，噪声越小阈值越低，
%     噪声越大阈值越高，从而期待在兼顾持久性的同时提升响应性。
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

cj_thresholds = 0.0:0.1:5.0;
num_runs = 50;
num_angles = 1;
pers_cfg = struct();
pers_cfg.burn_in_ratio = 0.25;
pers_cfg.min_diffusion = 1e-4;
pers_cfg.min_fit_points = 40;

adaptive_cfg = struct();
adaptive_cfg.cj_low = 0.4;
adaptive_cfg.cj_high = 2.5;
adaptive_cfg.order_threshold = 0.6;
adaptive_cfg.include_self = true;

modes = {
    struct('id', 'fixed', 'label', '固定阈值', 'useAdaptive', false, 'cfg', []), ...
    struct('id', 'adaptive', 'label', '自适应阈值', 'useAdaptive', true, 'cfg', adaptive_cfg)
};

time_vec_resp = (0:resp_params.T_max)' * resp_params.dt;
base_seed = 20250315;

results_cell = cell(numel(modes), 1);

%% 2. 依次运行两种机制 -----------------------------------------------------------
for mode_idx = 1:numel(modes)
    mode = modes{mode_idx};
    fprintf('[%s] 模式：%s\n', mode.id, mode.label);

    mode_results = run_tradeoff_mode(resp_params, pers_params, mode, ...
        cj_thresholds, num_runs, num_angles, pers_cfg, time_vec_resp, base_seed);

    results_cell{mode_idx} = mode_results;
    fprintf('  完成：平均 R 最高 %.3f，平均 P 最高 %.3f\n\n', ...
        max(mode_results.R_mean, [], 'omitnan'), max(mode_results.P_mean, [], 'omitnan'));
end

%% 3. 绘制对比权衡图 ------------------------------------------------------------
results_dir = fullfile('results', 'tradeoff');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end
timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
output_mat = fullfile(results_dir, sprintf('cj_tradeoff_adaptive_%s.mat', timestamp));
output_fig = fullfile(results_dir, sprintf('cj_tradeoff_adaptive_%s.png', timestamp));

figure('Name', '固定阈值 vs 自适应阈值', 'Color', 'white', 'Position', [160, 120, 960, 600]);
hold on;
color_list = lines(numel(modes));
markers = {'o', 's'};

for mode_idx = 1:numel(modes)
    mode = modes{mode_idx};
    data = results_cell{mode_idx};

    scatter(data.R_mean, data.P_mean, 60, color_list(mode_idx, :), markers{mode_idx}, ...
        'filled', 'DisplayName', mode.label);

    % 绘制误差棒
    for k = 1:numel(cj_thresholds)
        if isnan(data.R_mean(k)) || isnan(data.P_mean(k))
            continue;
        end
        errorbar(data.R_mean(k), data.P_mean(k), data.P_sem(k), data.P_sem(k), ...
            data.R_sem(k), data.R_sem(k), 'Color', color_list(mode_idx, :)*0.7, ...
            'LineWidth', 0.9, 'CapSize', 5);
    end

    plot(data.R_mean, data.P_mean, '-', 'Color', color_list(mode_idx, :), ...
        'LineWidth', 1.4);
end

xlabel('响应性 R');
ylabel('持久性 P');
title('自适应阈值提升响应性的同时保持持久性');
legend('Location', 'best');
grid on;

saveas(gcf, output_fig);
fprintf('图像已保存至: %s\n', output_fig);

%% 4. 保存数据 --------------------------------------------------------------------
summary = struct();
summary.description = 'Fixed vs adaptive cj threshold trade-off';
summary.timestamp = timestamp;
summary.base_params_resp = resp_params;
summary.base_params_pers = pers_params;
summary.adaptive_config = adaptive_cfg;
summary.cj_thresholds = cj_thresholds;
summary.num_runs = num_runs;
summary.mode_results = results_cell;
summary.matlab_version = version;

save(output_mat, 'summary', '-v7.3');
fprintf('数据已保存至: %s\n', output_mat);

%% -------------------------------------------------------------------------------
function mode_results = run_tradeoff_mode(resp_params, pers_params, mode, ...
    cj_thresholds, num_runs, num_angles, pers_cfg, time_vec_resp, base_seed)

    if mode.useAdaptive
        resp_params.useAdaptiveThreshold = true;
        resp_params.adaptiveThresholdConfig = mode.cfg;
        pers_params.useAdaptiveThreshold = true;
        pers_params.adaptiveThresholdConfig = mode.cfg;
    else
        resp_params.useAdaptiveThreshold = false;
        pers_params.useAdaptiveThreshold = false;
    end

    num_params = numel(cj_thresholds);
    R_raw = NaN(num_params, num_runs);
    P_raw = NaN(num_params, num_runs);
    D_raw = NaN(num_params, num_runs);
    trigger_failures = zeros(num_params, 1);

    total_timer = tic;
    for param_idx = 1:num_params
        current_cj = cj_thresholds(param_idx);
        resp_params.cj_threshold = current_cj;
        pers_params.cj_threshold = current_cj;

        for run_idx = 1:num_runs
            seed_base = base_seed + (param_idx - 1) * num_runs + run_idx;

            [R_value, triggered] = run_single_responsiveness_trial(resp_params, num_angles, time_vec_resp, seed_base);
            R_raw(param_idx, run_idx) = R_value;
            if ~triggered || isnan(R_value)
                trigger_failures(param_idx) = trigger_failures(param_idx) + 1;
            end

            [P_value, D_value] = run_single_persistence_trial(pers_params, pers_cfg, seed_base + 10000);
            P_raw(param_idx, run_idx) = P_value;
            D_raw(param_idx, run_idx) = D_value;
        end
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
