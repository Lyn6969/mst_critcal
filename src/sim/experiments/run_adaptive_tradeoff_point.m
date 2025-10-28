% run_adaptive_tradeoff_point
% =========================================================================
% 目的：
%   - 对比“固定运动显著性阈值”与“自适应阈值”在相同噪声条件下的单组实验。
%   - 每种模式重复 num_runs 次，统计响应性 R 与持久性 P 的均值和标准误差。
%   - 最终仅绘制两个统计点在 R–P 平面上的相对位置，便于判断自适应策略是否改进权衡。
%
% 使用方式：
%   直接运行脚本。可根据需要调整噪声、粒子数、重复次数、
%   以及自适应阈值策略参数（显著性方差阈值、cj_low/cj_high 等）。
% =========================================================================

clc;
clear;
close all;

%% 1. 基础参数配置 ---------------------------------------------------------------
fprintf('=================================================\n');
fprintf('   固定阈值 vs 自适应阈值：单点 R-P 对比\n');
fprintf('=================================================\n\n');

params_common = struct();
params_common.N = 200;
params_common.rho = 1;
params_common.v0 = 1;
params_common.angleUpdateParameter = 10;
params_common.angleNoiseIntensity = 0.05;
params_common.T_max = 600;
params_common.dt = 0.1;
params_common.radius = 5;
params_common.deac_threshold = 0.1745;
params_common.cj_threshold = 1.5;
params_common.fieldSize = 50;
params_common.initDirection = pi/4;
params_common.useFixedField = true;

resp_params = params_common;
resp_params.stabilization_steps = 200;
resp_params.forced_turn_duration = 200;

pers_params = params_common;
pers_params.T_max = 800;

num_runs = 50;
num_angles = 1;
pers_cfg = struct();
pers_cfg.burn_in_ratio = 0.25;
pers_cfg.min_fit_points = 40;
pers_cfg.min_diffusion = 1e-4;

adaptive_cfg = struct();
adaptive_cfg.cj_low = 0.5;
adaptive_cfg.cj_high = 5.0;
adaptive_cfg.saliency_threshold = 0.03;
adaptive_cfg.include_self = false;

time_vec_resp = (0:resp_params.T_max)' * resp_params.dt;
base_seed = 20250325;

modes = {
    struct('id', 'fixed', 'label', '固定阈值', 'useAdaptive', false, 'cfg', []), ...
    struct('id', 'adaptive', 'label', '自适应阈值', 'useAdaptive', true, 'cfg', adaptive_cfg)
};

results = struct();

%% 2. 依次运行固定/自适应两种模式 ----------------------------------------------
for idx = 1:numel(modes)
    mode = modes{idx};
    fprintf('[%s] 模式：%s\n', mode.id, mode.label);

    resp_local = resp_params;
    pers_local = pers_params;
    if mode.useAdaptive
        resp_local.useAdaptiveThreshold = true;
        resp_local.adaptiveThresholdConfig = mode.cfg;
        pers_local.useAdaptiveThreshold = true;
        pers_local.adaptiveThresholdConfig = mode.cfg;
    else
        resp_local.useAdaptiveThreshold = false;
        pers_local.useAdaptiveThreshold = false;
    end

    rng(mode_index_seed(base_seed, idx));

    R_values = NaN(num_runs, 1);
    P_values = NaN(num_runs, 1);
    trigger_fail = 0;

    parfor run_idx = 1:num_runs
        seed = base_seed + idx * 1e5 + run_idx;

        [R_val, triggered] = run_single_responsiveness(resp_local, num_angles, time_vec_resp, seed);
        if ~triggered || isnan(R_val)
            trigger_fail = trigger_fail + 1;
        end
        R_values(run_idx) = R_val;

        [P_val, ~] = run_single_persistence(pers_local, pers_cfg, seed + 12345);
        P_values(run_idx) = P_val;
    end

    mode_result = struct();
    mode_result.mode = mode;
    mode_result.R_values = R_values;
    mode_result.P_values = P_values;
    mode_result.trigger_failures = trigger_fail;
    mode_result.R_mean = mean(R_values, 'omitnan');
    mode_result.R_std = std(R_values, 'omitnan');
    mode_result.R_sem = mode_result.R_std / sqrt(sum(~isnan(R_values)));
    mode_result.P_mean = mean(P_values, 'omitnan');
    mode_result.P_std = std(P_values, 'omitnan');
    mode_result.P_sem = mode_result.P_std / sqrt(sum(~isnan(P_values)));

    results.(mode.id) = mode_result;
    fprintf('  完成：R=%.3f±%.3f, P=%.3f±%.3f (触发失败 %d 次)\n\n', ...
        mode_result.R_mean, mode_result.R_sem, mode_result.P_mean, mode_result.P_sem, trigger_fail);
end

%% 3. 绘制单点 Trade-off 图 ------------------------------------------------------
figure('Color', 'white', 'Position', [200, 160, 640, 480]);
hold on;

colors = lines(numel(modes));
markers = {'o', 's'};
for idx = 1:numel(modes)
    mode = modes{idx};
    data = results.(mode.id);
    errorbar(data.R_mean, data.P_mean, data.P_sem, data.P_sem, data.R_sem, data.R_sem, ...
        'LineStyle', 'none', 'Color', colors(idx, :), 'CapSize', 6);
    scatter(data.R_mean, data.P_mean, 80, colors(idx, :), markers{idx}, 'filled', ...
        'DisplayName', mode.label);
end

xlabel('响应性 R');
ylabel('持久性 P');
title('固定阈值 vs 自适应阈值：单点 Trade-off');
legend('Location', 'best');
grid on;

out_dir = fullfile('results', 'tradeoff');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end
stamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
fig_path = fullfile(out_dir, sprintf('adaptive_tradeoff_point_%s.png', stamp));
print(fig_path, '-dpng', '-r200');
fprintf('图像已保存至: %s\n', fig_path);

mat_path = fullfile(out_dir, sprintf('adaptive_tradeoff_point_%s.mat', stamp));
save(mat_path, 'results', 'params_common', 'pers_cfg', 'adaptive_cfg', 'num_runs');
fprintf('数据已保存至: %s\n', mat_path);

%% -------------------------------------------------------------------------------
function seed = mode_index_seed(base_seed, idx)
    seed = base_seed + idx * 5e4;
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
    sim = ParticleSimulation(params);

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
