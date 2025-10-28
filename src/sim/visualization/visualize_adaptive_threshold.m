% visualize_adaptive_threshold 自适应阈值机制演示脚本
% =========================================================================
% 功能：
%   - 启用局部熵驱动的自适应运动显著性阈值，直接观察单次响应过程。
%   - 左上显示粒子运动，右下同时绘制自适应阈值和局部熵随时间的变化。
%   - 结束后自动运行一次持久性测试，输出响应性 R 与持久性 P 的数值。
%
% 使用方式：
%   直接运行本脚本即可。如需调节噪声、阈值上下限或熵阈值，在脚本顶部修改参数。
% =========================================================================

clc;
clear;
close all;

%% 1. 参数设定 ---------------------------------------------------------------
fprintf('=================================================\n');
fprintf('   自适应阈值机制可视化\n');
fprintf('=================================================\n\n');

params = struct();
params.N = 200;
params.rho = 1;
params.v0 = 1;
params.angleUpdateParameter = 10;
params.angleNoiseIntensity = 0.05;
params.T_max = 400;
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
adaptive_cfg.cj_low = 0.4;
adaptive_cfg.cj_high = 5;
adaptive_cfg.entropy_bins = 16;
adaptive_cfg.entropy_threshold_low = 0.25;
adaptive_cfg.entropy_threshold_high = 0.55;
adaptive_cfg.min_neighbors = 7;
adaptive_cfg.include_self = true;
adaptive_cfg.smoothing_weight = 0.7;
adaptive_cfg.fallback_entropy = 0.9;
params.adaptiveThresholdConfig = adaptive_cfg;

pers_cfg = struct();
pers_cfg.burn_in_ratio = 0.25;
pers_cfg.min_fit_points = 40;
pers_cfg.min_diffusion = 1e-4;

num_angles = 1;         % 仅沿领导方向计算 r(phi)
base_seed = 20250326;   % 用固定种子保证可复现

time_vec = (0:params.T_max)' * params.dt;

%% 2. 响应性演示 ---------------------------------------------------------------
rng(base_seed);
result = run_visual_responsiveness(params, num_angles, time_vec);

fprintf('响应性 R = %.3f\n', result.R_value);

%% 3. 持久性测量 ---------------------------------------------------------------
rng(base_seed + 12345);
[P_value, D_value] = run_single_persistence(params, pers_cfg);
fprintf('持久性 P = %.3f (扩散系数 %.3e)\n', P_value, D_value);

%% -------------------------------------------------------------------------------
function result = run_visual_responsiveness(params, num_angles, time_vec)
    sim = ParticleSimulationWithExternalPulse(params);
    sim.external_pulse_count = 1;
    sim.resetCascadeTracking();
    sim.initializeParticles();

    fig = figure('Name', '自适应阈值响应演示', 'Color', 'white', 'Position', [80, 120, 720, 620]);

    ax_particles = subplot(2,1,1, 'Parent', fig);
    scatter_plot = scatter(ax_particles, sim.positions(:,1), sim.positions(:,2), 36, sim.theta, 'filled');
    hold(ax_particles, 'on');
    quiver_plot = quiver(ax_particles, sim.positions(:,1), sim.positions(:,2), cos(sim.theta), sin(sim.theta), 0.4, ...
        'Color', [0.3 0.3 0.3 0.6]);
    hold(ax_particles, 'off');
    axis(ax_particles, [0 params.fieldSize 0 params.fieldSize]);
    axis(ax_particles, 'square');
    grid(ax_particles, 'on');
    title(ax_particles, '粒子状态（自适应阈值）');
    xlabel(ax_particles, 'X');
    ylabel(ax_particles, 'Y');

    ax_threshold = subplot(2,1,2, 'Parent', fig);
    hold(ax_threshold, 'on');
    threshold_line = plot(ax_threshold, NaN, NaN, '-', 'LineWidth', 1.6, 'Color', [0.9 0.5 0.2]);
    entropy_line = plot(ax_threshold, NaN, NaN, '--', 'LineWidth', 1.2, 'Color', [0.4 0.4 0.4]);
    hold(ax_threshold, 'off');
    xlabel(ax_threshold, '时间步');
    ylabel(ax_threshold, '平均阈值 / 平均熵');
    legend(ax_threshold, {'阈值', '局部熵'}, 'Location', 'best');
    title(ax_threshold, '自适应阈值与局部熵演化');
    grid(ax_threshold, 'on');

    V_history = zeros(params.T_max + 1, 2);
    V_history(1, :) = compute_average_velocity(sim.theta, params.v0);
    projection_history = zeros(params.T_max + 1, num_angles);
    threshold_history = zeros(params.T_max + 1, 1);
    entropy_history = zeros(params.T_max + 1, 1);

    triggered = false;
    n_vectors = [];
    t_start = NaN;

    for t = 1:params.T_max
        sim.step();
        V_history(t + 1, :) = compute_average_velocity(sim.theta, params.v0);

        threshold_history(t + 1) = mean(sim.cj_threshold_dynamic);
        entropy_history(t + 1) = mean(sim.local_entropy_state);

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

        if mod(t, 5) == 0
            colors = repmat([0.75 0.75 0.75], params.N, 1);
            if any(sim.isExternallyActivated)
                colors(sim.isExternallyActivated, :) = repmat([0.9 0.2 0.2], sum(sim.isExternallyActivated), 1);
            end
            set(scatter_plot, 'XData', sim.positions(:,1), 'YData', sim.positions(:,2), 'CData', colors);
            set(quiver_plot, 'XData', sim.positions(:,1), 'YData', sim.positions(:,2), ...
                'UData', cos(sim.theta), 'VData', sin(sim.theta));

            set(threshold_line, 'XData', 0:t, 'YData', threshold_history(1:t+1));
            set(entropy_line, 'XData', 0:t, 'YData', entropy_history(1:t+1));

            drawnow limitrate;
        end
    end

    if ~triggered || isnan(t_start)
        warning('未检测到外源触发，响应性记为 NaN。');
        R_val = NaN;
    else
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
        R_val = mean(r_history(~isnan(r_history)));
    end

    text(ax_particles, 1, params.fieldSize + 3, sprintf('R = %.3f', R_val), 'Color', [0.9 0.2 0.2], ...
        'FontSize', 12, 'FontWeight', 'bold');

    result = struct();
    result.R_value = R_val;
end

function [P_value, D_value] = run_single_persistence(params, cfg)
    params_pers = params;
    remove_fields = {'stabilization_steps', 'forced_turn_duration'};
    for k = 1:numel(remove_fields)
        if isfield(params_pers, remove_fields{k})
            params_pers = rmfield(params_pers, remove_fields{k});
        end
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
