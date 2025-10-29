% visualize_adaptive_threshold 自适应阈值机制演示脚本
% =========================================================================
% 功能：
%   - 启用邻域运动显著性方差驱动的自适应阈值，直接观察单次响应过程。
%   - 响应性部分：展示外源触发下的粒子动态、阈值与显著性方差演化曲线。
%   - 持久性部分：展示粒子群质心轨迹与 MSD 拟合过程，直观理解 P 的计算。
%
% 使用方式：
%   直接运行本脚本即可。如需调节噪声、阈值上下限或序参量阈值，在脚本顶部修改参数。
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
adaptive_cfg.cj_high = 5;
adaptive_cfg.saliency_threshold = 0.031;  % 邻域显著性方差阈值
adaptive_cfg.include_self = true;
params.adaptiveThresholdConfig = adaptive_cfg;

pers_cfg = struct();
pers_cfg.burn_in_ratio = 0.25;
pers_cfg.min_fit_points = 40;
pers_cfg.min_diffusion = 1e-4;

num_angles = 1;         % 仅沿领导方向计算 r(phi)
base_seed = 20250319;   % 用固定种子保证可复现

time_vec = (0:params.T_max)' * params.dt;

%% 2. 响应性演示 ---------------------------------------------------------------
rng(base_seed);
result = run_visual_responsiveness(params, num_angles, time_vec);

fprintf('响应性 R = %.3f\n', result.R_value);

%% 3. 持久性测量 ---------------------------------------------------------------
rng(base_seed + 12345);
pers_result = run_visual_persistence(params, pers_cfg);
fprintf('持久性 P = %.3f (扩散系数 %.3e)\n', pers_result.P_value, pers_result.D_value);

%% -------------------------------------------------------------------------------
function result = run_visual_responsiveness(params, num_angles, time_vec)
    sim = ParticleSimulationWithExternalPulse(params);
    sim.external_pulse_count = 1;
    sim.resetCascadeTracking();
    sim.initializeParticles();

    fig = figure('Name', '自适应阈值响应演示', 'Color', 'white', 'Position', [80, 120, 720, 620]);

    ax_particles = subplot(2,1,1, 'Parent', fig);
    scatter_plot = scatter(ax_particles, sim.positions(:,1), sim.positions(:,2), 36, 'filled');
    set(scatter_plot, 'CData', repmat([0.75 0.75 0.75], params.N, 1));
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
    saliency_line = plot(ax_threshold, NaN, NaN, '--', 'LineWidth', 1.2, 'Color', [0.4 0.4 0.4]);
    hold(ax_threshold, 'off');
    xlabel(ax_threshold, '时间步');
    ylabel(ax_threshold, '平均阈值 / 显著性方差');
    legend(ax_threshold, {'阈值', '显著性方差'}, 'Location', 'best');
    title(ax_threshold, '自适应阈值与显著性方差演化');
    grid(ax_threshold, 'on');

    V_history = zeros(params.T_max + 1, 2);
    V_history(1, :) = compute_average_velocity(sim.theta, params.v0);
    projection_history = zeros(params.T_max + 1, num_angles);
    threshold_history = zeros(params.T_max + 1, 1);
    saliency_history = zeros(params.T_max + 1, 1);

    triggered = false;
    n_vectors = [];
    t_start = NaN;

    for t = 1:params.T_max
        sim.step();
        V_history(t + 1, :) = compute_average_velocity(sim.theta, params.v0);

        threshold_history(t + 1) = mean(sim.cj_threshold_dynamic);
        saliency_history(t + 1) = mean(sim.local_saliency_state);

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
            set(saliency_line, 'XData', 0:t, 'YData', saliency_history(1:t+1));

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

function result = run_visual_persistence(params, cfg)
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

    initial_positions = sim.positions;
    centroid0 = mean(initial_positions, 1);
    offsets0 = initial_positions - centroid0;

    centroids = zeros(T + 1, 2);
    centroids(1, :) = centroid0;

    msd_history = zeros(T + 1, 1);
    msd_history(1) = 0;
    time_vec = (0:T)' * dt;

    % 新增：追踪阈值和显著性方差历史
    threshold_history = zeros(T + 1, 1);
    saliency_history = zeros(T + 1, 1);
    if sim.useAdaptiveThreshold && ~isempty(sim.cj_threshold_dynamic)
        threshold_history(1) = mean(sim.cj_threshold_dynamic);
        saliency_history(1) = mean(sim.local_saliency_state);
    else
        threshold_history(1) = sim.cj_threshold;
        saliency_history(1) = 0;
    end

    fig = figure('Name', '自适应阈值持久性演示', 'Color', 'white', 'Position', [840, 120, 720, 880]);

    ax_particles = subplot(3,1,1, 'Parent', fig);
    scatter_plot = scatter(ax_particles, sim.positions(:,1), sim.positions(:,2), 36, 'filled');
    hold(ax_particles, 'on');
    quiver_plot = quiver(ax_particles, sim.positions(:,1), sim.positions(:,2), cos(sim.theta), sin(sim.theta), 0.35, ...
        'Color', [0.3 0.3 0.3 0.5]);
    centroid_trace = plot(ax_particles, centroids(1,1), centroids(1,2), '-', 'Color', [0.2 0.4 0.8], 'LineWidth', 1.2);
    hold(ax_particles, 'off');
    axis(ax_particles, [0 params.fieldSize 0 params.fieldSize]);
    axis(ax_particles, 'square');
    grid(ax_particles, 'on');
    xlabel(ax_particles, 'X');
    ylabel(ax_particles, 'Y');
    title(ax_particles, '粒子状态与质心轨迹');

    % 新增：阈值和显著性方差演化图
    ax_threshold = subplot(3,1,2, 'Parent', fig);
    hold(ax_threshold, 'on');
    threshold_line = plot(ax_threshold, time_vec, threshold_history, '-', 'LineWidth', 1.6, 'Color', [0.9 0.5 0.2]);
    saliency_line = plot(ax_threshold, time_vec, saliency_history, '--', 'LineWidth', 1.2, 'Color', [0.4 0.4 0.4]);
    hold(ax_threshold, 'off');
    grid(ax_threshold, 'on');
    xlabel(ax_threshold, '时间 (s)');
    ylabel(ax_threshold, '平均阈值 / 显著性方差');
    legend(ax_threshold, {'平均阈值', '显著性方差'}, 'Location', 'best');
    title(ax_threshold, '自适应阈值与显著性方差演化');

    ax_msd = subplot(3,1,3, 'Parent', fig);
    msd_line = plot(ax_msd, time_vec, msd_history, '-', 'LineWidth', 1.4, 'Color', [0.4 0.4 0.4]);
    hold(ax_msd, 'on');
    fit_line = plot(ax_msd, NaN, NaN, '--', 'LineWidth', 1.6, 'Color', [0.85 0.2 0.2]);
    burn_in_line = xline(ax_msd, time_vec(burn_in_index), ':', 'Color', [0.2 0.2 0.2], 'Label', '烧入区间');
    hold(ax_msd, 'off');
    grid(ax_msd, 'on');
    xlabel(ax_msd, '时间 (s)');
    ylabel(ax_msd, 'MSD');
    title(ax_msd, '平均平方位移随时间演化');

    for step_idx = 1:T
        sim.step();
        positions = sim.positions;
        centroid = mean(positions, 1);
        centroids(step_idx + 1, :) = centroid;

        centered = positions - centroid;
        rel_disp = centered - offsets0;
        msd_history(step_idx + 1) = mean(sum(rel_disp.^2, 2), 'omitnan');

        % 新增：记录阈值和显著性方差
        if sim.useAdaptiveThreshold && ~isempty(sim.cj_threshold_dynamic)
            threshold_history(step_idx + 1) = mean(sim.cj_threshold_dynamic);
            saliency_history(step_idx + 1) = mean(sim.local_saliency_state);
        else
            threshold_history(step_idx + 1) = sim.cj_threshold;
            saliency_history(step_idx + 1) = 0;
        end

        if mod(step_idx, 5) == 0 || step_idx == T
            colors = repmat([0.7 0.7 0.7], params.N, 1);
            if any(sim.isActive)
                colors(sim.isActive, :) = repmat([0.9 0.3 0.3], sum(sim.isActive), 1);
            end

            set(scatter_plot, 'XData', positions(:,1), 'YData', positions(:,2), 'CData', colors);
            set(quiver_plot, 'XData', positions(:,1), 'YData', positions(:,2), ...
                'UData', cos(sim.theta), 'VData', sin(sim.theta));
            set(centroid_trace, 'XData', centroids(1:step_idx+1,1), 'YData', centroids(1:step_idx+1,2));

            % 新增：更新阈值和显著性方差曲线
            set(threshold_line, 'XData', time_vec(1:step_idx+1), 'YData', threshold_history(1:step_idx+1));
            set(saliency_line, 'XData', time_vec(1:step_idx+1), 'YData', saliency_history(1:step_idx+1));

            set(msd_line, 'YData', msd_history);
            drawnow limitrate;
        end
    end

    [P_value, D_value, fit_time, fit_curve] = compute_persistence_metrics(time_vec, msd_history, burn_in_index, cfg);
    set(fit_line, 'XData', fit_time, 'YData', fit_curve);
    if ~isempty(burn_in_line)
        burn_in_line.Value = time_vec(burn_in_index);
    end
    title(ax_msd, sprintf('平均平方位移 (P = %.3f, D = %.3e)', P_value, D_value));

    scatter(ax_particles, centroids(1,1), centroids(1,2), 60, [0.3 0.7 0.3], 'filled', 'DisplayName', '初始质心');
    scatter(ax_particles, centroids(end,1), centroids(end,2), 70, [0.9 0.3 0.3], 'filled', 'DisplayName', '结束质心');
    legend(ax_particles, {'粒子位置', '方向', '质心轨迹', '初始质心', '结束质心'}, 'Location', 'bestoutside');

    result = struct();
    result.P_value = P_value;
    result.D_value = D_value;
    result.figure = fig;
end

function [P_value, D_value] = run_single_persistence(params, cfg)
    data = simulate_persistence_series(params, cfg);
    P_value = data.P_value;
    D_value = data.D_value;
end

function data = simulate_persistence_series(params, cfg)
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

    initial_positions = sim.positions;
    centroid0 = mean(initial_positions, 1);
    offsets0 = initial_positions - centroid0;

    centroids = zeros(T + 1, 2);
    centroids(1, :) = centroid0;

    msd = zeros(T + 1, 1);
    msd(1) = 0;

    for step_idx = 1:T
        sim.step();
        positions = sim.positions;
        centroid = mean(positions, 1);
        centroids(step_idx + 1, :) = centroid;

        centered = positions - centroid;
        rel_disp = centered - offsets0;
        msd(step_idx + 1) = mean(sum(rel_disp.^2, 2), 'omitnan');
    end

    time_vec = (0:T)' * dt;
    x = time_vec(burn_in_index:end);
    y = msd(burn_in_index:end);

    [P_value, D_value, fit_time, fit_curve] = compute_persistence_metrics(time_vec, msd, burn_in_index, cfg);

    data = struct();
    data.centroids = centroids;
    data.time_vec = time_vec;
    data.msd = msd;
    data.fit_time = fit_time;
    data.fit_curve = fit_curve;
    data.burn_in_index = burn_in_index;
    data.P_value = P_value;
    data.D_value = D_value;
end

function V = compute_average_velocity(theta, v0)
    V = [mean(v0 * cos(theta)), mean(v0 * sin(theta))];
end

function [P_value, D_value, fit_time, fit_curve] = compute_persistence_metrics(time_vec, msd, burn_in_index, cfg)
    x = time_vec(burn_in_index:end);
    y = msd(burn_in_index:end);

    D_value = NaN;
    fit_time = x;
    fit_curve = NaN(size(x));

    if numel(x) >= max(2, cfg.min_fit_points) && ~all(abs(y - y(1)) < eps)
        x_shift = x - x(1);
        y_shift = y - y(1);
        if any(x_shift > 0) && any(abs(y_shift) > eps)
            smooth_window = max(5, floor(numel(y_shift) * 0.1));
            if smooth_window > 1
                y_shift = smoothdata(y_shift, 'movmean', smooth_window);
            end
            slope = lsqnonneg(x_shift(:), y_shift(:));
            if slope > 0
                D_value = slope;
                fit_curve = y(1) + slope * x_shift;
            end
        end
    end

    if isnan(D_value)
        P_value = NaN;
        fit_curve(:) = NaN;
    else
        D_value = max(D_value, cfg.min_diffusion);
        P_value = 1 / sqrt(D_value);
    end
end
