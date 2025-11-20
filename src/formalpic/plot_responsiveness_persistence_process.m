% plot_responsiveness_persistence_process.m - 固定阈值下的响应性/持久性演示脚本
%
% 功能：
%   - 使用固定运动显著性阈值 (M_T) 运行一次响应性实验与一次持久性实验；
%   - 输出四幅高质量图：响应性最终状态、持久性最终状态、响应性计算过程、持久性计算过程；
%   - 图片保存到项目 pic 目录，格式与 formalpic 目录中其他脚本保持一致。

clear; clc; close all;

%% -------------------- 图像配置 --------------------
FIG_SIZE_STATE = [500, 500];
FIG_SIZE_TIMELINE = [560, 360];
FONT_NAME = 'Arial';
LABEL_FONT_SIZE = 13;
LABEL_FONT_WEIGHT = 'Bold';
TICK_FONT_SIZE = 12;
TICK_FONT_WEIGHT = 'Bold';
AXIS_LINE_WIDTH = 1.5;
LINE_WIDTH_TRAJ = 1.0;
LINE_WIDTH_SERIES = 2.0;
LINE_WIDTH_FIT = 2.2;
STATE_MARKER_SIZE = 18;
SHADE_COLOR = [0.9 0.8 0.6];

style = struct();
style.font_name = FONT_NAME;
style.label_font_size = LABEL_FONT_SIZE;
style.label_font_weight = LABEL_FONT_WEIGHT;
style.tick_font_size = TICK_FONT_SIZE;
style.tick_font_weight = TICK_FONT_WEIGHT;
style.axis_line_width = AXIS_LINE_WIDTH;
style.traj_line_width = LINE_WIDTH_TRAJ;
style.traj_color = [0.2 0.4 0.8];
style.state_marker_size = STATE_MARKER_SIZE;

%% -------------------- 仿真参数 --------------------
base_params = struct();
base_params.N = 200;
base_params.rho = 1;
base_params.v0 = 1;
base_params.angleUpdateParameter = 10;
base_params.angleNoiseIntensity = 0.045;
base_params.T_max = 400;
base_params.dt = 0.1;
base_params.radius = 5;
base_params.deac_threshold = 0.1745;
base_params.cj_threshold =5;   % 固定运动显著性阈值
base_params.fieldSize = 50;
base_params.initDirection = pi/4;
base_params.useFixedField = true;
base_params.stabilization_steps = 180;
base_params.forced_turn_duration = 160;
base_params.useAdaptiveThreshold = false;

pers_cfg = struct();
pers_cfg.burn_in_ratio = 0.5;
pers_cfg.min_fit_points = 80;
pers_cfg.min_diffusion = 1e-3;

RESP_SEED = 20251110;
PERS_SEED = 20251111;

%% -------------------- 路径与依赖 --------------------
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(script_dir));
pic_dir = fullfile(project_root, 'pic');
if ~exist(pic_dir, 'dir')
    mkdir(pic_dir);
end
addpath(genpath(fullfile(project_root, 'src', 'sim')));

%% -------------------- 运行仿真 --------------------
rng(RESP_SEED);
resp_data = simulate_responsiveness_process(base_params);

rng(PERS_SEED);
pers_params = base_params;
pers_params = rmfield(pers_params, {'stabilization_steps', 'forced_turn_duration'});
pers_data = simulate_persistence_process(pers_params, pers_cfg);

%% -------------------- 绘图：最终状态 --------------------
resp_state_fig = figure('Color', 'white', 'Position', [80, 80, FIG_SIZE_STATE]);
ax_resp_state = axes('Parent', resp_state_fig);
plot_final_state(ax_resp_state, resp_data, '响应性：最终状态', style);
annotate_axes(ax_resp_state, style);
if isnan(resp_data.R_value)
    resp_label = 'R 未触发';
else
    resp_label = sprintf('R = %.3f', resp_data.R_value);
end
text(ax_resp_state, 2, resp_data.field_size - 2, resp_label, ...
    'FontName', FONT_NAME, 'FontWeight', 'bold', 'FontSize', 12, 'Color', [0.8 0.2 0.2]);

pers_state_fig = figure('Color', 'white', 'Position', [120, 120, FIG_SIZE_STATE]);
ax_pers_state = axes('Parent', pers_state_fig);
plot_final_state(ax_pers_state, pers_data, '持久性：最终状态', style);
annotate_axes(ax_pers_state, style);
plot(ax_pers_state, pers_data.centroid_path(:,1), pers_data.centroid_path(:,2), '-', ...
    'Color', [0.2 0.45 0.8], 'LineWidth', 1.5, 'DisplayName', '质心轨迹');
legend(ax_pers_state, 'Location', 'northeastoutside');
text(ax_pers_state, 2, pers_data.field_size - 2, sprintf('P = %.3f', pers_data.P_value), ...
    'FontName', FONT_NAME, 'FontWeight', 'bold', 'FontSize', 12, 'Color', [0.1 0.4 0.7]);

%% -------------------- 绘图：响应性计算过程 --------------------
resp_series_fig = figure('Color', 'white', 'Position', [160, 180, FIG_SIZE_TIMELINE]);
ax_resp_series = axes('Parent', resp_series_fig);
hold(ax_resp_series, 'on');
area(ax_resp_series, resp_data.time_vec, resp_data.projection_series, 'FaceColor', SHADE_COLOR, ...
    'EdgeColor', 'none', 'FaceAlpha', 0.6);
plot(ax_resp_series, resp_data.time_vec, resp_data.projection_series, 'LineWidth', LINE_WIDTH_SERIES, ...
    'Color', [0.75 0.25 0.25]);
if ~isnan(resp_data.R_value)
    xline(ax_resp_series, resp_data.trigger_time, '--', 'Color', [0.3 0.3 0.3], ...
        'LineWidth', 1.2, 'Label', '触发');
    xline(ax_resp_series, resp_data.window_end_time, ':', 'Color', [0.3 0.3 0.3], ...
        'LineWidth', 1.2, 'Label', '窗口结束');
end
hold(ax_resp_series, 'off');
set_series_axes(ax_resp_series, '响应性投影 r(t)', '时间 (s)', style);
title(ax_resp_series, sprintf('响应性积分过程 (R = %.3f)', resp_data.R_value), ...
    'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);

%% -------------------- 绘图：持久性计算过程 --------------------
pers_series_fig = figure('Color', 'white', 'Position', [200, 240, FIG_SIZE_TIMELINE]);
ax_pers_series = axes('Parent', pers_series_fig);
hold(ax_pers_series, 'on');
plot(ax_pers_series, pers_data.time_vec, pers_data.msd_history, 'LineWidth', LINE_WIDTH_SERIES, ...
    'Color', [0.4 0.4 0.4], 'DisplayName', 'MSD');
plot(ax_pers_series, pers_data.fit_time, pers_data.fit_curve, '--', 'LineWidth', LINE_WIDTH_FIT, ...
    'Color', [0.85 0.2 0.2], 'DisplayName', '线性拟合');
xline(ax_pers_series, pers_data.burn_in_time, ':', 'Color', [0.2 0.2 0.2], ...
    'LineWidth', 1.1, 'Label', '烧入');
hold(ax_pers_series, 'off');
set_series_axes(ax_pers_series, 'MSD', '时间 (s)', style);
legend(ax_pers_series, 'Location', 'northwest');
title(ax_pers_series, sprintf('持久性估计 (P = %.3f, D = %.2e)', ...
    pers_data.P_value, pers_data.D_value), 'FontName', FONT_NAME, ...
    'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);

%% -------------------- 输出 --------------------
exportgraphics(resp_state_fig, fullfile(pic_dir, 'fixed_threshold_responsiveness_state.pdf'), 'ContentType', 'vector');
exportgraphics(pers_state_fig, fullfile(pic_dir, 'fixed_threshold_persistence_state.pdf'), 'ContentType', 'vector');
exportgraphics(resp_series_fig, fullfile(pic_dir, 'fixed_threshold_responsiveness_series.pdf'), 'ContentType', 'vector');
exportgraphics(pers_series_fig, fullfile(pic_dir, 'fixed_threshold_persistence_series.pdf'), 'ContentType', 'vector');

fprintf('图像已输出至: %s\n', pic_dir);

%% ========================================================================
%% 辅助函数
%% ========================================================================
function data = simulate_responsiveness_process(params)
    sim = ParticleSimulationWithExternalPulse(params);
    T = sim.T_max;
    num_agents = sim.N;
    positions_history = zeros(T + 1, num_agents, 2);
    positions_history(1, :, :) = sim.positions;

    time_vec = (0:T)' * sim.dt;
    projection_series = zeros(T + 1, 1);
    triggered = false;
    trigger_step = NaN;
    window_end_step = NaN;
    target_dir = [1; 0];

    for step = 1:T
        sim.step();
        positions_history(step + 1, :, :) = sim.positions;

        if sim.external_pulse_triggered && ~triggered
            triggered = true;
            trigger_step = step;
            window_end_step = min(step + params.forced_turn_duration, T);
            leader_idx = find(sim.isExternallyActivated, 1, 'first');
            if isempty(leader_idx)
                leader_idx = 1;
            end
            theta_target = sim.external_target_theta(leader_idx);
            target_dir = [cos(theta_target); sin(theta_target)];
        end

        velocity_vec = compute_mean_velocity(sim.theta, params.v0);
        if triggered
            projection_series(step + 1) = max(0, velocity_vec' * target_dir);
        end
    end

    if triggered
        dt = sim.dt;
        idx = trigger_step + 1:window_end_step + 1;
        integral_value = trapz(time_vec(idx), projection_series(idx));
        duration = (window_end_step - trigger_step) * dt;
        if duration > 0
            R_value = integral_value / (params.v0 * duration);
        else
            R_value = 0;
        end
    else
        R_value = NaN;
        trigger_step = 0;
        window_end_step = 0;
    end

    data = struct();
    data.positions_history = positions_history;
    data.field_size = params.fieldSize;
    data.time_vec = time_vec;
    data.projection_series = projection_series;
    data.R_value = R_value;
    data.trigger_time = time_vec(trigger_step + 1);
    data.window_end_time = time_vec(window_end_step + 1);
    data.sample_indices = select_trajectory_indices(num_agents, 35);
end

function data = simulate_persistence_process(params, cfg)
    sim = ParticleSimulation(params);
    T = sim.T_max;
    num_agents = sim.N;
    positions_history = zeros(T + 1, num_agents, 2);
    positions_history(1, :, :) = sim.positions;
    time_vec = (0:T)' * sim.dt;

    centroid0 = mean(sim.positions, 1);
    offsets0 = sim.positions - centroid0;
    centroids = zeros(T + 1, 2);
    centroids(1, :) = centroid0;
    msd_history = zeros(T + 1, 1);

    for step = 1:T
        sim.step();
        positions_history(step + 1, :, :) = sim.positions;
        centroid = mean(sim.positions, 1);
        centroids(step + 1, :) = centroid;
        centered = sim.positions - centroid;
        rel_disp = centered - offsets0;
        msd_history(step + 1) = mean(sum(rel_disp.^2, 2), 'omitnan');
    end

    burn_in_idx = max(2, floor((T + 1) * cfg.burn_in_ratio));
    fit_idx = burn_in_idx:(T + 1);
    if numel(fit_idx) < cfg.min_fit_points
        fit_idx = max(2, T - cfg.min_fit_points + 2):(T + 1);
    end
    x = time_vec(fit_idx);
    y = msd_history(fit_idx);
    if numel(x) >= 2
        p = polyfit(x, y, 1);
        slope = max(p(1), cfg.min_diffusion);
        intercept = y(1) - slope * (x(1));
        fit_curve = slope * x + intercept;
    else
        slope = cfg.min_diffusion;
        fit_curve = slope * x;
    end
    P_value = 1 / sqrt(slope);

    data = struct();
    data.positions_history = positions_history;
    data.field_size = params.fieldSize;
    data.sample_indices = select_trajectory_indices(num_agents, 35);
    data.centroid_path = centroids;
    data.time_vec = time_vec;
    data.msd_history = msd_history;
    data.fit_time = x;
    data.fit_curve = fit_curve;
    data.burn_in_time = time_vec(burn_in_idx);
    data.P_value = P_value;
    data.D_value = slope / 4;  % 二维扩散修正
end

function indices = select_trajectory_indices(num_agents, max_count)
    count = min(max_count, num_agents);
    indices = round(linspace(1, num_agents, count));
end

function vec = compute_mean_velocity(theta, v0)
    vec = [mean(cos(theta)); mean(sin(theta))] * v0;
end

function plot_final_state(ax, data, title_str, style)
    hold(ax, 'on');
    for idx = data.sample_indices
        traj = squeeze(data.positions_history(:, idx, :));
        plot(ax, traj(:,1), traj(:,2), 'Color', style.traj_color, 'LineWidth', style.traj_line_width);
    end
    final_positions = squeeze(data.positions_history(end, :, :));
    scatter(ax, final_positions(:,1), final_positions(:,2), style.state_marker_size, ...
        [0.6 0.6 0.6], 'filled', 'MarkerFaceAlpha', 0.75, 'MarkerEdgeAlpha', 0.3);
    axis(ax, [0 data.field_size 0 data.field_size]);
    axis(ax, 'square');
    grid(ax, 'on');
    title(ax, title_str, 'FontName', style.font_name, 'FontSize', style.label_font_size, ...
        'FontWeight', style.label_font_weight);
    hold(ax, 'off');
end

function annotate_axes(ax, style)
    ax.LineWidth = style.axis_line_width;
    ax.FontName = style.font_name;
    ax.FontSize = style.tick_font_size;
    ax.FontWeight = style.tick_font_weight;
    ax.TickDir = 'in';
    xlabel(ax, 'X', 'FontName', style.font_name, 'FontSize', style.label_font_size, ...
        'FontWeight', style.label_font_weight);
    ylabel(ax, 'Y', 'FontName', style.font_name, 'FontSize', style.label_font_size, ...
        'FontWeight', style.label_font_weight);
end

function set_series_axes(ax, y_label, x_label, style)
    ax.LineWidth = style.axis_line_width;
    ax.FontName = style.font_name;
    ax.FontSize = style.tick_font_size;
    ax.FontWeight = style.tick_font_weight;
    ax.TickDir = 'in';
    grid(ax, 'on');
    ylabel(ax, y_label, 'FontName', style.font_name, 'FontSize', style.label_font_size, ...
        'FontWeight', style.label_font_weight);
    xlabel(ax, x_label, 'FontName', style.font_name, 'FontSize', style.label_font_size, ...
        'FontWeight', style.label_font_weight);
end
