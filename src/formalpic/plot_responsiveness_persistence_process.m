% plot_responsiveness_persistence_process.m - 固定阈值下的持久性演示脚本（仅持久性，样式对齐 plot_adaptive_threshold_split）
%
% 功能：
%   - 在固定运动显著性阈值 M_T 下运行一次持久性实验；
%   - 仅绘制两张图（不保存）：
%       1) 持久性最终状态 + 质心轨迹（与 adaptive 脚本一致的风格）；
%       2) MSD 拟合过程（斜率除以 4 以匹配二维扩散公式）。
%
% 说明：
%   - 持久性定义与 plot_adaptive_threshold_split.m 完全一致：
%     以相对质心的 MSD 拟合扩散系数 D（二维：MSD 斜率 = 4D），P = 1/sqrt(D)。
%   - 不额外绘制质心位移或序参量时间序列，保持与原脚本持久性部分的粒度一致。

clear; clc; close all;

%% 图像/样式配置（与 adaptive 脚本保持一致）
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

style = struct();
style.font_name = FONT_NAME;
style.label_font_size = LABEL_FONT_SIZE;
style.label_font_weight = LABEL_FONT_WEIGHT;
style.tick_font_size = TICK_FONT_SIZE;
style.tick_font_weight = TICK_FONT_WEIGHT;
style.axis_line_width = AXIS_LINE_WIDTH;
style.traj_line_width = LINE_WIDTH_TRAJ;
style.state_marker_size = STATE_MARKER_SIZE;

%% 固定阈值持久性参数
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
base_params.cj_threshold = 5;      % 固定运动显著性阈值
base_params.fieldSize = 50;
base_params.initDirection = pi/4;
base_params.useFixedField = true;
base_params.useAdaptiveThreshold = false;

pers_cfg = struct();
pers_cfg.burn_in_ratio = 0.5;
pers_cfg.min_fit_points = 80;
pers_cfg.min_diffusion = 1e-3;

PERS_SEED = 20251111;

%% 依赖路径
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(script_dir));
addpath(genpath(fullfile(project_root, 'src', 'sim')));

%% 运行持久性实验
rng(PERS_SEED);
pers_data = compute_persistence_once(base_params, pers_cfg);

%% 图1：最终状态 + 质心轨迹
fig_state = figure('Color', 'white', 'Position', [100, 80, FIG_SIZE_STATE]);
ax_state = axes('Parent', fig_state);
plot_persistence_state(ax_state, pers_data, style);

%% 图2：MSD 拟合过程
fig_msd = figure('Color', 'white', 'Position', [140, 120, FIG_SIZE_TIMELINE]);
ax_msd = axes('Parent', fig_msd);
hold(ax_msd, 'on');
plot(ax_msd, pers_data.time_vec, pers_data.msd_history, 'LineWidth', LINE_WIDTH_SERIES, ...
    'Color', [0.4 0.4 0.4], 'DisplayName', 'MSD');
plot(ax_msd, pers_data.fit_time, pers_data.fit_curve, '--', 'LineWidth', LINE_WIDTH_FIT, ...
    'Color', [0.85 0.2 0.2], 'DisplayName', 'Linear fit');
xline(ax_msd, pers_data.burn_in_time, '--', 'Color', [0.3 0.3 0.3], ...
    'LineWidth', 2, 'Label', 'Burn-in');
hold(ax_msd, 'off');
set_series_axes(ax_msd, 'MSD', 'Time (s)', style);
legend(ax_msd, 'Location', 'northwest');

%% ========================================================================
%% 辅助函数（与 adaptive 脚本的持久性计算保持一致）
%% ========================================================================
function data = compute_persistence_once(params, cfg)
    sim = ParticleSimulation(params);
    T = sim.T_max;
    N = sim.N;
    dt = sim.dt;

    positions_history = zeros(T + 1, N, 2);
    positions_history(1, :, :) = sim.positions;
    time_vec = (0:T)' * dt;

    centroid0 = mean(sim.positions, 1);
    offsets0 = sim.positions - centroid0;
    centroid_path = zeros(T + 1, 2);
    centroid_path(1, :) = centroid0;
    msd_history = zeros(T + 1, 1);

    for step = 1:T
        sim.step();
        positions_history(step + 1, :, :) = sim.positions;
        centroid = mean(sim.positions, 1);
        centroid_path(step + 1, :) = centroid;
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
        intercept = y(1) - slope * x(1);
        fit_curve = slope * x + intercept;
    else
        slope = cfg.min_diffusion;
        fit_curve = slope * x;
    end

    D_value = max(slope / 4, cfg.min_diffusion);  % 二维扩散：MSD 斜率 = 4D
    P_value = 1 / sqrt(D_value);

    data = struct();
    data.positions_history = positions_history;
    data.field_size = sim.fieldSize;
    data.sample_indices = select_trajectory_indices(N, 35);
    data.centroid_path = centroid_path;
    data.time_vec = time_vec;
    data.msd_history = msd_history;
    data.fit_time = x;
    data.fit_curve = fit_curve;
    data.burn_in_time = time_vec(burn_in_idx);
    data.P_value = P_value;
    data.D_value = D_value;
end

function indices = select_trajectory_indices(num_agents, max_count)
    count = min(max_count, num_agents);
    indices = round(linspace(1, num_agents, count));
end

function set_series_axes(ax, y_label, x_label, style)
    % 与 plot_adaptive_threshold_split 的 apply_series_style 保持一致的轴样式
    grid(ax, 'off');
    ax.LineWidth = style.axis_line_width;
    ax.FontName = style.font_name;
    ax.FontSize = style.tick_font_size;
    ax.FontWeight = style.tick_font_weight;
    ax.TickDir = 'in';
    box(ax, 'on');
    ylabel(ax, y_label, 'FontName', style.font_name, 'FontSize', style.label_font_size, ...
        'FontWeight', style.label_font_weight);
    xlabel(ax, x_label, 'FontName', style.font_name, 'FontSize', style.label_font_size, ...
        'FontWeight', style.label_font_weight);
end

function plot_persistence_state(ax, pers_data, style)
    hold(ax, 'on');
    pos_hist = pers_data.positions_history;
    pos_final = squeeze(pos_hist(end, :, :));
    field_size = pers_data.field_size;

    % 粒子轨迹（灰色，与 adaptive 持久性状态图一致）
    for idx = 1:size(pos_hist, 2)
        traj = squeeze(pos_hist(:, idx, :));
        plot(ax, traj(:,1), traj(:,2), 'Color', [0.6 0.6 0.6], 'LineWidth', style.traj_line_width);
    end

    % 最终散点
    scatter(ax, pos_final(:,1), pos_final(:,2), style.state_marker_size, [0.6 0.6 0.6], ...
        'filled', 'MarkerFaceAlpha', 0.75, 'MarkerEdgeAlpha', 0.3);

    % 质心轨迹
    plot(ax, pers_data.centroid_path(:,1), pers_data.centroid_path(:,2), '-', ...
        'Color', [0.2 0.45 0.8], 'LineWidth', 1.6, 'DisplayName', 'Centroid path');
    scatter(ax, pers_data.centroid_path(1,1), pers_data.centroid_path(1,2), 55, [0.1 0.6 0.3], 'filled');
    scatter(ax, pers_data.centroid_path(end,1), pers_data.centroid_path(end,2), 65, [0.85 0.2 0.2], 'filled');

    if ~isnan(pers_data.P_value)
        text(ax, field_size - 15, field_size - 4, sprintf('P = %.3f', pers_data.P_value), ...
            'FontName', style.font_name, 'FontWeight', 'bold', 'FontSize', 12, 'Color', [0.15 0.35 0.8]);
    end

    % 轴样式对齐 apply_state_style
    lim = min(50, field_size);
    axis(ax, [0 lim 0 lim]);
    xticks(ax, 0:10:lim);
    yticks(ax, 0:10:lim);
    axis(ax, 'square');
    grid(ax, 'off');
    box(ax, 'on');
    ax.LineWidth = style.axis_line_width;
    ax.FontName = style.font_name;
    ax.FontSize = style.tick_font_size;
    ax.FontWeight = style.tick_font_weight;
    ax.TickDir = 'in';
    xlabel(ax, 'X', 'FontName', style.font_name, 'FontSize', style.label_font_size, ...
        'FontWeight', style.label_font_weight);
    ylabel(ax, 'Y', 'FontName', style.font_name, 'FontSize', style.label_font_size, ...
        'FontWeight', style.label_font_weight);
    hold(ax, 'off');
end
