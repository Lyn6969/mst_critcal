% plot_adaptive_threshold_split.m - Adaptive threshold responsiveness/persistence plots
% 本脚本用于批量生成自适应阈值机制在响应性与持久性方面的关键统计图
% 执行前需确保仿真依赖（ParticleSimulation* 系列类）已在路径下可用

clear; clc; close all;

%% -------------------- 图像与样式参数 --------------------
FIG_SIZE_TRAJ = [400, 400];   % 轨迹信息展示所需的方形画布尺寸
FIG_SIZE_SMALL = [400, 200];  % 统计曲线类图形统一使用的瘦长画布

style = struct();  % 统一的字体、粗细等排版规范，保证最终制图一致性
style.font_name = 'Arial';
style.label_font_size = 12;
style.label_font_weight = 'Bold';
style.tick_font_size = 11;
style.tick_font_weight = 'Bold';
style.axis_line_width = 1.5;
style.shape = struct('head_radius', 0.25, 'tail_length', 0.45, 'tail_width', 0.12, 'resolution', 38);

colors = struct();  % 归纳所有需要用到的颜色，方便集中调整
colors.base = [0.7 0.7 0.7];
colors.activated = [0.85 0.2 0.2];
colors.external = [1.0 0.55 0.0];
colors.saliency_mean = [0.55 0.0 0.0];
colors.resp_metric = [0.75 0.05 0.05];
colors.low_ratio = [0.1 0.4 0.8];
colors.active_ratio = [0.85 0.35 0.1];
colors.msd = [0.3 0.3 0.3];
colors.fit = [0.85 0.2 0.2];
colors.traj_gray_start = [0.85 0.85 0.85];
colors.traj_gray_end = [0.45 0.45 0.45];
colors.traj_red_start = [1.0 0.8 0.8];
colors.traj_red_end = [0.65 0.0 0.0];

%% -------------------- 仿真参数（与 visualize_adaptive_threshold 保持一致） --------------------
params = struct();  % 与仿真主脚本保持一致的核心参数配置
params.N = 200;
params.rho = 1;
params.v0 = 1;
params.angleUpdateParameter = 10;
params.angleNoiseIntensity = 0.045;
params.T_max = 400;
params.dt = 0.1;
params.radius = 5;
params.deac_threshold = 0.1745;
params.cj_threshold = 1.5;
params.fieldSize = 50;
params.initDirection = pi/4;
params.useFixedField = true;
params.stabilization_steps = 200;
params.forced_turn_duration = 200;
params.useAdaptiveThreshold = true;

time_marker = struct();  % 用于在所有时间序列图上标记稳定化结束的竖线
time_marker.step = params.stabilization_steps;
time_marker.color = [0.3 0.3 0.3];
time_marker.line_style = '--';
time_marker.line_width = 2;

adaptive_cfg = struct();  % 自适应阈值的上下限与显著性筛选门限
adaptive_cfg.cj_low = 0.5;
adaptive_cfg.cj_high = 5;
adaptive_cfg.saliency_threshold = 0.04;
adaptive_cfg.include_self = true;
params.adaptiveThresholdConfig = adaptive_cfg;

pers_cfg = struct();  % 持久性拟合所需的烧入比例与拟合点阈值
pers_cfg.burn_in_ratio = 0.5;
pers_cfg.min_fit_points = 40;
pers_cfg.min_diffusion = 1e-3;

base_seed = 20250315;              % 响应性仿真随机种子，方便复现实验
pers_seed = base_seed + 12345;     % 持久性仿真随机种子，通过偏移保持差异
time_vec = (0:params.T_max)' * params.dt;  % 对应物理时间，用于积分与标注

%% -------------------- 路径与依赖 --------------------
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(script_dir));  % formalpic 脚本严格向上两级
pic_dir = fullfile(project_root, 'pic');
if ~exist(pic_dir, 'dir')
    mkdir(pic_dir);
end
addpath(genpath(fullfile(project_root, 'src', 'sim')));

%% -------------------- 数据采集 --------------------
step_vec = (0:params.T_max)';  % 离散步数索引，与仿真内部计数保持一致
resp_data = collect_responsiveness_data(params, time_vec, step_vec, base_seed);  % 抽取响应性相关的所有统计量
pers_data = collect_persistence_data(params, pers_cfg, step_vec, pers_seed);    % 提取持久性所需的轨迹与MSD信息

fprintf('Responsivity R = %.3f\n', resp_data.R_value);

%% -------------------- 绘图与导出 --------------------
% 1. 响应性轨迹（最终状态）
resp_traj_path = fullfile(pic_dir, 'adaptive_resp_trajectory.pdf');
resp_traj_fig = figure('Color', 'white', 'Position', [80, 80, FIG_SIZE_TRAJ]);
ax_resp_traj = axes('Parent', resp_traj_fig);
plot_responsiveness_state(ax_resp_traj, resp_data, style, colors);
export_and_close(resp_traj_fig, resp_traj_path);

% 2. 响应性：显著性方差演化
resp_saliency_path = fullfile(pic_dir, 'adaptive_resp_saliency.pdf');
resp_saliency_fig = figure('Color', 'white', 'Position', [120, 120, FIG_SIZE_SMALL]);
ax_resp_saliency = axes('Parent', resp_saliency_fig);
plot_saliency_series(ax_resp_saliency, resp_data.step_vec, resp_data.saliency_all, ...
    resp_data.saliency_mean, colors.saliency_mean, style);
add_time_marker(ax_resp_saliency, time_marker);
apply_series_style(ax_resp_saliency, style, 'Saliency variance');
export_and_close(resp_saliency_fig, resp_saliency_path);

% 3. 响应性：低阈值占比
resp_low_ratio_path = fullfile(pic_dir, 'adaptive_resp_low_ratio.pdf');
resp_low_fig = figure('Color', 'white', 'Position', [160, 160, FIG_SIZE_SMALL]);
ax_resp_low = axes('Parent', resp_low_fig);
plot(ax_resp_low, resp_data.step_vec, resp_data.low_ratio, 'Color', colors.low_ratio, 'LineWidth', 2);
add_time_marker(ax_resp_low, time_marker);
apply_series_style(ax_resp_low, style, 'Low-threshold fraction');
export_and_close(resp_low_fig, resp_low_ratio_path);

% 4. 响应性：激活占比
resp_active_path = fullfile(pic_dir, 'adaptive_resp_active_ratio.pdf');
resp_active_fig = figure('Color', 'white', 'Position', [200, 200, FIG_SIZE_SMALL]);
ax_resp_active = axes('Parent', resp_active_fig);
plot(ax_resp_active, resp_data.step_vec, resp_data.active_ratio, 'Color', colors.active_ratio, 'LineWidth', 2);
add_time_marker(ax_resp_active, time_marker);
apply_series_style(ax_resp_active, style, 'Activated fraction');
export_and_close(resp_active_fig, resp_active_path);

% 5. 响应性：响应性指标
resp_metric_path = fullfile(pic_dir, 'adaptive_resp_metric.pdf');
resp_metric_fig = figure('Color', 'white', 'Position', [240, 240, FIG_SIZE_SMALL]);
ax_resp_metric = axes('Parent', resp_metric_fig);
plot(ax_resp_metric, resp_data.step_vec, resp_data.R_series, 'Color', colors.resp_metric, 'LineWidth', 2);
add_time_marker(ax_resp_metric, time_marker);
apply_series_style(ax_resp_metric, style, 'Responsivity');
export_and_close(resp_metric_fig, resp_metric_path);

% 6. 持久性：粒子状态 + 质心轨迹
% pers_state_path = fullfile(pic_dir, 'adaptive_pers_state.pdf');
% pers_state_fig = figure('Color', 'white', 'Position', [280, 280, FIG_SIZE_TRAJ]);
% ax_pers_state = axes('Parent', pers_state_fig);
% plot_persistence_state(ax_pers_state, pers_data, style, colors);
% export_and_close(pers_state_fig, pers_state_path);

% 7. 持久性：显著性方差演化
pers_saliency_path = fullfile(pic_dir, 'adaptive_pers_saliency.pdf');
pers_saliency_fig = figure('Color', 'white', 'Position', [320, 320, FIG_SIZE_SMALL]);
ax_pers_saliency = axes('Parent', pers_saliency_fig);
plot_saliency_series(ax_pers_saliency, pers_data.step_vec, pers_data.saliency_all, ...
    pers_data.saliency_mean, colors.saliency_mean, style);
add_time_marker(ax_pers_saliency, time_marker);
apply_series_style(ax_pers_saliency, style, 'Saliency variance');
export_and_close(pers_saliency_fig, pers_saliency_path);

% 8. 持久性：低阈值占比
pers_low_ratio_path = fullfile(pic_dir, 'adaptive_pers_low_ratio.pdf');
pers_low_fig = figure('Color', 'white', 'Position', [360, 360, FIG_SIZE_SMALL]);
ax_pers_low = axes('Parent', pers_low_fig);
plot(ax_pers_low, pers_data.step_vec, pers_data.low_ratio, 'Color', colors.low_ratio, 'LineWidth', 2);
add_time_marker(ax_pers_low, time_marker);
apply_series_style(ax_pers_low, style, 'Low-threshold fraction');
export_and_close(pers_low_fig, pers_low_ratio_path);

% 9. 持久性：激活占比
pers_active_ratio_path = fullfile(pic_dir, 'adaptive_pers_active_ratio.pdf');
pers_active_fig = figure('Color', 'white', 'Position', [400, 400, FIG_SIZE_SMALL]);
ax_pers_active = axes('Parent', pers_active_fig);
plot(ax_pers_active, pers_data.step_vec, pers_data.active_ratio, 'Color', colors.active_ratio, 'LineWidth', 2);
add_time_marker(ax_pers_active, time_marker);
apply_series_style(ax_pers_active, style, 'Activated fraction');
export_and_close(pers_active_fig, pers_active_ratio_path);

% 10. 持久性：MSD 曲线
pers_msd_path = fullfile(pic_dir, 'adaptive_pers_msd.pdf');
pers_msd_fig = figure('Color', 'white', 'Position', [440, 440, FIG_SIZE_SMALL]);
ax_pers_msd = axes('Parent', pers_msd_fig);
hold(ax_pers_msd, 'on');
plot(ax_pers_msd, pers_data.step_vec, pers_data.msd_history, 'Color', colors.msd, 'LineWidth', 2);
if ~any(isnan(pers_data.fit_curve))
    plot(ax_pers_msd, pers_data.fit_step, pers_data.fit_curve, '--', 'Color', colors.fit, 'LineWidth', 2);
end
add_time_marker(ax_pers_msd, time_marker);
hold(ax_pers_msd, 'off');
apply_series_style(ax_pers_msd, style, 'MSD');
export_and_close(pers_msd_fig, pers_msd_path);
fprintf('Figures saved to: %s\n', pic_dir);

%% ========================================================================
%% 辅助函数定义
%% ========================================================================
function data = collect_responsiveness_data(params, time_vec, step_vec, seed)
    rng(seed);  % 锁定随机种子，确保图像可重复再现
    sim = ParticleSimulationWithExternalPulse(params);  % 使用含外部脉冲的仿真器
    sim.external_pulse_count = 1;
    sim.resetCascadeTracking();
    sim.initializeParticles();
    sim.current_step = 0;

    T = params.T_max;
    N = params.N;
    saliency_all = zeros(T + 1, N);   % 每一帧的显著性分布
    low_ratio = zeros(T + 1, 1);      % 低阈值个体占比
    active_ratio = zeros(T + 1, 1);   % 激活状态占比
    avg_threshold = zeros(T + 1, 1);

    [saliency_all(1, :), low_ratio(1), avg_threshold(1)] = capture_saliency_snapshot(sim);  % 记录初态
    active_ratio(1) = mean(sim.isActive);

    positions_history = zeros(T + 1, N, 2);  % 用于轨迹绘制的全量位置缓存
    positions_history(1, :, :) = sim.positions;
    V_history = zeros(T + 1, 2);
    V_history(1, :) = compute_average_velocity(sim.theta, params.v0);
    projection_history = zeros(T + 1, 1);
    R_series = zeros(T + 1, 1);  % 响应性随时间变化的序列

    triggered = false;
    t_start = NaN;
    n_vec = [];
    external_indices = [];

    for t = 1:T  % 主仿真循环：推进粒子状态并采样统计量
        sim.step();
        positions_history(t + 1, :, :) = sim.positions;
        V_history(t + 1, :) = compute_average_velocity(sim.theta, params.v0);

        [saliency_all(t + 1, :), low_ratio(t + 1), avg_threshold(t + 1)] = capture_saliency_snapshot(sim);
        active_ratio(t + 1) = mean(sim.isActive);

        if ~triggered && sim.external_pulse_triggered
            triggered = true;
            t_start = t;
            leader_idx = find(sim.isExternallyActivated, 1, 'first');
            if isempty(leader_idx)
                leader_idx = 1;
            end
            theta_target = sim.external_target_theta(max(leader_idx, 1));
            n_vec = [cos(theta_target); sin(theta_target)];
            external_indices = find(sim.isExternallyActivated | sim.external_activation_timer > 0);
        end

        if triggered
            projection_history(t + 1) = V_history(t + 1, :) * n_vec;
            R_series(t + 1) = projection_history(t + 1) / params.v0;
        end
    end

    if ~triggered || isnan(t_start)
        R_val = NaN;
        trigger_time = NaN;
        window_end_time = NaN;
    else
        T_window = params.forced_turn_duration;
        t_end = min(t_start + T_window, T);
        integral_value = trapz(time_vec(t_start + 1:t_end + 1), projection_history(t_start + 1:t_end + 1));  % 数值积分衡量响应程度
        duration = time_vec(t_end + 1) - time_vec(t_start + 1);
        if duration > 0
            R_val = integral_value / (params.v0 * duration);
        else
            R_val = NaN;
        end
        trigger_time = time_vec(t_start + 1);
        window_end_time = time_vec(t_end + 1);
    end

    data = struct();  % 汇总所有中间量，方便后续绘图函数直接调用
    data.time_vec = time_vec;
    data.step_vec = step_vec;
    data.saliency_all = saliency_all;
    data.saliency_mean = mean(saliency_all, 2);
    data.low_ratio = low_ratio;
    data.active_ratio = active_ratio;
    data.avg_threshold = avg_threshold;
    data.positions_history = positions_history;
    data.final_positions = sim.positions;
    data.final_theta = sim.theta;
    data.field_size = sim.fieldSize;
    data.ever_activated = sim.everActivated;
    data.external_indices = external_indices;
    data.R_value = R_val;
    data.trigger_time = trigger_time;
    data.window_end_time = window_end_time;
    data.R_series = R_series;
end

function data = collect_persistence_data(params, cfg, step_vec, seed)
    rng(seed);          % 固定随机性，避免持久性评估结果漂移
    params_pers = params;
    removable = {'stabilization_steps', 'forced_turn_duration'};
    for k = 1:numel(removable)
        if isfield(params_pers, removable{k})
            params_pers = rmfield(params_pers, removable{k});
        end
    end

    sim = ParticleSimulation(params_pers);  % 持久性分析不依赖外部脉冲
    sim.current_step = 0;

    T = sim.T_max;
    N = sim.N;
    dt = sim.dt;
    time_vec = (0:T)' * dt;

    saliency_all = zeros(T + 1, N);  % 记录显著性演化
    low_ratio = zeros(T + 1, 1);
    active_ratio = zeros(T + 1, 1);
    [saliency_all(1, :), low_ratio(1)] = capture_saliency_snapshot(sim);
    active_ratio(1) = mean(sim.isActive);

    positions0 = sim.positions;
    centroid0 = mean(positions0, 1);
    offsets0 = positions0 - centroid0;  % 以初始质心为参考，方便计算 MSD
    centroid_path = zeros(T + 1, 2);
    centroid_path(1, :) = centroid0;
    msd_history = zeros(T + 1, 1);
    positions_history = zeros(T + 1, N, 2);
    positions_history(1, :, :) = sim.positions;

    for t = 1:T  % 持续推进粒子系统，并累积质心轨迹和 MSD
        sim.step();
        positions_history(t + 1, :, :) = sim.positions;
        positions = sim.positions;
        centroid = mean(positions, 1);
        centroid_path(t + 1, :) = centroid;

        centered = positions - centroid;
        rel_disp = centered - offsets0;
        msd_history(t + 1) = mean(sum(rel_disp.^2, 2), 'omitnan');

        [saliency_all(t + 1, :), low_ratio(t + 1)] = capture_saliency_snapshot(sim);
        active_ratio(t + 1) = mean(sim.isActive);
    end

    burn_in_index = max(2, floor((T + 1) * cfg.burn_in_ratio));  % 烧入阶段剔除早期波动
    [P_value, D_value, fit_time, fit_curve] = compute_persistence_metrics(time_vec, msd_history, burn_in_index, cfg);

    data = struct();
    data.time_vec = time_vec;
    data.step_vec = step_vec;
    data.saliency_all = saliency_all;
    data.saliency_mean = mean(saliency_all, 2);
    data.low_ratio = low_ratio;
    data.msd_history = msd_history;
    data.centroid_path = centroid_path;
    data.positions_history = positions_history;
    data.final_positions = sim.positions;
    data.final_theta = sim.theta;
    data.field_size = sim.fieldSize;
    data.active_ratio = active_ratio;
    data.P_value = P_value;
    data.D_value = D_value;
    data.fit_time = fit_time;
    data.fit_step = fit_time / sim.dt;
    data.fit_curve = fit_curve;
    data.burn_in_time = time_vec(burn_in_index);
    data.burn_in_step = burn_in_index - 1;
end

function [saliency_vec, ratio_low, avg_threshold] = capture_saliency_snapshot(sim)
    % 该函数从仿真对象中抽取当前帧的显著性与阈值占比信息
    if sim.useAdaptiveThreshold && ~isempty(sim.cj_threshold_dynamic)
        saliency_vec = sim.local_saliency_state(:)';
        thresholds_current = sim.cj_threshold_dynamic;
    else
        saliency_vec = zeros(1, sim.N);
        thresholds_current = repmat(sim.cj_threshold, sim.N, 1);
    end

    cfg = sim.adaptiveThresholdConfig;
    if isfield(cfg, 'cj_low')
        low_bound = cfg.cj_low;
    else
        low_bound = sim.cj_threshold;
    end
    ratio_low = mean(thresholds_current <= low_bound + 1e-6);  % 判定处于低阈值状态的粒子比例
    avg_threshold = mean(thresholds_current);                  % 记录整体阈值水平，便于诊断
end

function plot_responsiveness_state(ax, resp_data, style, colors)
    % 绘制响应性分析场景，包括不同类别粒子轨迹和最终指向
    hold(ax, 'on');  % 同一坐标系绘制轨迹与最终位置
    pos_hist = resp_data.positions_history;
    pos_final = resp_data.final_positions;
    theta_final = resp_data.final_theta;
    field_size = resp_data.field_size;

    mask_external = false(size(theta_final));
    if ~isempty(resp_data.external_indices)
        mask_external(resp_data.external_indices) = true;
    end
    mask_activated = resp_data.ever_activated;
    mask_activated(mask_external) = false;
    mask_base = ~(mask_activated | mask_external);

    draw_category_traj(ax, pos_hist, mask_base, colors.traj_gray_start, colors.traj_gray_end, 1.0, 0.25);      % 基础个体轨迹
    draw_category_traj(ax, pos_hist, mask_activated, colors.traj_gray_start, colors.traj_gray_end, 1.2, 0.32);  % 被连锁激活的个体轨迹
    draw_category_traj(ax, pos_hist, mask_external, colors.traj_red_start, colors.traj_red_end, 1.8, 0.55);     % 外部脉冲直接作用的个体

    draw_agents(ax, pos_final(mask_base, :), theta_final(mask_base), colors.base, style.shape);
    draw_agents(ax, pos_final(mask_activated, :), theta_final(mask_activated), colors.active_ratio, style.shape);
    draw_agents(ax, pos_final(mask_external, :), theta_final(mask_external), colors.external, style.shape);
    if ~isnan(resp_data.R_value)
        text(ax, field_size - 15, field_size - 4, sprintf('R = %.3f', resp_data.R_value), ...
            'FontName', style.font_name, 'FontWeight', 'bold', 'FontSize', 12, 'Color', 'r');
    end
    apply_state_style(ax, style, field_size);
    hold(ax, 'off');
end

function plot_persistence_state(ax, pers_data, style, colors)
    % 可视化持久性场景，展示群体整体迁移轨迹与质心迁移
    hold(ax, 'on');  % 绘制所有粒子的轨迹以及质心路径
    pos_hist = pers_data.positions_history;
    pos_final = pers_data.final_positions;
    theta_final = pers_data.final_theta;
    field_size = pers_data.field_size;

    draw_category_traj(ax, pos_hist, true(size(theta_final)), colors.traj_gray_start, colors.traj_gray_end, 1.0, 0.22);
    draw_agents(ax, pos_final, theta_final, colors.base, style.shape);
    plot(ax, pers_data.centroid_path(:,1), pers_data.centroid_path(:,2), '-', 'LineWidth', 1.6, ...
        'Color', [0.15 0.35 0.8]);
    scatter(ax, pers_data.centroid_path(1,1), pers_data.centroid_path(1,2), 55, [0.1 0.6 0.3], 'filled');   % 质心起点
    scatter(ax, pers_data.centroid_path(end,1), pers_data.centroid_path(end,2), 65, [0.85 0.2 0.2], 'filled'); % 质心终点
    if ~isnan(pers_data.P_value)
        text(ax, field_size - 15, field_size - 4, sprintf('P = %.3f', pers_data.P_value), ...
            'FontName', style.font_name, 'FontWeight', 'bold', 'FontSize', 12, 'Color', [0.15 0.35 0.8]);
    end

    apply_state_style(ax, style, field_size);
    hold(ax, 'off');
end

function plot_saliency_series(ax, time_vec, saliency_all, saliency_mean, mean_color, style)
    % 将单个个体的显著性轨迹与整体均值叠加，突出群体态势
    base_color = [0.75 0.75 0.75];
    hold_state = ishold(ax);
    hold(ax, 'on');
    plot(ax, time_vec, saliency_all, 'Color', base_color, 'LineWidth', 0.4, 'HandleVisibility', 'off');  % 灰色背景表示每个个体
    plot(ax, time_vec, saliency_mean, 'LineWidth', 2.4, 'Color', mean_color, 'HandleVisibility', 'off'); % 粗线强调全局均值

    legend_handles = gobjects(2,1);
    legend_handles(1) = plot(ax, NaN, NaN, 'Color', base_color, 'LineWidth', 0.8, ...
        'DisplayName', 'Indi. Variance');
    legend_handles(2) = plot(ax, NaN, NaN, 'Color', mean_color, 'LineWidth', 2.4, ...
        'DisplayName', 'Avg. Variance');
    legend(ax, legend_handles, 'Location', 'northwest', 'Box', 'on', ...
        'FontName', style.font_name, 'FontSize', style.tick_font_size - 1, 'Interpreter', 'none');

    if ~hold_state
        hold(ax, 'off');
    else
        hold(ax, 'on');
    end
end

function draw_category_traj(ax, pos_hist, mask, color_start, color_end, line_width, alpha)
    % 根据掩码选择不同类别粒子，绘制颜色渐变的完整轨迹
    if isempty(mask) || ~any(mask)
        return;
    end
    idx_list = find(mask);
    for k = idx_list(:)'
        traj = squeeze(pos_hist(:, k, :));
        if any(isnan(traj(:,1))) || any(isnan(traj(:,2)))
            continue;
        end
        draw_gradient_line(ax, traj(:,1), traj(:,2), color_start, color_end, line_width, alpha);
    end
end

function draw_agents(ax, positions, theta, color, shape)
    % 逐个绘制粒子形状，使最终状态细节更清晰
    if isempty(positions)
        return;
    end
    for idx = 1:size(positions, 1)
        if any(isnan(positions(idx, :))) || isnan(theta(idx))
            continue;
        end
        draw_agent_shape(ax, positions(idx, :), theta(idx), color, shape);
    end
end

function draw_agent_shape(ax, position, heading, color, shape)
    % 通过矩形尾翼 + 圆形头部组合，形成带方向性的“鱼”型粒子
    tail_corners = [
        0, -shape.tail_width/2;
        shape.tail_length, -shape.tail_width/2;
        shape.tail_length, shape.tail_width/2;
        0, shape.tail_width/2
        ];
    R = [cos(heading), -sin(heading); sin(heading), cos(heading)];
    rotated_tail = (R * tail_corners.').' + position;
    patch('Parent', ax, 'XData', rotated_tail(:,1), 'YData', rotated_tail(:,2), ...
        'FaceColor', color, 'EdgeColor', color, 'FaceAlpha', 0.85, 'EdgeAlpha', 0.85);

    t = linspace(0, 2*pi, shape.resolution);
    head_points = [shape.head_radius * cos(t); shape.head_radius * sin(t)];
    rotated_head = (R * head_points).' + position;
    patch('Parent', ax, 'XData', rotated_head(:,1), 'YData', rotated_head(:,2), ...
        'FaceColor', color, 'EdgeColor', color, 'FaceAlpha', 0.95, 'EdgeAlpha', 0.95);
end

function draw_gradient_line(ax, x, y, color_start, color_end, line_width, edge_alpha)
    % 通过 patch 将轨迹拆分为线段并插值颜色，呈现方向感
    if numel(x) < 2 || numel(y) < 2
        return;
    end
    x = x(:);
    y = y(:);
    if any(isnan(x)) || any(isnan(y))
        return;
    end
    if nargin < 7 || isempty(edge_alpha)
        edge_alpha = 1.0;
    end
    edge_alpha = min(max(edge_alpha, 0), 1);
    num_points = numel(x);
    t = linspace(0, 1, num_points - 1).';
    colors = (1 - t) .* color_start + t .* color_end;
    for idx = 1:num_points-1  % 将轨迹拆分为线段并逐段插入，从而实现颜色渐变
        patch('Parent', ax, 'XData', [x(idx), x(idx+1)], 'YData', [y(idx), y(idx+1)], ...
            'ZData', [0, 0], 'FaceColor', 'none', 'EdgeColor', colors(idx, :), ...
            'LineWidth', line_width, 'EdgeAlpha', edge_alpha);
    end
end

function add_time_marker(ax, marker_cfg)
    % 在时间序列图中统一绘制稳定化结束步的竖线，避免读图歧义
    if nargin < 2 || isempty(marker_cfg)
        return;
    end
    if ~isfield(marker_cfg, 'step') || isempty(marker_cfg.step)
        return;
    end
    if isfield(marker_cfg, 'color') && ~isempty(marker_cfg.color)
        color = marker_cfg.color;
    else
        color = [0.3 0.3 0.3];
    end
    if isfield(marker_cfg, 'line_style') && ~isempty(marker_cfg.line_style)
        line_style = marker_cfg.line_style;
    else
        line_style = '--';
    end
    if isfield(marker_cfg, 'line_width') && ~isempty(marker_cfg.line_width)
        line_width = marker_cfg.line_width;
    else
        line_width = 1.0;
    end
    h = xline(ax, marker_cfg.step, line_style, 'Color', color, 'LineWidth', line_width);
    if isgraphics(h)
        h.Annotation.LegendInformation.IconDisplayStyle = 'off';
    end
end

function apply_state_style(ax, style, field_size)
    % 限定场景坐标系、加粗坐标轴，确保打印版图像清晰
    lim = min(50, field_size);
    axis(ax, [0 lim 0 lim]);
    xticks(ax, 0:10:lim);
    yticks(ax, 0:10:lim);
    axis(ax, 'square');
    grid(ax, 'off');
    ax.LineWidth = style.axis_line_width;
    ax.FontName = style.font_name;
    ax.FontSize = style.tick_font_size;
    ax.FontWeight = style.tick_font_weight;
    ax.TickDir = 'in';
    xlabel(ax, 'X', 'FontName', style.font_name, 'FontSize', style.label_font_size, 'FontWeight', style.label_font_weight);
    ylabel(ax, 'Y', 'FontName', style.font_name, 'FontSize', style.label_font_size, 'FontWeight', style.label_font_weight);
    box(ax, 'on');
end

function apply_series_style(ax, style, ylabel_txt)
    % 对所有时间序列类图统一设置字体/刻度/坐标轴样式
    grid(ax, 'off');
    ax.LineWidth = style.axis_line_width;
    ax.FontName = style.font_name;
    ax.FontSize = style.tick_font_size;
    ax.FontWeight = style.tick_font_weight;
    ax.TickDir = 'in';
    xlabel(ax, 'Time', 'FontName', style.font_name, 'FontSize', style.label_font_size, 'FontWeight', style.label_font_weight);
    ylabel(ax, ylabel_txt, 'FontName', style.font_name, 'FontSize', style.label_font_size, 'FontWeight', style.label_font_weight);
    box(ax, 'on');
end

function export_and_close(fig_handle, file_path)
    exportgraphics(fig_handle, file_path, 'ContentType', 'vector');  % 默认导出矢量，便于后期排版
    fprintf('Saved figure: %s\n', file_path);
end

function V = compute_average_velocity(theta, v0)
    % 根据所有粒子朝向求得群体平均速度向量
    V = [mean(v0 * cos(theta)), mean(v0 * sin(theta))];
end

function [P_value, D_value, fit_time, fit_curve] = compute_persistence_metrics(time_vec, msd, burn_in_index, cfg)
    % 依据 MSD 曲线拟合扩散系数，再反推出持久性指标 P
    x = time_vec(burn_in_index:end);
    y = msd(burn_in_index:end);

    D_value = NaN;
    fit_time = x;
    fit_curve = NaN(size(x));

    if numel(x) >= max(2, cfg.min_fit_points) && ~all(abs(y - y(1)) < eps)
        x_shift = x - x(1);
        y_shift = y - y(1);
        if any(x_shift > 0) && any(abs(y_shift) > eps)
            smooth_window = max(5, floor(numel(y_shift) * 0.1));  % 使用滑动平均降低噪声
            if smooth_window > 1
                y_shift = smoothdata(y_shift, 'movmean', smooth_window);
            end
            slope = lsqnonneg(x_shift(:), y_shift(:));
            if slope > 0
                D_value = slope / 4;  % 二维扩散修正
                fit_curve = y(1) + slope * x_shift;
            end
        end
    end

    if isnan(D_value)
        P_value = NaN;
        fit_curve(:) = NaN;
    else
        D_value = max(D_value, cfg.min_diffusion);  % 避免零扩散导致的发散
        P_value = 1 / sqrt(D_value);
    end
end
