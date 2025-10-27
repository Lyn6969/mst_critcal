% visualize_cluster_responsiveness 集群响应性观察脚本
% =========================================================================
% 作用：
%   - 复用 c1 级联实验场景（单个外源种子），按 high order 定义计算响应性
%     R：统计群体平均速度与领导方向的投影积分，再对多次随机方向求平均。
%   - 提供可视化：左侧显示粒子运动；右侧展示 V(t)·n_phi 投影、
%     单次 r(phi) 曲线以及最终响应性统计结果。
%
% 使用方式：
%   - 直接运行脚本；可根据提示输入要采样的方向数与重复次数。
%
% =========================================================================

clc;
clear;
close all;

%% 1. 输入与参数 -----------------------------------------------------------------
fprintf('=== 集群响应性观察 ===\n');

num_runs = input('请输入重复次数（默认 5 ）: ');
if isempty(num_runs)
    num_runs = 5;
end

num_angles = input('请输入采样方向个数（默认 8 ）: ');
if isempty(num_angles)
    num_angles = 8;
end

fprintf('实验参数：重复 %d 次，每次采样 %d 个方向。\n', num_runs, num_angles);

% 从 delta_c 场景继承的默认参数
base_params = struct();
base_params.N = 200;
base_params.rho = 1;
base_params.v0 = 1;
base_params.angleUpdateParameter = 10;
base_params.angleNoiseIntensity = 0.05;
base_params.T_max = 400;
base_params.dt = 0.1;
base_params.radius = 5;
base_params.deac_threshold = 0.1745;
base_params.cj_threshold = 1.0;
base_params.fieldSize = 50;
base_params.initDirection = pi/4;
base_params.useFixedField = true;
base_params.stabilization_steps = 200;
base_params.forced_turn_duration = 200;

%% 2. 可视化窗口 -----------------------------------------------------------------
fig = figure('Name', '集群响应性演示', 'Color', 'white', ...
    'Position', [80, 80, 1400, 700]);

ax_particles = subplot('Position', [0.05 0.3 0.4 0.65]);
ax_proj = subplot('Position', [0.5 0.65 0.45 0.25]);
ax_resp = subplot('Position', [0.5 0.35 0.45 0.25]);
ax_stats = subplot('Position', [0.5 0.08 0.45 0.2]);

%% 3. 响应性计算准备 -------------------------------------------------------------
time_vec = (0:base_params.T_max)' * base_params.dt;
projection_curves = cell(num_runs, 1);
r_curves = cell(num_runs, 1);
R_values = NaN(num_runs, 1);

%% 4. 主循环 --------------------------------------------------------------------
for run_idx = 1:num_runs
    params = base_params;
    rng(run_idx);  % 固定随机性
    sim = ParticleSimulationWithExternalPulse(params);
    sim.external_pulse_count = 1;
    sim.resetCascadeTracking();
    sim.initializeParticles();

    phi_list = linspace(0, pi, num_angles);
    phi_list(end) = [];
    phi_list = phi_list + pi/(2*num_angles);
    n_vectors = [cos(phi_list); sin(phi_list)];

    V_history = zeros(base_params.T_max + 1, 2);
    V_history(1, :) = compute_average_velocity(sim.theta, params.v0);

    subplot(ax_particles);
    cla(ax_particles);
    axis(ax_particles, [0 params.fieldSize 0 params.fieldSize]);
    axis(ax_particles, 'square');
    grid(ax_particles, 'on');
    title(ax_particles, sprintf('运行 %d / %d', run_idx, num_runs));
    xlabel(ax_particles, 'X');
    ylabel(ax_particles, 'Y');
    particles_plot = scatter(sim.positions(:,1), sim.positions(:,2), 36, sim.theta, 'filled');
    hold(ax_particles, 'on');
    arrows_plot = quiver(sim.positions(:,1), sim.positions(:,2), cos(sim.theta), sin(sim.theta), 0.4, ...
        'Color', [0.3 0.3 0.3 0.6], 'LineWidth', 1.0);
    hold(ax_particles, 'off');

    triggered = false;
    projection_history = zeros(base_params.T_max + 1, num_angles);
    r_history = zeros(base_params.T_max + 1, num_angles);

    for t = 1:base_params.T_max
        sim.step();
        V_history(t + 1, :) = compute_average_velocity(sim.theta, params.v0);
        projection_history(t + 1, :) = V_history(t + 1, :) * n_vectors;

        if ~triggered && sim.external_pulse_triggered
            triggered = true;
            t_start = t;
        end

        if mod(t, 5) == 0
            subplot(ax_particles);
            set(particles_plot, 'XData', sim.positions(:,1), 'YData', sim.positions(:,2), 'CData', sim.theta);
            set(arrows_plot, 'XData', sim.positions(:,1), 'YData', sim.positions(:,2), ...
                'UData', cos(sim.theta), 'VData', sin(sim.theta));
            drawnow limitrate;
        end
    end

    if ~triggered
        warning('运行 %d 未检测到外源脉冲触发，跳过。', run_idx);
        continue;
    end

    dt = params.dt;
    v0 = params.v0;
    T_window = params.forced_turn_duration;
    t_end = min(t_start + T_window, base_params.T_max);

    for angle_idx = 1:num_angles
        proj = projection_history(:, angle_idx);
        integral_value = trapz(time_vec(t_start:t_end), proj(t_start:t_end));
        duration = time_vec(t_end) - time_vec(t_start);
        if duration <= 0
            r_history(angle_idx) = NaN;
        else
            r_history(angle_idx) = integral_value / (v0 * duration);
        end
    end

    R_values(run_idx) = mean(r_history(~isnan(r_history)));
    projection_curves{run_idx} = projection_history;
    r_curves{run_idx} = r_history;

    subplot(ax_proj);
    cla(ax_proj);
    plot(ax_proj, time_vec, projection_history);
    xlabel(ax_proj, '时间');
    ylabel(ax_proj, 'V(t) · n_φ');
    title(ax_proj, '投影曲线');

    subplot(ax_resp);
    cla(ax_resp);
    bar(ax_resp, r_history);
    xlabel(ax_resp, '方向样本');
    ylabel(ax_resp, 'r(φ)');
    title(ax_resp, sprintf('单次 r(φ) (R=%.3f)', R_values(run_idx)));
end

subplot(ax_stats);
cla(ax_stats);
valid_R = R_values(~isnan(R_values));
if isempty(valid_R)
    text(0.1, 0.5, '无有效响应数据', 'Parent', ax_stats, 'FontSize', 12);
else
    histogram(ax_stats, valid_R, 'FaceColor', [0.2 0.6 0.8]);
    xlabel(ax_stats, 'R');
    ylabel(ax_stats, '频数');
    title(ax_stats, sprintf('R 分布 (均值 %.3f)', mean(valid_R)));
end

fprintf('响应性统计完成。有效样本 %d 个，平均 R = %.3f\n', numel(valid_R), mean(valid_R));

%% ========================================================================
function V = compute_average_velocity(theta, v0)
    V = [mean(v0 * cos(theta)), mean(v0 * sin(theta))];
end
