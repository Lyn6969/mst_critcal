% run_cj_tradeoff_scan
% =========================================================================
% 目的：
%   - 扫描运动显著性阈值 cj_threshold ∈ [0, 5]，观察其对集群响应性 R 与
%     持久性 P 的影响，从而直观展示两者之间的权衡关系。
%   - 每个参数点重复 50 次实验，统计均值与标准误差，并输出权衡图。
%
% 数据指标：
%   - 响应性 R：复用外源脉冲实验，按 high order 定义进行投影积分。
%   - 持久性 P：在无外源脉冲条件下测量相对质心扩散，P = 1/sqrt(D)。
%
% 输出：
%   - 控制台进度与汇总信息
%   - results/tradeoff/ 目录下的 MAT 数据文件
%   - 带误差棒的 R-P 权衡散点图（颜色编码参数值）
%
% 使用方式：
%   直接运行脚本，可视情况调整下方的参数配置。
% =========================================================================

clc;
clear;
close all;

%% 1. 全局参数配置 ---------------------------------------------------------------
fprintf('=================================================\n');
fprintf('   运动显著性阈值扫描：响应性 vs 持久性\n');
fprintf('=================================================\n\n');

% 公共仿真参数（ParticleSimulation / ParticleSimulationWithExternalPulse 共用字段）
base_common = struct();
base_common.N = 200;
base_common.rho = 1;
base_common.v0 = 1;
base_common.angleUpdateParameter = 10;
base_common.angleNoiseIntensity = 0.05;   % 适度噪声，才能观察权衡趋势
base_common.T_max = 600;
base_common.dt = 0.1;
base_common.radius = 5;
base_common.deac_threshold = 0.1745;
base_common.cj_threshold = 1.0;           % 将在循环中覆盖
base_common.fieldSize = 50;
base_common.initDirection = pi/4;
base_common.useFixedField = true;

% 响应性专用参数
resp_params = base_common;
resp_params.stabilization_steps = 200;
resp_params.forced_turn_duration = 200;

% 持久性评估专用参数（无需外源设置）
pers_params = base_common;
pers_params.T_max = 800;                 % 更长时间窗口更稳定

% 参数扫描与重复次数
cj_thresholds = 0.0:0.1:5.0;
num_params = numel(cj_thresholds);
num_runs = 50;
num_angles = 1;                          % 只取领导方向

% 持久性拟合的一些配置
pers_cfg = struct();
pers_cfg.burn_in_ratio = 0.25;
pers_cfg.min_diffusion = 1e-4;
pers_cfg.min_fit_points = 40;

% 结果容器
time_vec = (0:resp_params.T_max)' * resp_params.dt;
R_raw = NaN(num_params, num_runs);
P_raw = NaN(num_params, num_runs);
trigger_failures = zeros(num_params, 1);
diffusion_values = NaN(num_params, num_runs);

% 结果输出目录
results_dir = fullfile('results', 'tradeoff');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end
timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
output_mat = fullfile(results_dir, sprintf('cj_tradeoff_%s.mat', timestamp));
output_fig = fullfile(results_dir, sprintf('cj_tradeoff_%s.png', timestamp));

fprintf('扫描参数点: %d，重复次数: %d，噪声强度: %.3f\n\n', ...
    num_params, num_runs, base_common.angleNoiseIntensity);

%% 2. 主循环 ---------------------------------------------------------------------
base_seed = 20250301;
loop_tic = tic;

for param_idx = 1:num_params
    current_cj = cj_thresholds(param_idx);
    resp_params.cj_threshold = current_cj;
    pers_params.cj_threshold = current_cj;

    fprintf('参数 %02d/%02d: cj_threshold = %.2f\n', param_idx, num_params, current_cj);
    param_tic = tic;

    for run_idx = 1:num_runs
        seed_base = base_seed + (param_idx - 1) * num_runs + run_idx;

        % 响应性试验
        [R_value, triggered] = run_single_responsiveness_trial(resp_params, num_angles, time_vec, seed_base);
        R_raw(param_idx, run_idx) = R_value;
        if ~triggered || isnan(R_value)
            trigger_failures(param_idx) = trigger_failures(param_idx) + 1;
        end

        % 持久性试验（使用不同随机序列避免交叉影响）
        [P_value, D_value] = run_single_persistence_trial(pers_params, pers_cfg, seed_base + 10_000);
        P_raw(param_idx, run_idx) = P_value;
        diffusion_values(param_idx, run_idx) = D_value;

        if mod(run_idx, 10) == 0 || run_idx == num_runs
            fprintf('  - 运行 %02d/%02d：R=%.3f, P=%.3f\n', run_idx, num_runs, R_value, P_value);
        end
    end

    fprintf('    完成。触发失败 %d 次，用时 %.1fs\n', ...
        trigger_failures(param_idx), toc(param_tic));
end

total_minutes = toc(loop_tic) / 60;
fprintf('\n全部实验完成，总耗时 %.2f 分钟\n\n', total_minutes);

%% 3. 统计分析 -------------------------------------------------------------------
R_mean = mean(R_raw, 2, 'omitnan');
R_std = std(R_raw, 0, 2, 'omitnan');
R_sem = R_std ./ sqrt(num_runs);

P_mean = mean(P_raw, 2, 'omitnan');
P_std = std(P_raw, 0, 2, 'omitnan');
P_sem = P_std ./ sqrt(num_runs);

D_mean = mean(diffusion_values, 2, 'omitnan');

%% 4. 绘制权衡图 -----------------------------------------------------------------
figure('Name', 'cj 阈值调节响应性-持久性权衡', 'Color', 'white', 'Position', [120, 120, 900, 600]);
hold on;

scatter_handle = scatter(R_mean, P_mean, 70, cj_thresholds, 'filled');
colormap(parula);
cb = colorbar;
cb.Label.String = 'cj\_threshold';

for idx = 1:num_params
    if isnan(R_mean(idx)) || isnan(P_mean(idx))
        continue;
    end
    errorbar(R_mean(idx), P_mean(idx), P_sem(idx), P_sem(idx), R_sem(idx), R_sem(idx), ...
        'Color', [0.35 0.35 0.35], 'LineWidth', 0.9, 'CapSize', 6);
end

plot(R_mean, P_mean, '-', 'Color', [0.25 0.45 0.8], 'LineWidth', 1.2);

xlabel('响应性 R');
ylabel('持久性 P');
title('运动显著性阈值下的响应性-持久性权衡');
grid on;
set(gca, 'FontSize', 11);

text(R_mean(1), P_mean(1), '  cj=0.0', 'Color', [0.2 0.2 0.2]);
text(R_mean(end), P_mean(end), sprintf('  cj=%.1f', cj_thresholds(end)), ...
    'Color', [0.2 0.2 0.2]);

saveas(gcf, output_fig);
fprintf('图像已保存至: %s\n', output_fig);

%% 5. 保存数据 --------------------------------------------------------------------
results = struct();
results.description = 'cj threshold trade-off scan';
results.timestamp = timestamp;
results.parameters = struct('resp', resp_params, 'pers', pers_params);
results.cj_thresholds = cj_thresholds;
results.num_runs = num_runs;
results.num_angles = num_angles;
results.base_seed = base_seed;
results.trigger_failures = trigger_failures;
results.R_raw = R_raw;
results.P_raw = P_raw;
results.D_raw = diffusion_values;
results.R_mean = R_mean;
results.R_std = R_std;
results.R_sem = R_sem;
results.P_mean = P_mean;
results.P_std = P_std;
results.P_sem = P_sem;
results.D_mean = D_mean;
results.total_minutes = total_minutes;
results.matlab_version = version;

save(output_mat, 'results', '-v7.3');
fprintf('数据已保存至: %s\n', output_mat);

%% ========================================================================
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
