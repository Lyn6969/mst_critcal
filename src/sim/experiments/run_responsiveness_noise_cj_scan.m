function run_responsiveness_noise_cj_scan()
%RUN_RESPONSIVENESS_NOISE_CJ_SCAN 扫描噪声×阈值对响应性 R 的影响
%
% 说明：
%   - 复用 run_cj_tradeoff_scan_shared_seed 中的响应性计算流程
%   - 同时遍历运动显著性阈值 cj_threshold 与角度噪声强度 angleNoiseIntensity
%   - 每个参数组合重复多次实验，统计响应性 R 的均值 / 标准差 / 标准误差
%   - 输出热力图 + MAT 数据，便于进一步分析

clc; clear; close all;
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..')));

fprintf('=================================================\n');
fprintf('   噪声 × 运动显著性阈值 扫描：响应性 R\n');
fprintf('=================================================\n\n');

%% 1. 基础仿真参数（与 tradeoff 脚本保持一致）
resp_params = struct();
resp_params.N = 200;
resp_params.rho = 1;
resp_params.v0 = 1;
resp_params.angleUpdateParameter = 10;
resp_params.angleNoiseIntensity = 0.05;   % 将被扫描覆盖
resp_params.T_max = 400;
resp_params.dt = 0.1;
resp_params.radius = 5;
resp_params.deac_threshold = 0.1745;
resp_params.cj_threshold = 1.5;           % 将被扫描覆盖
resp_params.fieldSize = 50;
resp_params.initDirection = pi/4;
resp_params.useFixedField = true;
resp_params.stabilization_steps = 200;
resp_params.forced_turn_duration = 200;
resp_params.useAdaptiveThreshold = false;

num_angles = 1;                            % 只沿领导方向投影
time_vec = (0:resp_params.T_max)' * resp_params.dt;

%% 2. 扫描范围与试验配置
cj_thresholds = 0.0:0.1:5.0;
noise_levels = 0.0:0.01:0.;
num_runs = 40;
base_seed = 20251104;

num_cj = numel(cj_thresholds);
num_noise = numel(noise_levels);
total_tasks = num_cj * num_noise;

fprintf('cj_threshold 点数: %d | 噪声点数: %d | 每组合重复: %d\n', ...
    num_cj, num_noise, num_runs);

%% 3. 结果容器
R_mean = NaN(num_noise, num_cj);
R_std = NaN(num_noise, num_cj);
R_sem = NaN(num_noise, num_cj);
fail_rates = NaN(num_noise, num_cj);

%% 4. 并行配置
pool = configure_parallel_pool();
fprintf('使用并行池: %d workers\n\n', pool.NumWorkers);

progress_queue = parallel.pool.DataQueue;
progress = struct('total', total_tasks, 'completed', 0, ...
    'start', tic, 'last', tic);
afterEach(progress_queue, @(info) report_progress(info));

shared_params = parallel.pool.Constant(resp_params);
shared_cj = parallel.pool.Constant(cj_thresholds);
shared_noise = parallel.pool.Constant(noise_levels);

%% 5. 并行扫描
parfor cj_idx = 1:num_cj
    local_R_mean = NaN(num_noise, 1);
    local_R_std = NaN(num_noise, 1);
    local_R_sem = NaN(num_noise, 1);
    local_fail = NaN(num_noise, 1);

    params_const = shared_params.Value;
    cj_value = shared_cj.Value(cj_idx);
    noise_array = shared_noise.Value;

    for noise_idx = 1:num_noise
        params_curr = params_const;
        params_curr.cj_threshold = cj_value;
        params_curr.angleNoiseIntensity = noise_array(noise_idx);

        R_runs = NaN(num_runs, 1);
        fail_count = 0;

        for run_idx = 1:num_runs
            seed_val = base_seed + cj_idx * 1e6 + noise_idx * 1e4 + run_idx * 97;
            [R_val, triggered] = run_single_responsiveness_trial(params_curr, num_angles, time_vec, seed_val);
            if ~triggered || isnan(R_val)
                fail_count = fail_count + 1;
            end
            R_runs(run_idx) = R_val;
        end

        local_R_mean(noise_idx) = mean(R_runs, 'omitnan');
        local_R_std(noise_idx) = std(R_runs, 0, 'omitnan');
        local_R_sem(noise_idx) = local_R_std(noise_idx) ./ sqrt(num_runs);
        local_fail(noise_idx) = fail_count / num_runs;

        send(progress_queue, struct('cj_idx', cj_idx, 'noise_idx', noise_idx));
    end

    R_mean(:, cj_idx) = local_R_mean;
    R_std(:, cj_idx) = local_R_std;
    R_sem(:, cj_idx) = local_R_sem;
    fail_rates(:, cj_idx) = local_fail;
end

%% 6. 可视化
figure('Color', 'white', 'Position', [160, 100, 900, 540]);
imagesc(cj_thresholds, noise_levels, R_mean);
axis xy;
xlabel('c_j threshold');
ylabel('angle noise \sigma');
title('响应性 R (均值)');
cb = colorbar; cb.Label.String = 'R';
grid on;

figure('Color', 'white', 'Position', [180, 120, 900, 540]);
imagesc(cj_thresholds, noise_levels, fail_rates);
axis xy;
xlabel('c_j threshold');
ylabel('angle noise \sigma');
title('触发失败率');
cb = colorbar; cb.Label.String = 'failure rate';
grid on;

%% 7. 保存结果
results = struct();
results.description = 'Responsiveness scan over noise × cj threshold';
results.timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
results.generated_at = datetime('now');
results.resp_params = resp_params;
results.cj_thresholds = cj_thresholds;
results.noise_levels = noise_levels;
results.num_runs = num_runs;
results.R_mean = R_mean;
results.R_std = R_std;
results.R_sem = R_sem;
results.fail_rates = fail_rates;

output_dir = fullfile('data', 'experiments', 'responsiveness_noise_cj_scan', results.timestamp);
if ~isfolder(output_dir), mkdir(output_dir); end

save(fullfile(output_dir, 'results.mat'), 'results', '-v7.3');
savefig(gcf, fullfile(output_dir, 'failure_rate.fig'));
print(fullfile(output_dir, 'failure_rate.png'), '-dpng', '-r300');

fprintf('数据已保存至: %s\n', output_dir);
fprintf('实验完成。\n');

%% ------------------------------------------------------------------------
    function report_progress(info)
        progress.completed = progress.completed + 1;
        if toc(progress.last) < 2 && progress.completed < progress.total
            return;
        end
        elapsed = toc(progress.start);
        eta = elapsed / progress.completed * (progress.total - progress.completed);
        fprintf('  进度: %d/%d (%.1f%%) | cj #%d, noise #%d | 已用 %.1f 分 | 预计剩余 %.1f 分\n', ...
            progress.completed, progress.total, ...
            100 * progress.completed / progress.total, ...
            info.cj_idx, info.noise_idx, elapsed/60, eta/60);
        progress.last = tic;
    end

end

%% ========================================================================
% 辅助函数：运行单次响应性实验（复用 tradeoff 脚本逻辑）
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
            if isempty(leader_idx), leader_idx = 1; end
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

function V = compute_average_velocity(theta, v0)
    V = [mean(v0 * cos(theta)), mean(v0 * sin(theta))];
end

function pool = configure_parallel_pool()
    if ~license('test', 'Distrib_Computing_Toolbox')
        error('需要 Parallel Computing Toolbox 才能运行此脚本。');
    end
    pool = gcp('nocreate');
    if isempty(pool)
        cluster = parcluster('local');
        max_workers = min(cluster.NumWorkers, 40);
        if max_workers < 1
            error('没有可用的并行工作线程。');
        end
        pool = parpool(cluster, max_workers);
    end
end
