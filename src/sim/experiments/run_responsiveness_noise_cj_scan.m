%RUN_RESPONSIVENESS_NOISE_CJ_SCAN 扫描噪声×阈值对响应性 R 的影响
%
% 说明：
%   - 复用 run_cj_tradeoff_scan_shared_seed 中的响应性计算流程
%   - 同时遍历运动显著性阈值 cj_threshold 与角度噪声强度 angleNoiseIntensity
%   - 每个参数组合重复多次实验，统计响应性 R 的均值 / 标准差 / 标准误差
%   - 输出热力图 + MAT 数据，便于进一步分析
%
% 脚本模式：便于调试和参数调整

clc; clear; close all;
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..')));

fprintf('=================================================\n');
fprintf('   噪声 × 运动显著性阈值 扫描：响应性 R\n');
fprintf('=================================================\n\n');

% === 调试信息：脚本开始执行 ===
fprintf('脚本开始执行 - %s\n', char(datetime('now')));
fprintf('工作目录: %s\n', pwd);
disp(' ');

%% 1. 基础仿真参数（与 tradeoff 脚本保持一致）
fprintf('配置基础仿真参数...\n');
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

% === 调试信息：参数配置显示 ===
fprintf('基础仿真参数配置完成:\n');
fprintf('  - 粒子数量 N: %d\n', resp_params.N);
fprintf('  - 仿真步数 T_max: %d\n', resp_params.T_max);
fprintf('  - 时间步长 dt: %.2f\n', resp_params.dt);
fprintf('  - 邻居半径: %.1f\n', resp_params.radius);
disp(' ');

num_angles = 1;                            % 只沿领导方向投影
time_vec = (0:resp_params.T_max)' * resp_params.dt;

%% 2. 扫描范围与试验配置
fprintf('配置扫描范围...\n');
cj_thresholds = 0.0:0.1:5.0;
noise_levels = 0.0:0.0025:0.125;
eta_levels = sqrt(2 * noise_levels);  % 根据论文定义 \eta = sqrt(2 D_\theta)
num_runs = 50;
base_seed = 20251104;

num_cj = numel(cj_thresholds);
num_noise = numel(noise_levels);
total_tasks = num_cj * num_noise;

% 预生成参数组合（向量形式，便于线性索引并行）
[cj_mesh, noise_mesh] = meshgrid(cj_thresholds, noise_levels);
cj_vector = cj_mesh(:);
noise_vector = noise_mesh(:);
cj_idx_vector = repmat(1:num_cj, num_noise, 1);
cj_idx_vector = cj_idx_vector(:);
noise_idx_vector = repelem((1:num_noise)', num_cj);

% === 调试信息：扫描参数显示 ===
fprintf('扫描参数配置:\n');
fprintf('  - cj_threshold 范围: [%.1f, %.1f], 步长: %.1f, 点数: %d\n', ...
    cj_thresholds(1), cj_thresholds(end), cj_thresholds(2)-cj_thresholds(1), num_cj);
fprintf('  - noise_levels 范围: [%.2f, %.2f], 步长: %.2f, 点数: %d\n', ...
    noise_levels(1), noise_levels(end), noise_levels(2)-noise_levels(1), num_noise);
fprintf('  - 对应 \eta 范围: [%.3f, %.3f]\n', eta_levels(1), eta_levels(end));
fprintf('  - 每参数组合重复次数: %d\n', num_runs);
fprintf('  - 总实验次数: %d × %d × %d = %d\n', num_cj, num_noise, num_runs, total_tasks);
fprintf('  - 预计总耗时(估算): %.1f 小时(假设每次实验0.1秒)\n', total_tasks * 0.1 / 3600);
disp(' ');

%% 3. 结果容器
fprintf('初始化结果存储容器...\n');
R_mean_lin = NaN(total_tasks, 1);
R_std_lin = NaN(total_tasks, 1);
R_sem_lin = NaN(total_tasks, 1);

% === 调试信息：结果容器初始化 ===
fprintf('结果矩阵大小: %d × %d (噪声×阈值)\n', num_noise, num_cj);
fprintf('线性任务总数: %d\n', total_tasks);
disp(' ');

%% 4. 并行配置
fprintf('配置并行计算环境...\n');
pool = configure_parallel_pool();
fprintf('使用并行池: %d workers\n\n', pool.NumWorkers);

progress_queue = parallel.pool.DataQueue;
progress_key = 'run_resp_noise_cj_progress';
start_token = tic;
progress_state = struct('total', total_tasks, 'completed', 0, ...
    'start', start_token, 'last', start_token);
setappdata(0, progress_key, progress_state);
afterEach(progress_queue, @(info) report_progress(info, progress_key, false));

shared_params = parallel.pool.Constant(resp_params);

% === 调试信息：开始并行扫描 ===
fprintf('开始参数扫描...\n');
fprintf('开始时间: %s\n', char(datetime('now')));
disp(' ');

%% 5. 并行扫描
scan_clock = start_token;

parfor task_idx = 1:total_tasks
    params_curr = shared_params.Value;
    params_curr.cj_threshold = cj_vector(task_idx);
    params_curr.angleNoiseIntensity = noise_vector(task_idx);

    R_runs = NaN(num_runs, 1);
    cj_idx = cj_idx_vector(task_idx);
    noise_idx = noise_idx_vector(task_idx);

    for run_idx = 1:num_runs
        seed_val = base_seed + task_idx * 1e6 + run_idx * 97;
        [R_val, ~] = run_single_responsiveness_trial(params_curr, num_angles, time_vec, seed_val);
        R_runs(run_idx) = R_val;
    end

    R_mean_lin(task_idx) = mean(R_runs, 'omitnan');
    R_std_lin(task_idx) = std(R_runs, 0, 'omitnan');
    R_sem_lin(task_idx) = R_std_lin(task_idx) ./ sqrt(num_runs);

    send(progress_queue, struct('cj_idx', cj_idx, 'noise_idx', noise_idx));
end

fprintf('参数扫描完成，开始处理结果...\n');
disp(' ');

R_mean = reshape(R_mean_lin, num_noise, num_cj);
R_std = reshape(R_std_lin, num_noise, num_cj);
R_sem = reshape(R_sem_lin, num_noise, num_cj);

if isappdata(0, progress_key)
    report_progress(struct('cj_idx', cj_idx_vector(end), 'noise_idx', noise_idx_vector(end)), progress_key, true);
end

%% 6. 可视化
fprintf('生成可视化图表...\n');

% === 调试信息：结果统计 ===
fprintf('扫描结果统计:\n');
fprintf('  - 响应性 R 均值范围: [%.3f, %.3f]\n', min(R_mean(:), [], 'omitnan'), max(R_mean(:), [], 'omitnan'));
fprintf('  - 有效数据点数: %d / %d\n', sum(~isnan(R_mean(:))), numel(R_mean));
fprintf('  - 响应性标准差均值: %.3f\n', mean(R_std(:), 'omitnan'));
disp(' ');

h_mean = figure('Color', 'white', 'Position', [160, 100, 900, 540]);
imagesc(cj_thresholds, eta_levels, R_mean);
axis xy;
xlabel('M_T');
ylabel('\eta');
title('响应性 R (均值)');
cb = colorbar; cb.Label.String = 'R';
grid on;

%% 7. 保存结果
results = struct();
results.description = 'Responsiveness scan over noise × cj threshold';
results.timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
results.generated_at = datetime('now');
results.resp_params = resp_params;
results.cj_thresholds = cj_thresholds;
results.noise_levels = noise_levels;
results.eta_levels = eta_levels;
results.num_runs = num_runs;
results.R_mean = R_mean;
results.R_std = R_std;
results.R_sem = R_sem;

output_dir = fullfile('data', 'experiments', 'responsiveness_noise_cj_scan', results.timestamp);
if ~isfolder(output_dir), mkdir(output_dir); end

save(fullfile(output_dir, 'results.mat'), 'results', '-v7.3');
savefig(h_mean, fullfile(output_dir, 'responsiveness_mean.fig'));
print(h_mean, fullfile(output_dir, 'responsiveness_mean.png'), '-dpng', '-r300');

fprintf('数据已保存至: %s\n', output_dir);
fprintf('实验完成。\n');

% === 调试信息：脚本执行完成 ===
fprintf('脚本执行完成 - %s\n', char(datetime('now')));
fprintf('总耗时: %.2f 分钟\n', toc(scan_clock)/60);
disp(' ');

if isappdata(0, progress_key)
    rmappdata(0, progress_key);
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

function report_progress(info, progress_key, force_output)
    if ~isappdata(0, progress_key)
        return;
    end
    progress = getappdata(0, progress_key);
    if nargin < 3
        force_output = false;
    end

    if force_output
        completed_count = progress.completed;
    else
        progress.completed = progress.completed + 1;
        completed_count = progress.completed;
    end

    if ~force_output && toc(progress.last) < 2 && completed_count < progress.total
        setappdata(0, progress_key, progress);
        return;
    end

    elapsed = toc(progress.start);
    remaining = max(progress.total - completed_count, 0);
    if completed_count > 0
        eta = elapsed / completed_count * remaining;
    else
        eta = NaN;
    end

    pct = 100 * completed_count / progress.total;
    fprintf('  进度: %d/%d (%.1f%%) | cj #%d, noise #%d | 已用 %.1f 分 | 预计剩余 %.1f 分\n', ...
        min(completed_count, progress.total), progress.total, pct, ...
        info.cj_idx, info.noise_idx, elapsed/60, eta/60);

    progress.last = tic;
    setappdata(0, progress_key, progress);
end

function pool = configure_parallel_pool()
    if ~license('test', 'Distrib_Computing_Toolbox')
        error('需要 Parallel Computing Toolbox 才能运行此脚本。');
    end
    pool = gcp('nocreate');
    if isempty(pool)
        cluster = parcluster('local');
        max_workers = min(cluster.NumWorkers, 200);
        if max_workers < 1
            error('没有可用的并行工作线程。');
        end
        pool = parpool(cluster, max_workers);
    end
end
