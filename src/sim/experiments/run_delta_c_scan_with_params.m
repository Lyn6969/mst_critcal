function results = run_delta_c_scan_with_params(num_runs, N, noise_intensity, num_workers)
%RUN_DELTA_C_SCAN_WITH_PARAMS 带参数的Delta_c扫描实验（原函数的参数化版本）
%
% 功能描述:
%   这是 run_delta_c_parameter_scan_parallel 的参数化版本，
%   允许外部调用时指定关键参数，便于批量实验。
%
% 输入参数:
%   num_runs        - 每个参数点的重复次数（例如：50）
%   N               - 粒子群体数量（例如：200）
%   noise_intensity - 角度噪声强度（例如：0.05）
%   num_workers     - 并行核心数（例如：8）
%
% 输出结果:
%   results - 包含所有实验结果的结构体，包括:
%             .delta_c, .c1_mean, .c2_mean, .cj_thresholds 等
%
% 使用示例:
%   % 单次调用
%   results = run_delta_c_scan_with_params(50, 200, 0.05, 8);
%
%   % 批量调用
%   for noise = [0.05, 0.10, 0.15]
%       run_delta_c_scan_with_params(50, 200, noise, 8);
%   end
%
% 数据存储:
%   结果自动保存到: data/experiments/delta_c_scan/N{N}_noise{noise}_{timestamp}/
%
% 作者: 基于原版 run_delta_c_parameter_scan_parallel 修改
% 日期: 2025
% 版本: MATLAB 2025a兼容

%% 参数验证
if nargin < 4
    error('需要4个输入参数: num_runs, N, noise_intensity, num_workers');
end

%% 1. 实验设置与参数范围
fprintf('=================================================\n');
fprintf('     Delta_c 参数扫描实验 (参数化版本)\n');
fprintf('=================================================\n\n');

% 构建仿真参数（使用输入参数）
params = default_simulation_parameters();
params.N = N;
params.angleNoiseIntensity = noise_intensity;

% 扫描范围（保持与原版一致）
cj_threshold_min = 0.1;
cj_threshold_max = 6.0;
cj_threshold_step = 0.1;
cj_thresholds = cj_threshold_min:cj_threshold_step:cj_threshold_max;
num_params = numel(cj_thresholds);

trials_per_param = num_runs * 2;  % c1 与 c2 各一次
completed_tasks = 0;               % 全局已完成任务计数

total_tasks = num_params * trials_per_param;

fprintf('实验配置:\n');
fprintf('  群体数量 (N): %d\n', N);
fprintf('  噪声强度 (σ): %.3f\n', noise_intensity);
fprintf('  重复次数: %d\n', num_runs);
fprintf('  并行核心: %d\n', num_workers);
fprintf('\n');
fprintf('扫描设置:\n');
fprintf('  参数范围: cj_threshold ∈ [%.1f, %.1f]\n', cj_threshold_min, cj_threshold_max);
fprintf('  步长: %.1f\n', cj_threshold_step);
fprintf('  参数点数: %d\n', num_params);
fprintf('  总实验数: %d\n', total_tasks);
fprintf('\n');

%% 2. 并行池配置
pool = configure_parallel_pool(num_workers);
fprintf('并行池已就绪: %d workers\n\n', pool.NumWorkers);

%% 3. 数据结构预分配与计时器初始化
c1_raw = NaN(num_params, num_runs);
c2_raw = NaN(num_params, num_runs);

experiment_start_time = tic;
progress_update_timer = tic;
progress_step = max(1, floor(total_tasks / 100));

% 生成输出路径（添加噪声标识）
timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
output_dir = ensure_data_directory(timestamp, N, noise_intensity);
output_filename = fullfile(output_dir, 'data.mat');
temp_filename = fullfile(output_dir, 'temp.mat');

% DataQueue 用于进度监控
progress_queue = parallel.pool.DataQueue;
afterEach(progress_queue, @(increment) update_progress(increment));

%% 4. 使用 parfor 执行参数扫描
fprintf('开始并行实验...\n');
fprintf('----------------------------------------\n');

shared_params = parallel.pool.Constant(params);

parfor param_idx = 1:num_params
    local_c1 = NaN(1, num_runs);
    local_c2 = NaN(1, num_runs);

    current_params = shared_params.Value;
    current_params.cj_threshold = cj_thresholds(param_idx);

    for run_idx = 1:num_runs
        local_c1(run_idx) = run_single_experiment(current_params, 1);
        send(progress_queue, 1);

        local_c2(run_idx) = run_single_experiment(current_params, 2);
        send(progress_queue, 1);
    end

    c1_raw(param_idx, :) = local_c1;
    c2_raw(param_idx, :) = local_c2;
end

fprintf('\n实验执行完毕，用时 %.2f 分钟。\n', toc(experiment_start_time) / 60);

total_elapsed_seconds = toc(experiment_start_time);

%% 5. 统计分析
[c1_mean, c1_std, c1_sem] = compute_statistics(c1_raw);
[c2_mean, c2_std, c2_sem] = compute_statistics(c2_raw);
delta_c = c2_mean - c1_mean;
error_count = [sum(isnan(c1_raw), 2), sum(isnan(c2_raw), 2)];

%% 6. 结果保存
results = struct();
results.description = sprintf('Delta_c scan: N=%d, noise=%.3f, runs=%d', N, noise_intensity, num_runs);
results.parameters = params;
results.scan_variable = 'cj_threshold';
results.cj_thresholds = cj_thresholds;
results.num_runs = num_runs;
results.c1_raw = c1_raw;
results.c2_raw = c2_raw;
results.c1_mean = c1_mean;
results.c2_mean = c2_mean;
results.c1_std = c1_std;
results.c2_std = c2_std;
results.c1_sem = c1_sem;
results.c2_sem = c2_sem;
results.delta_c = delta_c;
results.timestamp = timestamp;
results.date = datetime('now');
results.total_experiments = total_tasks;
results.total_time_seconds = total_elapsed_seconds;
results.total_time_hours = total_elapsed_seconds / 3600;
results.error_count = error_count;
results.parallel_workers = pool.NumWorkers;
results.matlab_version = version;

save(output_filename, 'results', '-v7.3');
fprintf('结果已保存至: %s\n', output_filename);

if exist(temp_filename, 'file')
    delete(temp_filename);
end

%% 7. 摘要输出与快速预览图
fprintf('\n=================================================\n');
fprintf('                实验完成摘要\n');
fprintf('=================================================\n');
fprintf('总耗时: %.2f 小时 (%.1f 分钟)\n', total_elapsed_seconds / 3600, total_elapsed_seconds / 60);
fprintf('并行工作线程: %d\n', pool.NumWorkers);

success_rate_c1 = 1 - sum(error_count(:, 1)) / (num_runs * num_params);
success_rate_c2 = 1 - sum(error_count(:, 2)) / (num_runs * num_params);
fprintf('成功率: %.1f%% (c1), %.1f%% (c2)\n', ...
    100 * success_rate_c1, 100 * success_rate_c2);

[max_delta_c, max_idx] = max(delta_c);
optimal_cj = cj_thresholds(max_idx);
fprintf('\nDelta_c 峰值: %.4f (在 cj_threshold = %.2f)\n', max_delta_c, optimal_cj);

quicklook_figure_path = fullfile(output_dir, 'result');
render_quicklook_figure(cj_thresholds, c1_mean, c1_sem, ...
    c2_mean, c2_sem, delta_c, optimal_cj, max_delta_c, error_count, ...
    N, num_runs, pool.NumWorkers, quicklook_figure_path);

fprintf('\n实验完成！\n');
fprintf('=================================================\n\n');

%% ==================== 嵌套辅助函数 ====================
    function params_out = default_simulation_parameters()
        params_out.N = 200;
        params_out.rho = 1;
        params_out.v0 = 1;
        params_out.angleUpdateParameter = 10;
        params_out.angleNoiseIntensity = 0.0;
        params_out.T_max = 400;
        params_out.dt = 0.1;
        params_out.radius = 5;
        params_out.deac_threshold = 0.1745;
        params_out.cj_threshold = 1;
        params_out.fieldSize = 50;
        params_out.initDirection = pi / 4;
        params_out.useFixedField = true;
        params_out.stabilization_steps = 200;
        params_out.forced_turn_duration = 200;
    end

    function [mean_values, std_values, sem_values] = compute_statistics(raw_data)
        mean_values = NaN(size(raw_data, 1), 1);
        std_values = mean_values;
        sem_values = mean_values;

        for idx = 1:size(raw_data, 1)
            valid_samples = raw_data(idx, ~isnan(raw_data(idx, :)));
            if isempty(valid_samples)
                continue;
            end
            mean_values(idx) = mean(valid_samples);
            std_values(idx) = std(valid_samples, 0);
            sem_values(idx) = std_values(idx) / sqrt(numel(valid_samples));
        end
    end

    function update_progress(increment)
        completed_tasks = completed_tasks + increment;
        if completed_tasks == 0
            return;
        end

        if toc(progress_update_timer) < 5 && ...
                mod(completed_tasks, progress_step) ~= 0
            return;
        end

        elapsed_seconds = toc(experiment_start_time);
        avg_time_per_task = elapsed_seconds / completed_tasks;
        remaining_seconds = avg_time_per_task * (total_tasks - completed_tasks);

        fprintf('  进度: %.1f%% (%d/%d) | 已用: %.1f分 | 预计剩余: %.1f分\n', ...
            100 * completed_tasks / total_tasks, completed_tasks, total_tasks, ...
            elapsed_seconds / 60, remaining_seconds / 60);

        progress_update_timer = tic;

        if mod(completed_tasks, progress_step * 10) == 0
            interim_results = struct('c1_raw', c1_raw, 'c2_raw', c2_raw, ...
                'timestamp', timestamp, 'completed_tasks', completed_tasks);
            save(temp_filename, 'interim_results', '-v7.3');
        end
    end
end

%% ==================== 局部函数定义 ====================
function result = run_single_experiment(params, exp_type)
    try
        sim = ParticleSimulationWithExternalPulse(params);
        sim.external_pulse_count = exp_type;
        cascade_size = sim.runSingleExperiment(exp_type);
        result = (cascade_size * params.N - exp_type) / params.N;
    catch ME
        warning('run_single_experiment:failure', ...
            'Experiment failed (cj=%.3f, exp_type=%d): %s', ...
            params.cj_threshold, exp_type, ME.message);
        result = NaN;
    end
end

function output_dir = ensure_data_directory(timestamp, N, noise)
    % 目录名包含 N 和 noise 信息，便于识别
    dir_name = sprintf('N%d_noise%.3f_%s', N, noise, timestamp);
    output_dir = fullfile(pwd, 'data', 'experiments', 'delta_c_scan', dir_name);
    if ~isfolder(output_dir)
        mkdir(output_dir);
    end
end

function pool = configure_parallel_pool(num_workers)
    if ~license('test', 'Distrib_Computing_Toolbox')
        error('需要 Parallel Computing Toolbox 许可证。');
    end

    pool = gcp('nocreate');

    % 如果池不存在，或者worker数不匹配，重新配置
    if isempty(pool)
        cluster = parcluster('local');
        max_workers = min(cluster.NumWorkers, num_workers);
        if max_workers < 1
            error('没有可用的并行工作线程。');
        end
        pool = parpool(cluster, max_workers);
    elseif pool.NumWorkers ~= num_workers
        % 如果已有池但worker数不匹配，给出警告但继续使用
        warning('现有并行池有 %d workers，请求的是 %d，将使用现有池。', ...
            pool.NumWorkers, num_workers);
    end
end

function render_quicklook_figure(thresholds, c1_mean, c1_sem, ...
    c2_mean, c2_sem, delta_c, optimal_cj, max_delta_c, error_count, ...
    N, num_runs, num_workers, output_prefix)

    figure_handle = figure('Name', 'Delta_c 扫描结果', ...
        'Position', [100, 100, 1200, 400]);

    % (1) 平均级联规模趋势
    subplot(1, 3, 1);
    hold on;
    errorbar(thresholds, c1_mean, c1_sem, 'b.-', 'LineWidth', 1.5, ...
        'DisplayName', 'c_1');
    errorbar(thresholds, c2_mean, c2_sem, 'r.-', 'LineWidth', 1.5, ...
        'DisplayName', 'c_2');
    xlabel('c_j threshold');
    ylabel('平均级联规模');
    title('级联规模 vs 运动显著性阈值');
    legend('Location', 'best');
    grid on;
    hold off;

    % (2) Δc 曲线及峰值标记
    subplot(1, 3, 2);
    plot(thresholds, delta_c, 'k.-', 'LineWidth', 2);
    hold on;
    plot(optimal_cj, max_delta_c, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
    xlabel('c_j threshold');
    ylabel('\Delta c = c_2 - c_1');
    title('集群敏感性 vs 运动显著性阈值');
    grid on;
    hold off;

    % (3) 失败次数统计
    subplot(1, 3, 3);
    bar(1:2, [sum(error_count(:, 1)), sum(error_count(:, 2))]);
    set(gca, 'XTickLabel', {'c1实验', 'c2实验'});
    ylabel('失败次数');
    title('实验失败统计');

    sgtitle(sprintf('Delta_c 扫描结果 (N=%d, %d次重复, %d workers)', ...
        N, num_runs, num_workers));

    savefig([output_prefix, '.fig']);
    print([output_prefix, '.png'], '-dpng', '-r300');
    close(figure_handle);
end
