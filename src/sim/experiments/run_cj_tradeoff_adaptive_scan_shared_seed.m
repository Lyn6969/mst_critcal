% run_cj_tradeoff_adaptive_scan_shared_seed
% =========================================================================
% 目的：
%   - 对比“固定阈值扫描”与“自适应显著性阈值”在共享随机环境下的 R-P 权衡。
%   - 通过为所有阈值复用同一组随机种子，使差异主要来源于阈值机制本身。
%
% 输出：
%   - 控制台进度提示
%   - results/tradeoff/ 目录下的 MAT 数据与 R-P 图像
%
% 使用方式：
%   直接运行脚本。
% =========================================================================

clc;
clear;
close all;

addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..')));

fprintf('=================================================\n');
fprintf('   固定阈值 vs 自适应阈值（共享随机环境）\n');
fprintf('=================================================\n\n');

% 基础参数设置（响应性和持久性实验共享）
base_common = struct();
base_common.N = 200;                            % 个体数量
base_common.rho = 1;
base_common.v0 = 1;
base_common.angleUpdateParameter = 10;
base_common.angleNoiseIntensity = 0.05;
base_common.T_max = 400;                        % 最大时间步
base_common.dt = 0.1;
base_common.radius = 5;                         % 感知半径
base_common.deac_threshold = 0.1745;            % 失活阈值（10度）
base_common.cj_threshold = 1.5;                 % 默认cj阈值（会被扫描值覆盖）
base_common.fieldSize = 50;
base_common.initDirection = pi/4;
base_common.useFixedField = true;

% 响应性实验参数
resp_params = base_common;
resp_params.stabilization_steps = 200;          % 稳定化步数
resp_params.forced_turn_duration = 200;         % 强制转向持续时长

% 持久性实验参数
pers_params = base_common;
pers_params.T_max = 400;

% 实验设置
cj_thresholds_fixed = 0.0:0.1:5.0;              % 固定阈值扫描范围（51个点）
num_runs = 200;                                 % 每个阈值重复次数
num_angles = 1;                                 % 投影角度数量

% 持久性计算配置
pers_cfg = struct();
pers_cfg.burn_in_ratio = 0.5;                  % 去除前25%数据
pers_cfg.min_diffusion = 1e-3;                  % 最小扩散系数下限
pers_cfg.min_fit_points = 40;                   % 拟合最小数据点数

% 自适应阈值配置
adaptive_cfg = struct();
adaptive_cfg.cj_low = 0.5;                      % cj下限
adaptive_cfg.cj_high = 5.0;                     % cj上限
adaptive_cfg.saliency_threshold = 0.031;        % 显著性方差阈值
adaptive_cfg.include_self = false;              % 不包含自身

% 定义两种实验模式
modes = {
    struct('id', 'fixed', 'label', '固定阈值扫描', 'useAdaptive', false, ...
           'cfg', [], 'cj_thresholds', cj_thresholds_fixed), ...
    struct('id', 'adaptive', 'label', '自适应阈值', 'useAdaptive', true, ...
           'cfg', adaptive_cfg, 'cj_thresholds', 1) % 占位，实际不使用
};

% 时间向量和随机种子设置
time_vec_resp = (0:resp_params.T_max)' * resp_params.dt;
base_seed = 20250314;
shared_seeds = base_seed + (0:num_runs-1);      % 所有阈值共享同一组种子

% 配置并行计算池
parallel_cfg = struct();
parallel_cfg.desired_workers = [];              % 自动检测核心数
pool = configure_parallel_pool(parallel_cfg.desired_workers);
fprintf('并行模式已启用：%d 个 workers\n\n', pool.NumWorkers);

results = struct();

%% 主循环：依次运行两种模式
for mode_idx = 1:numel(modes)
    mode = modes{mode_idx};
    fprintf('[%s] 模式：%s\n', mode.id, mode.label);

    % 根据模式配置参数
    if mode.useAdaptive
        resp_params.useAdaptiveThreshold = true;
        resp_params.adaptiveThresholdConfig = mode.cfg;
        pers_params.useAdaptiveThreshold = true;
        pers_params.adaptiveThresholdConfig = mode.cfg;
        cj_list = 1;                            % 占位，自适应模式不扫描cj
    else
        resp_params.useAdaptiveThreshold = false;
        pers_params.useAdaptiveThreshold = false;
        cj_list = cj_thresholds_fixed;          % 固定模式扫描cj范围
    end

    % 运行当前模式的实验
    mode_results = run_tradeoff_mode_shared(resp_params, pers_params, cj_list, mode.useAdaptive, ...
        adaptive_cfg, num_runs, num_angles, pers_cfg, time_vec_resp, shared_seeds);

    results.(mode.id) = mode_results;
    fprintf('  完成：平均 R = %.3f, 平均 P = %.3f\n\n', ...
        mean(mode_results.R_mean, 'omitnan'), mean(mode_results.P_mean, 'omitnan'));
end

%% 归一化持久性（使用固定 + 自适应所有样本）
% 收集所有模式的持久性数据
all_P = [];
mode_ids = fieldnames(results);
for k = 1:numel(mode_ids)
    all_P = [all_P; results.(mode_ids{k}).P_raw(:)]; %#ok<AGROW>
end

% 计算归一化范围（基于所有模式的均值）
P_means_all = [];
for k = 1:numel(mode_ids)
    P_means_all = [P_means_all; results.(mode_ids{k}).P_mean(:)]; %#ok<AGROW>
end
P_means_all = P_means_all(~isnan(P_means_all));
if isempty(P_means_all)
    P_min = 0;
    P_max = 1;
else
    P_min = min(P_means_all);
    P_max = max(P_means_all);
end
P_range = max(P_max - P_min, eps);

% 对每个模式进行min-max归一化
for k = 1:numel(mode_ids)
    key = mode_ids{k};
    res = results.(key);
    res.P_mean_norm = (res.P_mean - P_min) / P_range;
    res.P_std_norm = res.P_std / P_range;
    res.P_sem_norm = res.P_sem / P_range;
    results.(key) = res;
end
fprintf('持久性归一化：min = %.4f, max = %.4f\n\n', P_min, P_max);

%% 绘制 R-P 权衡图
results_dir = fullfile('results', 'tradeoff');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end
timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
output_mat = fullfile(results_dir, sprintf('cj_tradeoff_adaptive_shared_seed_%s.mat', timestamp));
output_fig = fullfile(results_dir, sprintf('cj_tradeoff_adaptive_shared_seed_%s.png', timestamp));

figure('Name', '固定 vs 自适应阈值（共享随机环境）', 'Color', 'white', 'Position', [160, 120, 960, 600]);
hold on;

% 绘制固定阈值扫描结果（散点图 + 连线）
fixed_data = results.fixed;
scatter(fixed_data.R_mean, fixed_data.P_mean_norm, 70, cj_thresholds_fixed, 'filled', 'DisplayName', '固定阈值');
colormap('turbo');
cb = colorbar;
cb.Label.String = '固定阈值';
plot(fixed_data.R_mean, fixed_data.P_mean_norm, '-', 'Color', [0.3 0.3 0.3], 'LineWidth', 1.2);

% 绘制自适应阈值结果（单点，五角星标记）
adaptive_data = results.adaptive;
scatter(adaptive_data.R_mean, adaptive_data.P_mean_norm, 160, [0.85 0.2 0.2], 'filled', ...
    'DisplayName', sprintf('自适应阈值 %.3f', adaptive_cfg.saliency_threshold), 'Marker', 'p', ...
    'MarkerEdgeColor', [0.5 0 0], 'MarkerFaceColor', [0.85 0.2 0.2]);

xlabel('响应性 R');
ylabel('归一化持久性 P (min-max)');
title('固定阈值扫描 vs 自适应显著性阈值（共享随机环境）');
legend('Location', 'best');
grid on;

saveas(gcf, output_fig);
fprintf('图像已保存至: %s\n', output_fig);

summary = struct();
summary.description = 'Fixed vs adaptive threshold trade-off (shared seeds)';
summary.timestamp = timestamp;
summary.base_params_resp = resp_params;
summary.base_params_pers = pers_params;
summary.adaptive_config = adaptive_cfg;
summary.cj_thresholds_fixed = cj_thresholds_fixed;
summary.num_runs = num_runs;
summary.results = results;
summary.shared_seeds = shared_seeds;
summary.P_min = P_min;
summary.P_max = P_max;
summary.P_range = P_range;
summary.matlab_version = version;
summary.base_seed = base_seed;

%% 运行结束后：计算并记录响应性增益（自适应 vs 最近邻固定阈值）
% 计算方法：以归一化持久性 P_mean_norm 为度量，选取与自适应点最接近的固定阈值点，
%           计算 R 的比值增益 = R_adapt / R_fixed_nearest，并写入 summary.metrics，同时在控制台输出。
summary.metrics = struct();
try
    fixed = results.fixed;
    adaptive = results.adaptive;
    % 自适应点（目前仅一个参数点）
    R_adapt = adaptive.R_mean(1);
    P_adapt = adaptive.P_mean_norm(1);
    % 最近邻（按归一化持久性匹配）
    [~, idx_ref] = min(abs(fixed.P_mean_norm - P_adapt));
    R_ref = fixed.R_mean(idx_ref);
    P_ref = fixed.P_mean_norm(idx_ref);
    if ~isnan(R_adapt) && ~isnan(R_ref) && R_ref > 0
        resp_gain = R_adapt / R_ref;
        summary.metrics.R_adapt = R_adapt;
        summary.metrics.R_ref = R_ref;
        summary.metrics.P_adapt = P_adapt;
        summary.metrics.P_ref = P_ref;
        summary.metrics.resp_gain = resp_gain;
        fprintf('响应性增益（自适应 vs 最近邻固定）：%.2f%%  (R_adapt=%.3f, R_fixed=%.3f, P_match=%.3f)\n', ...
            (resp_gain - 1) * 100, R_adapt, R_ref, P_ref);
    else
        summary.metrics.R_adapt = R_adapt;
        summary.metrics.R_ref = R_ref;
        summary.metrics.P_adapt = P_adapt;
        summary.metrics.P_ref = P_ref;
        summary.metrics.resp_gain = NaN;
        fprintf('响应性增益无法计算：R_adapt 或 R_fixed 无效。\n');
    end
catch ME
    warning('计算响应性增益失败：%s', ME.message);
    summary.metrics.R_adapt = NaN;
    summary.metrics.R_ref = NaN;
    summary.metrics.P_adapt = NaN;
    summary.metrics.P_ref = NaN;
    summary.metrics.resp_gain = NaN;
end

save(output_mat, 'summary', '-v7.3');
fprintf('数据已保存至: %s\n', output_mat);

%% ===============================================================================
% 子函数：运行单个模式的完整实验（使用共享随机种子）
% ===============================================================================
function mode_results = run_tradeoff_mode_shared(resp_params, pers_params, cj_thresholds, ...
    useAdaptive, adaptive_cfg, num_runs, num_angles, pers_cfg, time_vec_resp, shared_seeds)

    % 配置自适应/固定阈值模式
    if useAdaptive
        resp_params.useAdaptiveThreshold = true;
        resp_params.adaptiveThresholdConfig = adaptive_cfg;
        pers_params.useAdaptiveThreshold = true;
        pers_params.adaptiveThresholdConfig = adaptive_cfg;
    else
        resp_params.useAdaptiveThreshold = false;
        pers_params.useAdaptiveThreshold = false;
    end

    % 预分配结果矩阵
    num_params = numel(cj_thresholds);
    R_raw = NaN(num_params, num_runs);          % 响应性原始数据
    P_raw = NaN(num_params, num_runs);          % 持久性原始数据
    D_raw = NaN(num_params, num_runs);          % 扩散系数原始数据
    trigger_failures = zeros(num_params, 1);    % 触发失败计数

    % 进度跟踪设置
    progress.total_tasks = num_params * num_runs;
    progress.completed = 0;
    progress.start_time = tic;
    progress.last_report = tic;
    progress.cj_thresholds = cj_thresholds;

    dq = parallel.pool.DataQueue;
    afterEach(dq, @(info) report_progress(info));

    % 自适应模式：单参数，并行化重复次数
    if num_params == 1
        current_cj = cj_thresholds(1);
        R_vec = NaN(num_runs, 1);
        P_vec = NaN(num_runs, 1);
        D_vec = NaN(num_runs, 1);
        fail_vec = false(num_runs, 1);

        parfor run_idx = 1:num_runs
            seed_base = shared_seeds(run_idx);      % 使用共享种子
            resp_local = resp_params;
            pers_local = pers_params;
            resp_local.cj_threshold = current_cj;
            pers_local.cj_threshold = current_cj;

            % 运行响应性和持久性实验
            [R_val, triggered] = run_single_responsiveness_trial(resp_local, num_angles, time_vec_resp, seed_base);
            [P_val, D_val] = run_single_persistence_trial(pers_local, pers_cfg, seed_base + 10000);

            R_vec(run_idx) = R_val;
            P_vec(run_idx) = P_val;
            D_vec(run_idx) = D_val;
            fail_vec(run_idx) = (~triggered || isnan(R_val));

            send(dq, struct('param_idx', 1, 'run_idx', run_idx));
        end

        R_raw(1, :) = R_vec.';
        P_raw(1, :) = P_vec.';
        D_raw(1, :) = D_vec.';
        trigger_failures(1) = sum(fail_vec);
    % 固定阈值模式：多参数扫描，并行化参数点
    else
        parfor param_idx = 1:num_params
            current_cj = cj_thresholds(param_idx);
            resp_local = resp_params;
            pers_local = pers_params;
            resp_local.cj_threshold = current_cj;
            pers_local.cj_threshold = current_cj;

            local_R = NaN(1, num_runs);
            local_P = NaN(1, num_runs);
            local_D = NaN(1, num_runs);
            local_fail = false(1, num_runs);

            % 内层循环：对每个参数点运行多次实验（串行）
            for run_idx = 1:num_runs
                seed_base = shared_seeds(run_idx);  % 使用共享种子

                [R_val, triggered] = run_single_responsiveness_trial(resp_local, num_angles, time_vec_resp, seed_base);
                [P_val, D_val] = run_single_persistence_trial(pers_local, pers_cfg, seed_base + 10000);

                local_R(run_idx) = R_val;
                local_P(run_idx) = P_val;
                local_D(run_idx) = D_val;
                local_fail(run_idx) = (~triggered || isnan(R_val));

                send(dq, struct('param_idx', param_idx, 'run_idx', run_idx));
            end

            R_raw(param_idx, :) = local_R;
            P_raw(param_idx, :) = local_P;
            D_raw(param_idx, :) = local_D;
            trigger_failures(param_idx) = sum(local_fail);
        end
    end

    % 汇总统计结果
    mode_results = struct();
    mode_results.R_raw = R_raw;
    mode_results.P_raw = P_raw;
    mode_results.D_raw = D_raw;
    mode_results.trigger_failures = trigger_failures;

    % 计算响应性统计量
    mode_results.R_mean = mean(R_raw, 2, 'omitnan');
    mode_results.R_std = std(R_raw, 0, 2, 'omitnan');
    mode_results.R_sem = mode_results.R_std ./ sqrt(num_runs);

    % 计算持久性统计量
    mode_results.P_mean = mean(P_raw, 2, 'omitnan');
    mode_results.P_std = std(P_raw, 0, 2, 'omitnan');
    mode_results.P_sem = mode_results.P_std ./ sqrt(num_runs);

    % 计算扩散系数统计量
    mode_results.D_mean = mean(D_raw, 2, 'omitnan');
    mode_results.cj_thresholds = cj_thresholds(:);
    if useAdaptive
        mode_results.saliency_threshold = adaptive_cfg.saliency_threshold;
    end

    function report_progress(info)
        progress.completed = progress.completed + 1;
        if toc(progress.last_report) < 5 && progress.completed < progress.total_tasks
            return;
        end
        elapsed = toc(progress.start_time);
        avg_time = elapsed / progress.completed;
        remaining = avg_time * max(progress.total_tasks - progress.completed, 0);
        current_thr = progress.cj_thresholds(min(info.param_idx, numel(progress.cj_thresholds)));
        fprintf('    [%.1f%%] (%d/%d runs) 当前阈值 %.3f | 已用 %.1f 分钟 | 预计剩余 %.1f 分钟\n', ...
            100 * progress.completed / progress.total_tasks, progress.completed, progress.total_tasks, ...
            current_thr, elapsed/60, remaining/60);
        progress.last_report = tic;
    end
end

% ===============================================================================
% 辅助函数：配置并行计算池
% ===============================================================================
function pool = configure_parallel_pool(desired_workers)
    pool = gcp('nocreate');
    if isempty(pool)
        if isempty(desired_workers)
            pool = parpool;
        else
            pool = parpool(desired_workers);
        end
    elseif ~isempty(desired_workers) && pool.NumWorkers ~= desired_workers
        delete(pool);
        pool = parpool(desired_workers);
    end
end

% ===============================================================================
% 辅助函数：运行单次响应性实验
% 输入：params - 仿真参数, num_angles - 投影角度数, time_vec - 时间向量, seed - 随机种子
% 输出：R_value - 响应性值, triggered - 是否成功触发
% ===============================================================================
function [R_value, triggered] = run_single_responsiveness_trial(params, num_angles, time_vec, seed)
    rng(seed);
    sim = ParticleSimulationWithExternalPulse(params);
    sim.external_pulse_count = 1;
    sim.resetCascadeTracking();
    sim.initializeParticles();

    % 记录平均速度历史
    V_history = zeros(params.T_max + 1, 2);
    V_history(1, :) = compute_average_velocity(sim.theta, params.v0);

    projection_history = zeros(params.T_max + 1, num_angles);
    triggered = false;
    n_vectors = [];
    t_start = NaN;

    % 仿真主循环
    for t = 1:params.T_max
        sim.step();
        V_history(t + 1, :) = compute_average_velocity(sim.theta, params.v0);

        % 检测外部脉冲触发
        if ~triggered && sim.external_pulse_triggered
            triggered = true;
            t_start = t;
            leader_idx = find(sim.isExternallyActivated, 1, 'first');
            if isempty(leader_idx)
                leader_idx = 1;
            end
            target_theta = sim.external_target_theta(leader_idx);
            % 构建投影方向向量
            if num_angles <= 1
                phi_list = target_theta;
            else
                phi_offsets = linspace(0, pi, num_angles);
                phi_list = target_theta + phi_offsets;
            end
            n_vectors = [cos(phi_list); sin(phi_list)];
        end

        % 计算投影
        if triggered
            projection_history(t + 1, :) = V_history(t + 1, :) * n_vectors;
        end
    end

    % 检查是否成功触发
    if ~triggered || isnan(t_start)
        R_value = NaN;
        return;
    end

    % 计算响应性指标（在强制转向窗口内的速度投影积分）
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

% ===============================================================================
% 辅助函数：运行单次持久性实验
% 输入：params - 仿真参数, cfg - 持久性计算配置, seed - 随机种子
% 输出：P_value - 持久性值, D_value - 扩散系数
% ===============================================================================
function [P_value, D_value] = run_single_persistence_trial(params, cfg, seed)
    rng(seed);
    sim = ParticleSimulation(params);

    T = sim.T_max;
    dt = sim.dt;
    burn_in_index = max(2, floor((T + 1) * cfg.burn_in_ratio));

    % 记录初始位置（相对于质心）
    init_pos = sim.positions;
    centroid0 = mean(init_pos, 1);
    offsets0 = init_pos - centroid0;

    % 计算均方位移（MSD）
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

    % 去除burn-in阶段数据
    time_vec = (0:T)' * dt;
    x = time_vec(burn_in_index:end);
    y = msd(burn_in_index:end);

    % 线性拟合MSD ~ time，斜率即为扩散系数
    if numel(x) < max(2, cfg.min_fit_points) || all(abs(y - y(1)) < eps)
        D_value = NaN;
    else
        x_shift = x - x(1);
        y_shift = y - y(1);
        if any(x_shift > 0) && any(abs(y_shift) > eps)
            % 平滑处理
            smooth_window = max(5, floor(numel(y_shift) * 0.1));
            if smooth_window > 1
                y_shift = smoothdata(y_shift, 'movmean', smooth_window);
            end
            % 非负最小二乘拟合
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

    % 计算持久性：P = 1/sqrt(D)
    if isnan(D_value)
        P_value = NaN;
    else
        D_value = max(D_value, cfg.min_diffusion);
        P_value = 1 / sqrt(D_value);
    end
end

% ===============================================================================
% 辅助函数：计算平均速度向量
% ===============================================================================
function V = compute_average_velocity(theta, v0)
    V = [mean(v0 * cos(theta)), mean(v0 * sin(theta))];
end
