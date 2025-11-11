% run_saliency_threshold_nsga2
% =========================================================================
% 目的：
%   - 在多个噪声场景 (η = 0.10 / 0.20 / 0.30) 下，利用 NSGA-II 寻找自适应
%     显著性阈值的非支配解集合，使响应性 R 与持久性 P 同时最优。
%   - 扫描逻辑与 scan_saliency_threshold 一致，但改为以优化方式自动搜索。
%   - 输出每个噪声的 Pareto 前沿、数据表和 MAT 结果，图像保存至 pic/ 目录。
% =========================================================================

clc; clear; close all;
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..')));

eta_values =  0.30;        % 需要单独优化的噪声幅度
threshold_bounds = [0.01, 0.08];        % 方差阈值搜索区间

% Monte Carlo 与优化配置 -------------------------------------------------
eval_cfg = struct();
eval_cfg.runs_per_threshold = 50;        % 单个候选阈值的重复次数（权衡准确度/耗时）
eval_cfg.base_seed = 20250405;           % 基础随机种子，保证可复现
eval_cfg.num_angles = 1;                 % 与 scan_saliency_threshold 保持一致
eval_cfg.use_parallel = true;            % 启用并行以充分利用多核资源

ga_options = optimoptions('gamultiobj', ...
    'PopulationSize', 28, ...
    'MaxGenerations', 25, ...
    'FunctionTolerance', 1e-3, ...
    'UseParallel', eval_cfg.use_parallel, ...
    'Display', 'iter');

if eval_cfg.use_parallel
    eval_cfg.parallel_pool = ensure_parallel_pool(200);
else
    eval_cfg.parallel_pool = [];
end

% 共享基础参数（与 scan_saliency_threshold 一致） ---------------------------
base_params = struct();
base_params.N = 200;
base_params.rho = 1;
base_params.v0 = 1;
base_params.angleUpdateParameter = 10;
base_params.angleNoiseIntensity = 0.02;  % 将被 η 覆盖
base_params.T_max = 400;
base_params.dt = 0.1;
base_params.radius = 5;
base_params.deac_threshold = 0.1745;
base_params.cj_threshold = 1.5;
base_params.fieldSize = 50;
base_params.initDirection = pi/4;
base_params.useFixedField = true;
base_params.stabilization_steps = 200;
base_params.forced_turn_duration = 200;
base_params.useAdaptiveThreshold = true;

adaptive_cfg = struct('cj_low', 0.5, 'cj_high', 5.0, 'include_self', false);
pers_cfg = struct('burn_in_ratio', 0.5, 'min_fit_points', 40, 'min_diffusion', 1e-4);

results_dir = fullfile('results', 'adaptive_threshold_nsga2');
if ~exist(results_dir, 'dir'); mkdir(results_dir); end
pic_dir = fullfile('pic');
if ~exist(pic_dir, 'dir'); mkdir(pic_dir); end

for eta = eta_values
    fprintf('\n====================== η = %.2f ======================\n', eta);
    params = base_params;
    params.angleNoiseIntensity = max((eta^2) / 2, 1e-4);
    eval_cfg.time_vec = (0:params.T_max)' * params.dt;

    metric_cache = containers.Map('KeyType', 'char', 'ValueType', 'any');
    obj_fun = @(x) saliency_objective(x, params, adaptive_cfg, pers_cfg, eval_cfg, metric_cache);

    [saliency_set, raw_obj] = gamultiobj(obj_fun, 1, [], [], [], [], ...
        threshold_bounds(1), threshold_bounds(2), [], ga_options);

    R_vals = -raw_obj(:, 1);
    P_vals = -raw_obj(:, 2);
    valid_mask = ~isnan(R_vals) & ~isnan(P_vals);
    R_vals = R_vals(valid_mask);
    P_vals = P_vals(valid_mask);
    saliency_set = saliency_set(valid_mask, :);

    if isempty(R_vals)
        warning('η = %.2f 未获得有效解，请检查配置。', eta);
        continue;
    end

    P_norm = P_vals / max(P_vals);
    result_table = table(saliency_set, R_vals, P_vals, P_norm, ...
        'VariableNames', {'saliency_threshold', 'R_mean', 'P_mean', 'P_norm'});
    disp(result_table);

    eta_tag = strrep(sprintf('eta_%0.3f', eta), '.', 'p');
    timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
    mat_path = fullfile(results_dir, sprintf('nsga2_saliency_%s_%s.mat', eta_tag, timestamp));
    save(mat_path, 'eta', 'saliency_set', 'R_vals', 'P_vals', 'P_norm', ...
        'threshold_bounds', 'eval_cfg', 'ga_options');

    % 绘制 Pareto 前沿 ----------------------------------------------------
    fig = figure('Color', 'white', 'Position', [180, 140, 560, 420]);
    scatter(R_vals, P_vals, 70, saliency_set, 'filled');
    colormap('turbo');
    cb = colorbar; cb.Label.String = 'saliency threshold';
    xlabel('响应性 R'); ylabel('持久性 P');
    title(sprintf('η = %.2f 的 Pareto 前沿 (NSGA-II)', eta));
    grid on; box on;
    output_pic = fullfile(pic_dir, sprintf('nsga2_saliency_front_%s.pdf', eta_tag));
    exportgraphics(fig, output_pic, 'ContentType', 'vector');
    fprintf('η = %.2f 图像已保存：%s\n', eta, output_pic);
end

%% ========================================================================
% 目标函数与仿真评估
% ========================================================================
function f = saliency_objective(thr_vec, params, adaptive_cfg, pers_cfg, eval_cfg, cache)
    sal_thr = thr_vec(1);
    key = sprintf('%.5f', sal_thr);
    if cache.isKey(key)
        metrics = cache(key);
    else
        metrics = simulate_metrics(sal_thr, params, adaptive_cfg, pers_cfg, eval_cfg);
        cache(key) = metrics;
    end

    R_mean = metrics(1);
    P_mean = metrics(2);
    if isnan(R_mean) || isnan(P_mean)
        f = [1e3, 1e3];
    else
        f = [-R_mean, -P_mean];
    end
end

function metrics = simulate_metrics(sal_thr, params, adaptive_cfg, pers_cfg, eval_cfg)
    params_local = params;
    cfg_local = adaptive_cfg;
    cfg_local.saliency_threshold = max(sal_thr, 1e-4);
    params_local.adaptiveThresholdConfig = cfg_local;

    runs = eval_cfg.runs_per_threshold;
    R_runs = NaN(runs, 1);
    P_runs = NaN(runs, 1);

    for idx = 1:runs
        seed = eval_cfg.base_seed + idx * 97 + round(sal_thr * 1e6);
        [R_val, triggered] = run_single_responsiveness(params_local, ...
            eval_cfg.num_angles, eval_cfg.time_vec, seed);
        if triggered && ~isnan(R_val)
            R_runs(idx) = R_val;
        end
        [P_val, ~] = run_single_persistence(params_local, pers_cfg, seed + 31);
        P_runs(idx) = P_val;
    end

    R_mean = mean(R_runs, 'omitnan');
    P_mean = mean(P_runs, 'omitnan');
    metrics = [R_mean, P_mean];
end

%% ========================================================================
% 以下函数与 scan_saliency_threshold 中版本一致，用于保持仿真一致性
% ========================================================================
function [R_value, triggered] = run_single_responsiveness(params, num_angles, time_vec, seed)
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

function [P_value, D_value] = run_single_persistence(params, cfg, seed)
    rng(seed);
    params_pers = params;
    if isfield(params_pers, 'stabilization_steps')
        params_pers = rmfield(params_pers, 'stabilization_steps');
    end
    if isfield(params_pers, 'forced_turn_duration')
        params_pers = rmfield(params_pers, 'forced_turn_duration');
    end
    sim = ParticleSimulation(params_pers);

    T = sim.T_max;
    dt = sim.dt;
    burn_in_index = max(2, floor((T + 1) * cfg.burn_in_ratio));

    init_pos = sim.positions;
    centroid0 = mean(init_pos, 1);
    offsets0 = init_pos - centroid0;

    msd = zeros(T + 1, 1);
    for step_idx = 1:T
        sim.step();
        positions = sim.positions;
        centroid = mean(positions, 1);
        centered = positions - centroid;
        rel_disp = centered - offsets0;
        msd(step_idx + 1) = mean(sum(rel_disp.^2, 2), 'omitnan');
    end

    time_vec_pers = (0:T)' * dt;
    x = time_vec_pers(burn_in_index:end);
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

function pool = ensure_parallel_pool(desired_workers)
    pool = gcp('nocreate');
    cluster = parcluster('local');
    max_workers = min(cluster.NumWorkers, desired_workers);
    if isempty(pool)
        pool = parpool(cluster, max_workers);
        return;
    end
    if pool.NumWorkers ~= max_workers
        delete(pool);
        pool = parpool(cluster, max_workers);
    end
end
