% tune_adaptive_threshold_evo 使用 MATLAB GA 搜索自适应阈值门限
% =========================================================================
% 目标：
%   - 调用 Global Optimization Toolbox 提供的遗传算法 (ga)，在给定仿真参数下
%     自动寻找 adaptive_cfg.order_threshold 的最佳取值，
%     使响应性指标 R 最大。
%
% 使用方式：
%   1. 确保已安装 Global Optimization Toolbox。
%   2. 按需调整“基础仿真参数”“GA 配置”“评估配置”中的参数。
%   3. 运行脚本，命令行会输出 GA 的迭代记录，并在结束时给出最佳阈值与 R。
%   4. 如需保存结果或绘制曲线，可启用脚本末尾的可选段落。
%
% 约束条件：
%   - 0.9 ≤ order_threshold ≤ 1.0
% =========================================================================

clc;
clear;
close all;

%% 1. 基础仿真参数 -------------------------------------------------------------
params = struct();
params.N = 200;
params.rho = 1;
params.v0 = 1;
params.angleUpdateParameter = 10;
params.angleNoiseIntensity = 0.05;
params.T_max = 400;
params.dt = 0.1;
params.radius = 5;
params.deac_threshold = 0.1745;
params.cj_threshold = 1.5;
params.fieldSize = 50;
params.initDirection = pi/4;
params.useFixedField = true;
params.stabilization_steps = 200;
params.forced_turn_duration = 400;
params.useAdaptiveThreshold = true;

adaptive_cfg_base = struct();
adaptive_cfg_base.cj_low = 0.5;
adaptive_cfg_base.cj_high = 5.0;
adaptive_cfg_base.include_self = true;

time_vec = (0:params.T_max)' * params.dt;

%% 2. GA 配置 ------------------------------------------------------------------
ga_cfg = struct();
ga_cfg.populationSize = 30;
ga_cfg.maxGenerations = 40;
ga_cfg.useParallel = true;     % 若具备并行环境，可设为 true
ga_cfg.initialRange = [0.9; 1.0]; % 两行分别为下界与上界

%% 3. 评估配置 -----------------------------------------------------------------
eval_cfg = struct();
eval_cfg.num_repeats = 10;        % 单个候选重复仿真次数
eval_cfg.base_seed = 20250401;   % 基础随机种子
eval_cfg.seed_stride = 7919;     % 不同重复间的种子增量

%% 4. 构建 GA 求解 -------------------------------------------------------------
% 变量：order_threshold
lb = 0.9;
ub = 1.0;
nvars = 1;

% 外层共享的历史记录，用于输出函数记录演化过程
history.generations = [];
history.best_R = [];
history.threshold = [];

fitness_fun = @(x) objective_wrapper(x, params, adaptive_cfg_base, eval_cfg, time_vec);
ga_opts = optimoptions('ga', ...
    'PopulationSize', ga_cfg.populationSize, ...
    'MaxGenerations', ga_cfg.maxGenerations, ...
    'InitialPopulationRange', ga_cfg.initialRange, ...
    'UseParallel', ga_cfg.useParallel, ...
    'Display', 'iter', ...
    'OutputFcn', @ga_output_logger, ...
    'FunctionTolerance', 1e-4, ...
    'ConstraintTolerance', 1e-6);

[x_best, fval_best, exitflag, output_ga] = ga(fitness_fun, nvars, [], [], [], [], lb, ub, [], ga_opts);

best_R = -fval_best;
best_threshold = x_best(1);

fprintf('\n搜索完成 >>>\n');
fprintf('最佳响应性 R = %.5f\n', best_R);
fprintf('最佳序参量阈值：%.5f\n', best_threshold);
fprintf('GA 退出标志: %d, 总迭代:%d\n', exitflag, output_ga.generations);

%% 5. 可选：绘制 GA 历史 -------------------------------------------------------
if ~isempty(history.generations)
    figure('Color', 'white', 'Name', 'GA 演化过程');
    subplot(2,1,1);
    plot(history.generations, history.best_R, '-o', 'LineWidth', 1.4);
    xlabel('代数');
    ylabel('最佳 R');
    title('GA 每代最佳响应性');
    grid on;

    subplot(2,1,2);
    plot(history.generations, history.threshold, '-s', 'LineWidth', 1.2);
    xlabel('代数');
    ylabel('序参量阈值');
    title('阈值演化轨迹');
    grid on;
end

%% 6. 可选：保存结果 -----------------------------------------------------------
% results_dir = fullfile('results', 'tuning');
% if ~exist(results_dir, 'dir')
%     mkdir(results_dir);
% end
% timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
% save(fullfile(results_dir, sprintf('adaptive_threshold_ga_%s.mat', timestamp)), ...
%     'x_best', 'best_R', 'history', 'params', 'adaptive_cfg_base', 'ga_cfg', 'eval_cfg', 'output_ga');

%% ============================================================================
% 辅助函数区域
% ============================================================================
function f = objective_wrapper(x, params, adaptive_cfg_base, eval_cfg, time_vec)
    % objective_wrapper GA 目标函数（求最小化）
    % 返回 -R_mean，以便 GA 最大化 R
    if numel(x) ~= 1
        f = 1e3;
        return;
    end

    thr = x(1);
    if ~(thr >= 0.9 && thr <= 1.0)
        f = 1e3;
        return;
    end

    cfg = adaptive_cfg_base;
    cfg.order_threshold = thr;

    params_local = params;
    params_local.adaptiveThresholdConfig = cfg;

    num_angles = 1;
    R_values = NaN(eval_cfg.num_repeats, 1);

    for k = 1:eval_cfg.num_repeats
        seed = eval_cfg.base_seed + (k-1) * eval_cfg.seed_stride;
        [R_value, triggered] = run_single_responsiveness(params_local, num_angles, time_vec, seed);
        if triggered && ~isnan(R_value)
            R_values(k) = R_value;
        end
    end

    valid = ~isnan(R_values);
    if any(valid)
        R_mean = mean(R_values(valid));
        f = -R_mean;   % GA 最小化 -> 取负
    else
        f = 1e3;       % 若无有效结果，返回较大惩罚
    end
end

function [state, options, optchanged] = ga_output_logger(options, state, flag)
    % ga_output_logger 记录每代最佳个体及其 R
    persistent history_ref
    optchanged = false;

    if strcmp(flag, 'init')
        history_ref = evalin('base', 'history');
    end

    switch flag
        case {'iter', 'init'}
            scores = state.Score;
            pop = state.Population;
            [best_score, idx] = min(scores);
            best_x = pop(idx, :);
            best_R = -best_score;

            history_ref.generations(end+1,1) = state.Generation;
            history_ref.best_R(end+1,1) = best_R;
            history_ref.threshold(end+1,1) = best_x(1);

            assignin('base', 'history', history_ref);
        case 'done'
            history_ref = evalin('base', 'history');
            assignin('base', 'history', history_ref);
    end
end

function [R_value, triggered] = run_single_responsiveness(params, num_angles, time_vec, seed)
    % run_single_responsiveness 复制实验脚本中的单次响应性评估逻辑
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

function V = compute_average_velocity(theta, v0)
    % compute_average_velocity 计算平均速度向量
    V = [mean(v0 * cos(theta)), mean(v0 * sin(theta))];
end
