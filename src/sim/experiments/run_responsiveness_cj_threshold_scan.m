% run_responsiveness_cj_threshold_scan
% =========================================================================
% 脚本功能：
%   针对高阶拓扑外源脉冲场景，扫描“运动显著性阈值 cj_threshold”，
%   统计集群响应性指标 R 与其关系。每个参数点重复 50 次，取随机种子
%   1:50，默认只沿领导者方向计算 r(φ) 并取均值。
%
% 参考：visualize_cluster_responsiveness.m 的响应性定义与计算流程。
%
% 输出：
%   - 控制台实时进度与统计信息
%   - results/responsiveness/ 下保存实验数据 (MAT 文件)
%   - 自动绘制 R-阈值 关系图（含标准误差）
%
% 注意：
%   - 若某次实验外源脉冲未成功触发，将记录为 NaN 并统计失败次数
%   - 可根据研究需求调整 num_angles、num_runs 等参数
% =========================================================================

% --- 环境初始化 ---
clc;            % 清空命令行窗口
clear;          % 清除工作区所有变量
close all;      % 关闭所有图形窗口

%% 1. 实验总体配置 ---------------------------------------------------------------
fprintf('=================================================\n');
fprintf('   运动显著性阈值参数扫描：响应性 R\n');
fprintf('=================================================\n\n');

% --- 基础仿真参数定义 ---
% 定义一个结构体 `base_params`，用于存储所有仿真共享的基础参数。
% 这样做便于管理和传递参数。
base_params = struct();
base_params.N = 200;                        % 粒子数量
base_params.rho = 1;                        % 粒子间相互作用的拓扑范围参数
base_params.v0 = 1;                         % 粒子速度大小
base_params.angleUpdateParameter = 10;      % 角度更新参数 (eta)
base_params.angleNoiseIntensity = 0;        % 角度更新噪声强度
base_params.T_max = 400;                    % 单次仿真总步数
base_params.dt = 0.1;                       % 仿真时间步长
base_params.radius = 5;                     % 粒子间的物理作用半径（用于邻居搜索）
base_params.deac_threshold = 0.1745;        % 失活阈值 (0.05*pi)，用于级联动力学
base_params.cj_threshold = 1;               % 运动显著性阈值，此值将在扫描中被覆盖
base_params.fieldSize = 50;                 % 仿真区域边长
base_params.initDirection = pi/4;           % 粒子初始运动方向
base_params.useFixedField = true;           % 是否使用固定的、周期性边界的仿真区域
base_params.stabilization_steps = 200;      % 外源脉冲施加前的系统稳定化步数
base_params.forced_turn_duration = 200;     % 外源脉冲强制转向的持续时间

% --- 参数扫描设置 ---
cj_thresholds = 0.0:0.1:5.0;                % 定义要扫描的 cj_threshold 参数范围
num_params = numel(cj_thresholds);          % 扫描的参数点总数
num_runs = 50;                              % 每个参数点的重复实验次数，用于统计平均
num_angles = 1;                             % 计算响应性时，投影方向的样本数。1 表示只沿领导者方向。

% --- 在控制台输出实验配置信息 ---
fprintf('扫描范围 cj_threshold ∈ [%.1f, %.1f]，步长 %.1f，共 %d 个参数点。\n', ...
    cj_thresholds(1), cj_thresholds(end), cj_thresholds(2) - cj_thresholds(1), num_params);
fprintf('每个参数重复次数: %d，方向样本: %d\n\n', num_runs, num_angles);

%% 2. 数据预分配 -----------------------------------------------------------------
% 为了提高性能，预先分配存储结果的矩阵内存。

time_vec = (0:base_params.T_max)' * base_params.dt; % 创建时间向量，用于后续积分计算
R_raw = NaN(num_params, num_runs);          % 预分配矩阵，用于存储每次实验原始的响应性 R 值
trigger_failures = zeros(num_params, 1);    % 预分配向量，用于记录每个参数点下脉冲触发失败的次数

%% 3. 运行参数扫描 ----------------------------------------------------------------
experiment_tic = tic; % 记录整个实验开始的时间
total_experiments = num_params * num_runs; % 计算总实验次数

% --- 外层循环：遍历所有参数点 ---
for param_idx = 1:num_params
    current_threshold = cj_thresholds(param_idx); % 获取当前参数值
    params = base_params; % 复制基础参数
    params.cj_threshold = current_threshold; % 设置当前扫描的 cj_threshold

    fprintf('参数 %d/%d：cj_threshold = %.2f\n', param_idx, num_params, current_threshold);
    param_tic = tic; % 记录当前参数点开始的时间

    % --- 内层循环：对每个参数点重复运行多次 ---
    for run_idx = 1:num_runs
        % 设置随机种子，确保实验的可复现性
        seed = (param_idx - 1) * num_runs + run_idx;
        
        % 调用核心函数，运行单次仿真并计算响应性
        [R_value, triggered] = run_single_responsiveness_trial(params, num_angles, time_vec, seed);
        
        % 存储结果
        R_raw(param_idx, run_idx) = R_value;
        
        % 如果脉冲未触发或结果为 NaN，则增加失败计数
        if ~triggered || isnan(R_value)
            trigger_failures(param_idx) = trigger_failures(param_idx) + 1;
        end

        % 每 10 次运行或最后一次运行时，在控制台输出进度
        if mod(run_idx, 10) == 0 || run_idx == num_runs
            fprintf('  运行 %2d/%2d 完成 (R=%.3f)\n', run_idx, num_runs, R_value);
        end
    end

    elapsed = toc(param_tic); % 计算当前参数点所有运行的总耗时
    fprintf('  >>> 参数点耗时 %.1f s，触发失败 %d 次\n', elapsed, trigger_failures(param_idx));
end

total_elapsed = toc(experiment_tic); % 计算整个实验的总耗时
fprintf('\n全部实验完成，总耗时 %.1f 分钟\n\n', total_elapsed / 60);

%% 4. 统计分析 --------------------------------------------------------------------
% 对每个参数点的多次实验结果进行统计，计算均值、标准差和标准误差。

% --- 预分配统计结果向量 ---
R_mean = NaN(num_params, 1); % 存储每个参数点的响应性均值
R_std = NaN(num_params, 1);  % 存储每个参数点的响应性标准差
R_sem = NaN(num_params, 1);  % 存储每个参数点的响应性标准误差 (Standard Error of the Mean)

for param_idx = 1:num_params
    % 获取当前参数点所有运行的原始 R 值
    valid_values = R_raw(param_idx, :);
    % 剔除由触发失败导致的 NaN 值
    valid_values = valid_values(~isnan(valid_values));
    
    % 如果没有有效值（所有运行都失败），则跳过
    if isempty(valid_values)
        continue;
    end
    
    % 计算均值、标准差和标准误差
    R_mean(param_idx) = mean(valid_values);
    R_std(param_idx) = std(valid_values);
    R_sem(param_idx) = R_std(param_idx) / sqrt(numel(valid_values));
end

%% 5. 结果可视化 ------------------------------------------------------------------
% 创建图形窗口并设置属性
figure('Name', '响应性 vs 运动显著性阈值', 'Color', 'white', 'Position', [120, 120, 900, 540]);

% 使用 errorbar 绘制均值和标准误差
errorbar(cj_thresholds, R_mean, R_sem, 'o-', 'LineWidth', 1.6, ...
    'MarkerSize', 5, 'MarkerFaceColor', [0.2 0.5 0.8]);

grid on; % 显示网格
xlabel('运动显著性阈值 c_j'); % X轴标签
ylabel('响应性指标 R'); % Y轴标签
title(sprintf('R 随 c_j 变化 (每点重复 %d 次)', num_runs)); % 图形标题

%% 6. 保存数据 --------------------------------------------------------------------
% 将实验配置和所有结果保存到 MAT 文件中，以便后续分析。

% 创建保存结果的目录
results_dir = fullfile('results', 'responsiveness');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

% 生成带时间戳的文件名
timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
output_path = fullfile(results_dir, sprintf('responsiveness_cj_scan_%s.mat', timestamp));

% --- 将所有相关数据存入一个结构体 ---
results = struct();
results.description = 'Responsiveness vs cj_threshold scan';
results.base_parameters = base_params;      % 基础参数
results.cj_thresholds = cj_thresholds;      % 扫描的阈值
results.num_runs = num_runs;                % 重复次数
results.num_angles = num_angles;            % 投影方向数
results.time_vec = time_vec;                % 时间向量
results.R_raw = R_raw;                      % 原始 R 值
results.R_mean = R_mean;                    % R 均值
results.R_std = R_std;                      % R 标准差
results.R_sem = R_sem;                      % R 标准误差
results.trigger_failures = trigger_failures;% 失败次数
results.total_experiments = total_experiments; % 总实验次数
results.total_time_seconds = total_elapsed; % 总耗时
results.timestamp = timestamp;              % 时间戳
results.matlab_version = version;           % MATLAB 版本

% 保存结果结构体到文件
save(output_path, 'results', '-v7.3');
fprintf('实验结果已保存至: %s\n', output_path);

%% ============================================================================
%                        辅助函数定义
% ============================================================================

function [R_value, triggered] = run_single_responsiveness_trial(params, num_angles, time_vec, seed)
    % run_single_responsiveness_trial
    %   功能：针对给定的参数运行一次仿真，计算并返回响应性 R 值，
    %         以及一个布尔值，指示外源脉冲是否成功触发。
    %   输入：
    %     - params: 包含所有仿真参数的结构体
    %     - num_angles: 计算响应性时使用的投影方向数量
    %     - time_vec: 时间向量
    %     - seed: 随机数生成器种子
    %   输出：
    %     - R_value: 计算得到的响应性指标 R
    %     - triggered: 一个布尔值，true 表示脉冲成功触发

    % --- 初始化 ---
    rng(seed); % 设置随机种子以保证结果可复现
    sim = ParticleSimulationWithExternalPulse(params); % 创建仿真对象
    sim.external_pulse_count = 1; % 设置外源脉冲数量
    sim.resetCascadeTracking(); % 重置级联跟踪
    sim.initializeParticles(); % 初始化粒子状态

    % --- 预分配历史数据存储 ---
    V_history = zeros(params.T_max + 1, 2); % 存储每一步的平均速度向量
    V_history(1, :) = compute_average_velocity(sim.theta, params.v0); % 计算初始平均速度

    projection_history = zeros(params.T_max + 1, num_angles); % 存储平均速度在目标方向上的投影
    triggered = false; % 标志位，记录脉冲是否已触发
    n_vectors = []; % 存储投影方向的单位向量
    t_start = NaN; % 记录脉冲触发的精确时间步

    % --- 仿真主循环 ---
    for t = 1:params.T_max
        sim.step(); % 执行一步仿真
        V_history(t + 1, :) = compute_average_velocity(sim.theta, params.v0); % 计算并记录当前平均速度

        % --- 检查外源脉冲是否在本步触发 ---
        if ~triggered && sim.external_pulse_triggered
            triggered = true; % 设置触发标志
            t_start = t; % 记录触发时间步
            
            % 确定领导者（第一个被外部激活的粒子）
            leader_idx = find(sim.isExternallyActivated, 1, 'first');
            if isempty(leader_idx)
                leader_idx = 1; % 如果找不到，则默认第一个为领导者
            end
            
            % 获取领导者的目标转向角度
            target_theta = sim.external_target_theta(leader_idx);
            
            % 根据 num_angles 确定投影方向
            if num_angles <= 1
                phi_list = target_theta; % 只沿领导者目标方向投影
            else
                % 在 [0, pi] 范围内均匀采样多个方向进行投影
                phi_offsets = linspace(0, pi, num_angles);
                phi_list = target_theta + phi_offsets;
            end
            % 计算并存储投影方向的单位向量
            n_vectors = [cos(phi_list); sin(phi_list)];
        end

        % 如果脉冲已触发，则计算并记录速度投影
        if triggered
            projection_history(t + 1, :) = V_history(t + 1, :) * n_vectors;
        end
    end

    % --- 计算响应性 R ---
    % 如果脉冲从未触发，则返回 NaN
    if ~triggered || isnan(t_start)
        R_value = NaN;
        return;
    end

    v0 = params.v0; % 粒子速度
    T_window = params.forced_turn_duration; % 响应性计算的时间窗口长度
    t_end = min(t_start + T_window, params.T_max); % 确定积分结束时间步

    r_history = NaN(num_angles, 1); % 存储每个投影方向的响应性 r(φ)
    for angle_idx = 1:num_angles
        proj = projection_history(:, angle_idx); % 获取单个方向上的投影历史
        
        % 使用梯形法则进行数值积分
        integral_value = trapz(time_vec(t_start+1:t_end+1), proj(t_start+1:t_end+1));
        
        % 计算积分区间的总时长
        duration = time_vec(t_end+1) - time_vec(t_start+1);
        
        % 根据公式计算 r(φ)
        if duration > 0
            r_history(angle_idx) = integral_value / (v0 * duration);
        end
    end

    % 最终的响应性 R 是所有有效 r(φ) 的均值
    R_value = mean(r_history(~isnan(r_history)));
end

function V = compute_average_velocity(theta, v0)
    % compute_average_velocity
    %   功能：根据所有粒子的运动方向计算群体的平均速度向量。
    %   输入：
    %     - theta: 包含所有粒子角度的向量
    %     - v0: 粒子速率
    %   输出：
    %     - V: 群体的平均速度向量 [Vx, Vy]
    V = [mean(v0 * cos(theta)), mean(v0 * sin(theta))];
end
