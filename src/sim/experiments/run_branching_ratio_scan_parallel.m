function run_branching_ratio_scan_parallel()
%RUN_BRANCHING_RATIO_SCAN_PARALLEL 并行扫描运动显著性阈值对平均分支比的影响。
%
% 功能概述:
%   该脚本是一个高性能的并行参数扫描工具，用于研究运动显著性阈值(cj_threshold)
%   对级联过程中平均分支比(b)的影响。分支比是级联理论中的关键参数，表示一个激活
%   粒子平均能够激活的新粒子数量。
%
% 输入参数:
%   - cj_threshold: 运动显著性阈值，控制粒子何时能够激活其他粒子
%   - num_runs: 每个阈值参数的重复实验次数，用于统计平均
%
% 输出结果:
%   - 每个阈值下的平均分支比、标准差和标准误差
%   - 实验数据和时间戳记录
%   - 可视化结果图表
%
% 使用方法:
%   1. 直接运行函数: run_branching_ratio_scan_parallel()
%   2. 结果将自动保存到 data/experiments/branching_ratio_scan/ 目录下
%   3. 可视化图表将生成并保存为 PNG 和 FIG 格式
%
% 注意事项:
%   - 需要并行计算工具箱(Parallel Computing Toolbox)
%   - 运行时间取决于参数范围和重复次数
%   - 大规模扫描可能需要大量内存和计算资源
%
% 算法原理:
%   1. 对每个 cj_threshold 值，运行多次单初发个体实验
%   2. 通过事件日志经验计算分支比
%   3. 使用并行计算加速大规模参数扫描
%   4. 统计分析各参数下的分支比分布

%% 1. 实验设置
clc

fprintf('=================================================\n');
fprintf('   Branching Ratio 参数扫描实验 (并行化版本)\n');
fprintf('=================================================\n\n');

% 获取默认仿真参数
params = default_simulation_parameters();

% 设置参数扫描范围
cj_threshold_min = 0;                           % 最小阈值 (无量纲)
cj_threshold_max = 5.0;                           % 最大阈值 (无量纲)
cj_threshold_step = 0.05;                          % 扫描步长 (无量纲)
cj_thresholds = cj_threshold_min:cj_threshold_step:cj_threshold_max;  % 阈值序列
num_params = numel(cj_thresholds);                % 参数点总数

% 设置实验重复次数和计数器
num_runs = 50;                                   % 每个阈值重复次数 (统计可靠性)
completed_tasks = 0;                              % 全局已完成任务计数器

% 计算总实验数量
total_tasks = num_params * num_runs;
fprintf('参数扫描范围: cj_threshold = [%.1f, %.1f], 步长 = %.1f\n', ...
    cj_threshold_min, cj_threshold_max, cj_threshold_step);
fprintf('参数点数量: %d\n', num_params);
fprintf('每个参数重复次数: %d\n', num_runs);
fprintf('总实验次数: %d\n', total_tasks);

%% 2. 并行池配置
% 配置并行计算环境，利用多核处理器加速大规模参数扫描
% 并行计算策略：将不同参数点的实验分配到不同的worker上并行执行
pool = configure_parallel_pool();
fprintf('使用并行池: %d workers\n', pool.NumWorkers);

%% 3. 数据结构预分配与计时器
% 内存预分配优化：避免在循环中动态扩展数组，提高性能
b_raw = NaN(num_params, num_runs);                % 原始分支比数据矩阵 (参数点×重复次数)

% 实验计时和进度跟踪设置
experiment_start_time = tic;                      % 整体实验计时器开始
progress_update_timer = tic;                      % 进度更新计时器
progress_step = max(1, floor(total_tasks / 100)); % 进度更新步长 (每1%更新一次)

% 创建带时间戳的输出目录和文件
timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));  % 生成唯一时间戳
output_dir = ensure_data_directory(timestamp, params.N);     % 确保数据目录存在
output_filename = fullfile(output_dir, 'data.mat'); % 最终结果文件路径
temp_filename = fullfile(output_dir, 'temp.mat');  % 临时文件路径(用于中间结果保存)

% 设置并行进度监控队列
% DataQueue 允许worker向主线程发送进度更新，而不需要共享变量
progress_queue = parallel.pool.DataQueue;         % 创建数据队列用于进度通信
afterEach(progress_queue, @(increment) update_progress(increment)); % 注册进度更新回调函数

%% 4. 并行扫描
% 核心并行计算部分：使用parfor循环实现参数空间的并行扫描
% 并行策略：每个worker处理一个完整的参数点(包括所有重复实验)
fprintf('\n开始并行实验...\n');
fprintf('----------------------------------------\n');

% 使用Constant对象共享只读参数，减少内存开销和数据传输
% Constant对象在所有worker间共享，避免每个worker复制完整参数结构
shared_params = parallel.pool.Constant(params);

% 并行for循环：自动将任务分配到可用的worker上
% MATLAB会自动处理负载均衡和任务分配
parfor param_idx = 1:num_params
    % 为每个参数点预分配本地结果数组
    local_b = NaN(1, num_runs);

    % 获取当前参数点的配置
    % 每个worker获得参数对象的独立副本，避免竞态条件
    current_params = shared_params.Value;
    current_params.cj_threshold = cj_thresholds(param_idx);

    % 对当前参数点进行多次重复实验
    % 内部循环是串行的，但不同参数点之间是并行的
    for run_idx = 1:num_runs
        % 执行单次分支比实验
        local_b(run_idx) = run_branching_experiment(current_params, 1);
        
        % 发送进度更新信号到主线程
        % 通过DataQueue实现线程安全的进度通信
        send(progress_queue, 1);
    end

    % 将本地结果写入全局结果矩阵
    % parfor确保每个param_idx只被一个worker写入，避免数据竞争
    b_raw(param_idx, :) = local_b;
end

% 计算总实验时间
fprintf('\n实验执行完毕，用时 %.2f 分钟。\n', toc(experiment_start_time) / 60);
total_elapsed_seconds = toc(experiment_start_time);

%% 5. 统计分析
% 对原始实验数据进行统计分析，计算各参数点的分支比统计特性
% 统计指标包括：均值、标准差、标准误差
[b_mean, b_std, b_sem] = compute_statistics(b_raw);

% 统计每个参数点的实验失败次数
% NaN值表示实验失败或异常，用于评估参数范围的可靠性
error_count = sum(isnan(b_raw), 2);

%% 6. 保存结果
% 创建结构化的结果数据容器，包含所有实验数据和元数据
% 这种结构化保存方式便于后续分析和重现实验结果
results = struct();
results.description = 'Branching ratio parameter scan (parallel)';  % 实验描述
results.parameters = params;                                          % 仿真参数
results.scan_variable = 'cj_threshold';                              % 扫描变量名
results.cj_thresholds = cj_thresholds;                               % 扫描参数值
results.num_runs = num_runs;                                         % 每个参数的重复次数
results.b_raw = b_raw;                                               % 原始分支比数据
results.b_mean = b_mean;                                             % 平均分支比
results.b_std = b_std;                                               % 标准差
results.b_sem = b_sem;                                               % 标准误差
results.timestamp = timestamp;                                        % 实验时间戳
results.date = datetime('now');                                       % 实验日期时间
results.total_experiments = total_tasks;                             % 总实验次数
results.total_time_seconds = total_elapsed_seconds;                  % 总耗时(秒)
results.total_time_hours = total_elapsed_seconds / 3600;             % 总耗时(小时)
results.error_count = error_count;                                   % 每个参数的错误次数
results.parallel_workers = pool.NumWorkers;                          % 并行worker数量
results.matlab_version = version;                                     % MATLAB版本信息

% 保存结果到MAT文件，使用-v7.3格式支持大文件和压缩
% 该格式支持大于2GB的文件，并且具有良好的压缩率
save(output_filename, 'results', '-v7.3');
fprintf('结果已保存至: %s\n', output_filename);

% 导出结果为JSON格式，便于大模型解析
% JSON文件保存到与MAT文件相同的目录
json_filename = fullfile(output_dir, 'data.json');
try
    % 创建JSON格式的结果结构体
    json_results = struct();
    json_results.experiment = struct();
    json_results.experiment.description = results.description;
    json_results.experiment.scan_variable = results.scan_variable;
    json_results.experiment.scan_variable_units = 'dimensionless';
    json_results.experiment.timestamp = results.timestamp;
    json_results.experiment.date = char(results.date);
    json_results.experiment.total_experiments = results.total_experiments;
    json_results.experiment.total_time_seconds = results.total_time_seconds;
    json_results.experiment.total_time_hours = results.total_time_hours;
    json_results.experiment.parallel_workers = results.parallel_workers;
    json_results.experiment.matlab_version = results.matlab_version;
    
    % 添加扫描参数
    json_results.parameters = struct();
    json_results.parameters.cj_thresholds = results.cj_thresholds;
    json_results.parameters.num_runs = results.num_runs;
    json_results.parameters.N = results.parameters.N;
    json_results.parameters.rho = results.parameters.rho;
    json_results.parameters.v0 = results.parameters.v0;
    json_results.parameters.angleUpdateParameter = results.parameters.angleUpdateParameter;
    json_results.parameters.angleNoiseIntensity = results.parameters.angleNoiseIntensity;
    json_results.parameters.T_max = results.parameters.T_max;
    json_results.parameters.dt = results.parameters.dt;
    json_results.parameters.radius = results.parameters.radius;
    json_results.parameters.deac_threshold = results.parameters.deac_threshold;
    json_results.parameters.fieldSize = results.parameters.fieldSize;
    json_results.parameters.initDirection = results.parameters.initDirection;
    json_results.parameters.useFixedField = results.parameters.useFixedField;
    json_results.parameters.stabilization_steps = results.parameters.stabilization_steps;
    json_results.parameters.forced_turn_duration = results.parameters.forced_turn_duration;
    
    % 添加分支比数据
    json_results.branching_ratio = struct();
    json_results.branching_ratio.raw_data = results.b_raw;
    json_results.branching_ratio.mean = results.b_mean;
    json_results.branching_ratio.std = results.b_std;
    json_results.branching_ratio.sem = results.b_sem;
    json_results.branching_ratio.error_count = results.error_count;
    
    % 使用MATLAB 2025a的jsonencode函数导出JSON
    options = struct('PrettyPrint', true);
    json_str = jsonencode(json_results, options);
    
    % 写入JSON文件
    fid = fopen(json_filename, 'w', 'n', 'UTF-8');
    fprintf(fid, '%s', json_str);
    fclose(fid);
    
    fprintf('JSON结果已保存至: %s\n', json_filename);
catch ME
    warning(ME.identifier, 'JSON导出失败: %s', ME.message);
end

% 清理临时文件
% 临时文件用于在长时间实验中保存中间结果，防止意外中断导致数据丢失
if exist(temp_filename, 'file')
    delete(temp_filename);
end

%% 7. 快速预览图
% 生成实验结果的可视化图表，提供直观的数据分析
% 包括分支比趋势图和错误统计图
fprintf('\n生成快速预览图...\n');
quicklook_prefix = fullfile(output_dir, 'result');  % 图表文件前缀
render_quicklook_figure(cj_thresholds, b_mean, b_sem, ...
    error_count, params.N, num_runs, pool.NumWorkers, quicklook_prefix);

fprintf('\n实验完成！\n');

%% 嵌套函数 ---------------------------------------------------------------
    function update_progress(increment)
        %UPDATE_PROGRESS 进度更新回调函数，由DataQueue触发
        % 输入参数:
        %   increment - 完成的任务数量(通常为1)
        % 功能: 更新实验进度显示，估算剩余时间，定期保存中间结果
        
        % 更新全局任务计数器
        completed_tasks = completed_tasks + increment;
        
        % 避免除零错误
        if completed_tasks == 0
            return;
        end

        % 控制更新频率：避免过于频繁的进度显示影响性能
        % 条件1: 距离上次更新不足5秒 AND 条件2: 未达到进度步长
        if toc(progress_update_timer) < 5 && ...
                mod(completed_tasks, progress_step) ~= 0
            return;
        end

        % 计算时间统计信息
        elapsed_seconds = toc(experiment_start_time);
        avg_time_per_task = elapsed_seconds / completed_tasks;
        remaining_seconds = avg_time_per_task * (total_tasks - completed_tasks);

        % 显示进度信息
        fprintf('  进度: %.1f%% (%d/%d) | 已用: %.1f分 | 预计剩余: %.1f分\n', ...
            100 * completed_tasks / total_tasks, completed_tasks, total_tasks, ...
            elapsed_seconds / 60, remaining_seconds / 60);

        % 重置进度更新计时器
        progress_update_timer = tic;

        % 定期保存中间结果(每10%进度保存一次)
        % 防止长时间实验因意外中断导致数据丢失
        if mod(completed_tasks, progress_step * 10) == 0
            interim_results = struct('b_raw', b_raw, ...
                'timestamp', timestamp, 'completed_tasks', completed_tasks);
            save(temp_filename, 'interim_results', '-v7.3');
        end
    end
end

%% 局部函数定义 ------------------------------------------------------------
function result = run_branching_experiment(params, pulse_count)
%RUN_BRANCHING_EXPERIMENT 执行一次分支比实验并返回结果
%
% 功能描述:
%   创建粒子仿真对象，运行单次级联实验，并计算平均分支比
%   该函数是并行计算的基本单元，每个worker独立调用
%
% 输入参数:
%   params - 仿真参数结构体，包含cj_threshold等关键参数
%   pulse_count - 外部脉冲数量(通常为1，表示单初发个体)
%
% 输出结果:
%   result - 平均分支比数值，如果实验失败则返回NaN
%
% 错误处理:
%   使用try-catch机制捕获实验异常，返回NaN而不是中断整个并行循环
%   异常信息会被记录到警告日志中，便于调试
    
    try
        % 创建带外部脉冲的粒子仿真对象
        sim = ParticleSimulationWithExternalPulse(params);
        sim.external_pulse_count = pulse_count;  % 设置外部脉冲数量
        
        % 计算分支比
        result = compute_branching_ratio(sim, pulse_count);
    catch ME
        % 异常处理：记录错误信息并返回NaN
        warning('run_branching_experiment:failure', ...
            'Experiment failed (cj=%.3f, pulse=%d): %s', ...
            params.cj_threshold, pulse_count, ME.message);
        result = NaN;
    end
end

function ratio = compute_branching_ratio(sim, pulse_count)
%COMPUTE_BRANCHING_RATIO 运行一次级联并统计平均分支比
%
% 功能描述:
%   执行完整的级联过程，跟踪每个粒子的激活来源，计算平均分支比
%   分支比定义为：每个激活粒子平均能够激活的新粒子数量
%
% 输入参数:
%   sim - 粒子仿真对象，已配置好参数
%   pulse_count - 外部脉冲数量，控制初发个体数量
%
% 输出结果:
%   ratio - 计算得到的平均分支比(无量纲)
%
% 算法流程:
%   1. 初始化仿真环境和跟踪变量
%   2. 运行稳定期(不计入统计)
%   3. 运行级联期，跟踪激活关系
%   4. 计算平均分支比

    % 重置级联跟踪状态并初始化粒子
    sim.resetCascadeTracking();
    sim.initializeParticles();

    % 保存原始脉冲计数并设置新的脉冲计数
    original_pulse_count = sim.external_pulse_count;
    sim.external_pulse_count = pulse_count;
    sim.current_step = 0;

    % 初始化跟踪变量
    parent_flags = false(sim.N, 1);         % 标记父节点
    children_count = zeros(sim.N, 1);       % 统计每个粒子的子节点数量
    counting_enabled = false;               % 是否开始统计
    tracking_deadline = sim.T_max;          % 统计截止步
    track_steps_after_trigger = 100;        % 触发后最大统计步数

    % 稳定期：让系统达到平衡状态，不统计激活关系
    % 这个阶段的激活不纳入分支比计算，避免初始瞬态影响
    for step = 1:sim.stabilization_steps
        sim.step();
    end

    % 级联期：跟踪粒子激活关系，计算分支比
    max_remaining_steps = max(1, sim.T_max - sim.current_step);
    for step = 1:max_remaining_steps
        % 记录当前活跃粒子状态
        prev_active = sim.isActive;
        
        % 执行一步仿真
        sim.step();

        % 捕捉外源脉冲触发时刻，开启统计窗口
        if ~counting_enabled && sim.external_pulse_triggered
            counting_enabled = true;
            tracking_deadline = min(sim.T_max, sim.current_step + track_steps_after_trigger);
        end

        % 识别新激活的粒子
        newly_active = sim.isActive & ~prev_active;
        if counting_enabled && sim.current_step <= tracking_deadline && any(newly_active)
            % 获取新激活粒子的索引
            activated_indices = find(newly_active);
            
            % 遍历每个新激活粒子，确定其父代
            for idx = activated_indices'
                % 标记新激活粒子
                if ~parent_flags(idx)
                    parent_flags(idx) = true;
                end
                
                % 获取激活来源信息
                parent = sim.src_ids{idx};
                if ~isempty(parent)
                    % 提取父代粒子索引
                    parent_idx = parent(1);
                    
                    % 验证父代索引有效性
                    if parent_idx >= 1 && parent_idx <= sim.N
                        parent_idx = round(parent_idx);
                        
                        % 增加父代的子代计数
                        children_count(parent_idx) = children_count(parent_idx) + 1;
                        
                        % 标记父代粒子
                        parent_flags(parent_idx) = true;
                    end
                end
            end
        end

        % 达到统计窗口或级联结束则退出
        if counting_enabled && (~sim.cascade_active || sim.current_step >= tracking_deadline)
            break;
        end
    end

    % 计算平均分支比
    if ~counting_enabled
        ratio = 0;
    else
        parents = find(parent_flags);  % 找到所有父代粒子
        if isempty(parents)
            ratio = 0;  % 没有父代粒子，分支比为0
        else
            % 平均分支比 = 总子代数 / 父代粒子数
            ratio = sum(children_count(parents)) / numel(parents);
        end
    end

    % 恢复原始脉冲计数
    sim.external_pulse_count = original_pulse_count;
end

function params_out = default_simulation_parameters()
%DEFAULT_SIMULATION_PARAMETERS 生成与 Delta_c 扫描一致的默认参数
%
% 功能描述:
%   创建粒子仿真所需的默认参数集合，确保与其他实验的一致性
%   这些参数经过精心调整，适合研究分支比行为
%
% 输出结果:
%   params_out - 包含所有仿真参数的结构体
%
% 参数说明:
%   N - 粒子总数，影响统计可靠性和计算复杂度
%   rho - 粒子密度，影响相互作用频率
%   v0 - 粒子基础速度，控制运动尺度
%   angleUpdateParameter - 角度更新参数，影响转向行为
%   angleNoiseIntensity - 角度噪声强度，增加随机性
%   T_max - 最大仿真时间，防止无限运行
%   dt - 时间步长，影响数值精度和计算速度
%   radius - 粒子相互作用半径，决定激活范围
%   deac_threshold - 去激活阈值，控制粒子失活条件
%   cj_threshold - 运动显著性阈值，关键参数，控制激活条件
%   fieldSize - 仿真场域大小，影响边界效应
%   initDirection - 初始方向，设置初始运动状态
%   useFixedField - 是否使用固定场域，影响边界条件
%   stabilization_steps - 稳定期步数，确保系统达到平衡
%   forced_turn_duration - 强制转向持续时间，控制外部刺激

    params_out.N = 200;                      % 粒子总数 (个)
    params_out.rho = 1;                      % 粒子密度 (无量纲)
    params_out.v0 = 1;                       % 基础速度 (单位/时间)
    params_out.angleUpdateParameter = 10;    % 角度更新参数 (无量纲)
    params_out.angleNoiseIntensity = 0;      % 角度噪声强度 (弧度)
    params_out.T_max = 400;                  % 最大仿真时间 (时间单位)
    params_out.dt = 0.1;                     % 时间步长 (时间单位)
    params_out.radius = 5;                   % 相互作用半径 (空间单位)
    params_out.deac_threshold = 0.1745;      % 去激活阈值 (弧度，约10度)
    params_out.cj_threshold = 1;             % 运动显著性阈值 (无量纲，将被扫描)
    params_out.fieldSize = 50;               % 场域大小 (空间单位)
    params_out.initDirection = pi / 4;       % 初始方向 (弧度，45度)
    params_out.useFixedField = true;         % 使用固定场域 (布尔值)
    params_out.stabilization_steps = 200;    % 稳定期步数 (步)
    params_out.forced_turn_duration = 200;   % 强制转向持续时间 (步)
end

function [mean_values, std_values, sem_values] = compute_statistics(raw_data)
%COMPUTE_STATISTICS 计算均值、标准差与标准误差
%
% 功能描述:
%   对原始实验数据进行统计分析，处理可能的缺失值(NaN)
%   提供三种统计指标：均值、标准差和标准误差
%
% 输入参数:
%   raw_data - 原始数据矩阵，每行代表一个参数点，每列代表一次重复实验
%
% 输出结果:
%   mean_values - 每个参数点的平均值
%   std_values - 每个参数点的标准差
%   sem_values - 每个参数点的标准误差(标准差/√样本数)
%
% 算法特点:
%   - 自动处理NaN值，排除无效数据
%   - 使用向量化操作提高计算效率
%   - 标准误差提供均值估计的可靠性度量

    % 预分配输出数组，初始化为NaN
    mean_values = NaN(size(raw_data, 1), 1);
    std_values = mean_values;
    sem_values = mean_values;

    % 逐行计算统计量
    for idx = 1:size(raw_data, 1)
        % 提取有效样本(排除NaN值)
        valid_samples = raw_data(idx, ~isnan(raw_data(idx, :)));
        
        % 跳过没有有效样本的行
        if isempty(valid_samples)
            continue;
        end
        
        % 计算统计量
        mean_values(idx) = mean(valid_samples);                                    % 均值
        std_values(idx) = std(valid_samples, 0);                                  % 样本标准差(N-1分母)
        sem_values(idx) = std_values(idx) / sqrt(numel(valid_samples));           % 标准误差
    end
end

function output_dir = ensure_data_directory(timestamp, N)
%ENSURE_DATA_DIRECTORY 创建实验数据目录（按时间戳组织）
%
% 功能描述:
%   创建层次化的数据目录结构，确保实验数据有序组织
%   使用时间戳作为子目录名，避免不同实验的数据混淆
%
% 输入参数:
%   timestamp - 时间戳字符串，格式为'yyyyMMdd_HHmmss'
%   N - 个体数目，用于标识实验规模
%
% 输出结果:
%   output_dir - 完整的目录路径
%
% 目录结构:
%   ./data/experiments/branching_ratio_scan/N{N}_{timestamp}/
%
% 设计理念:
%   - 层次化组织便于管理和查找
%   - 时间戳确保唯一性
%   - 个体数目标识实验规模
%   - 与其他实验类型保持一致的命名约定

    % 构建完整目录路径，添加个体数目标记
    output_dir = fullfile(pwd, 'data', 'experiments', 'branching_ratio_scan', sprintf('N%d_%s', N, timestamp));
    
    % 检查目录是否存在，不存在则创建
    if ~isfolder(output_dir)
        mkdir(output_dir);
    end
end

function pool = configure_parallel_pool()
%CONFIGURE_PARALLEL_POOL 确保本地并行池已就绪，如无则按上限启动
%
% 功能描述:
%   智能配置并行计算环境，确保最优的并行性能
%   自动检测可用资源并创建适当规模的并行池
%
% 输出结果:
%   pool - 配置好的并行池对象
%
% 并行策略:
%   - 检查并行计算工具箱许可证
%   - 复用现有并行池(避免重复创建开销)
%   - 限制最大worker数(防止资源耗尽)
%   - 优先使用本地集群(减少网络开销)
%
% 性能考虑:
%   - 最大worker数限制为40，避免过多进程导致上下文切换开销
%   - 使用本地集群确保高效通信
%   - 错误处理提供清晰的诊断信息

    % 检查并行计算工具箱许可证
    if ~license('test', 'Distrib_Computing_Toolbox')
        error('Parallel Computing Toolbox license is required.');
    end

    % 获取当前并行池(如果存在)
    pool = gcp('nocreate');
    
    % 如果没有并行池，则创建一个新的
    if isempty(pool)
        % 获取本地集群配置
        cluster = parcluster('local');
        
        % 确定最优worker数量
        % 限制最大值以平衡性能和资源使用
        max_workers = min(cluster.NumWorkers, 40);
        
        % 检查可用worker数量
        if max_workers < 1
            error('No available workers for parallel pool.');
        end
        
        % 创建并行池
        pool = parpool(cluster, max_workers);
    end
end

function render_quicklook_figure(thresholds, b_mean, b_sem, ...
    error_count, N, num_runs, num_workers, output_prefix)
%RENDER_QUICKLOOK_FIGURE 生成分支比扫描的双联图预览
%
% 功能描述:
%   创建包含两个子图的综合可视化图表，直观展示分支比扫描结果
%   左图显示分支比趋势，右图显示实验失败统计
%
% 输入参数:
%   thresholds - 运动显著性阈值数组 (x轴数据)
%   b_mean - 平均分支比数组 (左图y轴数据)
%   b_sem - 标准误差数组 (左图误差棒)
%   error_count - 失败次数数组 (右图y轴数据)
%   N - 粒子总数 (用于图标题)
%   num_runs - 重复次数 (用于图标题)
%   num_workers - 并行worker数 (用于图标题)
%   output_prefix - 输出文件前缀路径
%
% 输出文件:
%   - [output_prefix].fig: MATLAB可编辑图形格式
%   - [output_prefix].png: 高分辨率PNG图像(300dpi)
%
% 设计特点:
%   - 双联图设计提供全面的数据视图
%   - 误差棒显示统计不确定性
%   - 高分辨率输出适合发表使用
%   - 自动保存多种格式满足不同需求

    % 创建图形窗口，设置标题和位置
    fig = figure('Name', 'Branching Ratio 并行扫描结果', ...
        'Position', [100, 100, 900, 400]);  % 窗口位置和大小

    % 子图(1): 平均分支比趋势图
    subplot(1, 2, 1);
    errorbar(thresholds, b_mean, b_sem, 'b.-', 'LineWidth', 1.5);  % 蓝色误差棒图
    xlabel('c_j threshold');                    % x轴标签
    ylabel('平均分支比 b');                      % y轴标签
    title('平均分支比 vs 运动显著性阈值');       % 子图标题
    grid on;                                     % 显示网格

    % 子图(2): 失败次数统计图
    subplot(1, 2, 2);
    bar(thresholds, error_count, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'none');  % 灰色柱状图
    xlabel('c_j threshold');                    % x轴标签
    ylabel('失败次数');                          % y轴标签
    title('每个阈值的实验失败数');               % 子图标题
    grid on;                                     % 显示网格

    % 添加总标题，包含关键实验参数
    sgtitle(sprintf('分支比分析 (N=%d, %d次重复, %d workers)', ...
        N, num_runs, num_workers));

    % 保存图形文件
    % .fig格式: MATLAB原生格式，可编辑
    savefig([output_prefix, '.fig']);
    
    % .png格式: 高分辨率位图，适合文档和演示
    % -r300参数设置300dpi分辨率，确保打印质量
    print([output_prefix, '.png'], '-dpng', '-r300');
    
    % 关闭图形窗口，释放内存
    close(fig);
end
