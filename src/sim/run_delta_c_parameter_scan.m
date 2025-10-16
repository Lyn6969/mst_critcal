% run_delta_c_parameter_scan Delta_c 参数扫描实验脚本
%
% 功能描述:
%   该脚本实现了对运动显著性阈值(cj_threshold)的系统性扫描，
%   计算每个阈值下的集群敏感性Δc，用于研究系统对外源刺激的响应特性
%
% 主要功能:
%   - 参数空间扫描(cj_threshold)
%   - c1/c2实验批量执行
%   - 统计分析和误差估计
%   - 结果可视化和保存
%   - 实验进度跟踪
%
% 算法流程:
%   1. 设置参数扫描范围
%   2. 对每个参数点运行多次c1/c2实验
%   3. 计算统计量(均值、标准差、标准误差)
%   4. 生成可视化结果
%   5. 保存完整实验数据
%
% 输出结果:
%   - Δc随参数变化曲线
%   - 统计不确定性分析
%   - 原始实验数据
%   - 可视化图表
%
% 性能优化:
%   - 内存预分配
%   - 进度跟踪和中间结果保存
%   - 错误处理和恢复机制
%
% 作者：系统生成
% 日期：2025年
% 版本：MATLAB 2025a兼容

clc;        % 清除命令行窗口
clear;      % 清除工作空间变量
close all;  % 关闭所有图形窗口

%% 1. 实验参数设置
% 配置粒子仿真系统的固定参数和扫描参数

% 显示实验标题
fprintf('=================================================\n');
fprintf('          Delta_c 参数扫描实验\n');
fprintf('=================================================\n\n');

% 固定粒子系统参数
params.N = 200;                          % 粒子数量
params.rho = 1;                          % 粒子密度
params.v0 = 1;                           % 粒子速度
params.angleUpdateParameter = 10;        % 角度更新参数
params.angleNoiseIntensity = 0;          % 角度噪声强度
params.T_max = 400;                      % 仿真总时长
params.dt = 0.1;                         % 时间步长
params.radius = 5;                       % 邻居查找半径
params.deac_threshold = 0.1745;          % 取消激活阈值（10度）
params.cj_threshold = 1;                 % 激活阈值（将在循环中修改）

% 固定场地参数
params.fieldSize = 50;                   % 场地边长
params.initDirection = pi/4;             % 初始方向（45度）
params.useFixedField = true;             % 使用固定场地模式

% 外源激活参数
params.stabilization_steps = 200;        % 稳定期步数
params.forced_turn_duration = 200;       % 强制转向后独立状态持续时间

% 参数扫描设置
cj_threshold_min = 0.1;                 % 最小阈值
cj_threshold_max = 8.0;                 % 最大阈值
cj_threshold_step = 0.1;               % 扫描步长
cj_thresholds = cj_threshold_min:cj_threshold_step:cj_threshold_max;  % 阈值序列
num_params = length(cj_thresholds);      % 参数点数量

% 实验重复次数
num_runs = 50;                          % 每个参数点的重复次数

% 计算总实验次数
total_experiments = num_params * num_runs * 2;  % ×2因为c1和c2
fprintf('参数扫描范围: cj_threshold = [%.1f, %.1f], 步长 = %.1f\n', ...
    cj_threshold_min, cj_threshold_max, cj_threshold_step);
fprintf('参数点数量: %d\n', num_params);
fprintf('每个参数重复次数: %d\n', num_runs);
fprintf('总实验次数: %d (预计耗时: %.1f-%.1f 小时)\n\n', ...
    total_experiments, total_experiments*2/3600, total_experiments*3/3600);

%% 2. 数据预分配
% 预分配所有数据存储数组，提高内存使用效率

fprintf('初始化数据结构...\n');

% 原始数据矩阵
c1_raw = zeros(num_params, num_runs);       % c1原始数据 [参数点 × 重复次数]
c2_raw = zeros(num_params, num_runs);       % c2原始数据 [参数点 × 重复次数]

% 统计量数组
c1_mean = zeros(num_params, 1);             % c1平均值
c2_mean = zeros(num_params, 1);             % c2平均值
c1_std = zeros(num_params, 1);              % c1标准差
c2_std = zeros(num_params, 1);              % c2标准差
c1_sem = zeros(num_params, 1);              % c1标准误差
c2_sem = zeros(num_params, 1);              % c2标准误差
delta_c = zeros(num_params, 1);             % 集群敏感性

% 实验记录和错误统计
experiment_log = cell(num_params, 1);       % 实验日志
error_count = zeros(num_params, 2);         % 错误计数 [c1_errors, c2_errors]

%% 3. 创建数据保存文件夹
% 确保数据目录存在，并生成带时间戳的文件名

if ~exist('data', 'dir')
    mkdir('data');
    fprintf('创建data文件夹...\n');
end

% 生成实验文件名
timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
output_filename = sprintf('data/delta_c_scan_%s.mat', timestamp);
temp_filename = sprintf('data/delta_c_scan_%s_temp.mat', timestamp);

fprintf('输出文件: %s\n\n', output_filename);

%% 4. 主实验循环
% 遍历所有参数点，对每个点运行多次c1/c2实验，计算统计量

fprintf('开始参数扫描实验...\n');
fprintf('----------------------------------------\n');

% 记录实验开始时间
experiment_start_time = tic;

% 创建进度条
if usejava('desktop')
    h_waitbar = waitbar(0, '初始化...', 'Name', 'Delta_c 参数扫描进度');
end

% 遍历所有参数点
for param_idx = 1:num_params
    current_cj = cj_thresholds(param_idx);
    params.cj_threshold = current_cj;

    fprintf('\n参数点 %d/%d: cj_threshold = %.2f\n', param_idx, num_params, current_cj);
    param_start_time = tic;

    % 为当前参数点运行多次实验
    for run_idx = 1:num_runs
        % 更新进度
        overall_progress = ((param_idx-1)*num_runs*2 + (run_idx-1)*2) / total_experiments;

        if usejava('desktop')
            waitbar(overall_progress, h_waitbar, ...
                sprintf('参数 %d/%d (cj=%.2f), 运行 %d/%d', ...
                param_idx, num_params, current_cj, run_idx, num_runs));
        end

        if mod(run_idx, 10) == 1
            fprintf('  运行 %d-%d / %d...\n', run_idx, min(run_idx+9, num_runs), num_runs);
        end

        %% 4.1 运行c1实验（1个初发个体）
        try
            % 创建仿真实例
            sim_c1 = ParticleSimulationWithExternalPulse(params);
            sim_c1.external_pulse_count = 1;  % 设置1个初发个体

            % 运行实验
            cascade_size_c1 = sim_c1.runSingleExperiment(1);

            % 计算级联规模（减去初发个体）
            c1_value = (cascade_size_c1 * params.N - 1) / params.N;
            c1_raw(param_idx, run_idx) = c1_value;

        catch ME
            fprintf('    [警告] c1实验失败 (运行%d): %s\n', run_idx, ME.message);
            c1_raw(param_idx, run_idx) = NaN;
            error_count(param_idx, 1) = error_count(param_idx, 1) + 1;
        end

        % 清理内存
        clear sim_c1;

        %% 4.2 运行c2实验（2个初发个体）
        try
            % 创建仿真实例
            sim_c2 = ParticleSimulationWithExternalPulse(params);
            sim_c2.external_pulse_count = 2;  % 设置2个初发个体

            % 运行实验
            cascade_size_c2 = sim_c2.runSingleExperiment(2);

            % 计算级联规模（减去初发个体）
            c2_value = (cascade_size_c2 * params.N - 2) / params.N;
            c2_raw(param_idx, run_idx) = c2_value;

        catch ME
            fprintf('    [警告] c2实验失败 (运行%d): %s\n', run_idx, ME.message);
            c2_raw(param_idx, run_idx) = NaN;
            error_count(param_idx, 2) = error_count(param_idx, 2) + 1;
        end

        % 清理内存
        clear sim_c2;
    end

    %% 4.3 计算当前参数点的统计量
    % 去除NaN值后计算
    c1_valid = c1_raw(param_idx, :);
    c1_valid = c1_valid(~isnan(c1_valid));
    c2_valid = c2_raw(param_idx, :);
    c2_valid = c2_valid(~isnan(c2_valid));

    if ~isempty(c1_valid)
        c1_mean(param_idx) = mean(c1_valid);
        c1_std(param_idx) = std(c1_valid);
        c1_sem(param_idx) = c1_std(param_idx) / sqrt(length(c1_valid));
    else
        c1_mean(param_idx) = NaN;
        c1_std(param_idx) = NaN;
        c1_sem(param_idx) = NaN;
    end

    if ~isempty(c2_valid)
        c2_mean(param_idx) = mean(c2_valid);
        c2_std(param_idx) = std(c2_valid);
        c2_sem(param_idx) = c2_std(param_idx) / sqrt(length(c2_valid));
    else
        c2_mean(param_idx) = NaN;
        c2_std(param_idx) = NaN;
        c2_sem(param_idx) = NaN;
    end

    % 计算delta_c
    delta_c(param_idx) = c2_mean(param_idx) - c1_mean(param_idx);

    % 记录实验信息
    param_elapsed_time = toc(param_start_time);
    experiment_log{param_idx} = sprintf(...
        'cj=%.2f: c1=%.4f±%.4f, c2=%.4f±%.4f, Δc=%.4f, 耗时%.1fs, 错误[%d,%d]', ...
        current_cj, c1_mean(param_idx), c1_std(param_idx), ...
        c2_mean(param_idx), c2_std(param_idx), delta_c(param_idx), ...
        param_elapsed_time, error_count(param_idx, 1), error_count(param_idx, 2));

    fprintf('  完成: %s\n', experiment_log{param_idx});

    %% 4.4 定期保存中间结果（每5个参数点）
    if mod(param_idx, 5) == 0 || param_idx == num_params
        fprintf('  保存中间结果...\n');

        % 准备保存的数据结构
        results = struct();
        results.parameters = params;
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
        results.error_count = error_count;
        results.experiment_log = experiment_log;
        results.timestamp = timestamp;
        results.current_progress = param_idx / num_params;

        save(temp_filename, 'results', '-v7.3');
    end
end

% 关闭进度条
if usejava('desktop') && ishandle(h_waitbar)
    close(h_waitbar);
end

%% 5. 最终数据保存
% 保存完整的实验数据和元数据，确保结果可重现和分析

fprintf('\n----------------------------------------\n');
fprintf('保存最终结果...\n');

% 记录总实验时间
total_experiment_time = toc(experiment_start_time);

% 准备完整的结果结构
results = struct();
results.description = 'Delta_c parameter scan experiment';  % 实验描述
results.parameters = params;                                  % 仿真参数
results.scan_variable = 'cj_threshold';                      % 扫描变量名
results.cj_thresholds = cj_thresholds;                         % 扫描参数值
results.num_runs = num_runs;                                    % 每个参数的重复次数

% 原始数据
results.c1_raw = c1_raw;                                        % c1原始数据
results.c2_raw = c2_raw;                                        % c2原始数据

% 统计量
results.c1_mean = c1_mean;                                      % c1平均值
results.c2_mean = c2_mean;                                      % c2平均值
results.c1_std = c1_std;                                        % c1标准差
results.c2_std = c2_std;                                        % c2标准差
results.c1_sem = c1_sem;                                        % c1标准误差
results.c2_sem = c2_sem;                                        % c2标准误差
results.delta_c = delta_c;                                      % 集群敏感性

% 实验元数据
results.timestamp = timestamp;                                  % 实验时间戳
results.date = datetime('now');                                  % 实验日期时间
results.total_experiments = total_experiments;                   % 总实验次数
results.total_time_seconds = total_experiment_time;             % 总耗时(秒)
results.total_time_hours = total_experiment_time / 3600;          % 总耗时(小时)
results.error_count = error_count;                              % 每个参数的错误次数
results.experiment_log = experiment_log;                         % 实验日志

% MATLAB版本信息
results.matlab_version = version;                               % MATLAB版本信息

% 保存到文件
save(output_filename, 'results', '-v7.3');
fprintf('结果已保存至: %s\n', output_filename);

% 删除临时文件
if exist(temp_filename, 'file')
    delete(temp_filename);
    fprintf('清理临时文件\n');
end

%% 6. 结果摘要
fprintf('\n=================================================\n');
fprintf('                实验完成摘要\n');
fprintf('=================================================\n');
fprintf('总耗时: %.2f 小时\n', total_experiment_time/3600);
fprintf('成功率: %.1f%% (c1), %.1f%% (c2)\n', ...
    100*(1-sum(error_count(:,1))/total_experiments*2), ...
    100*(1-sum(error_count(:,2))/total_experiments*2));

% 找出delta_c的峰值
[max_delta_c, max_idx] = max(delta_c);
optimal_cj = cj_thresholds(max_idx);
fprintf('\nDelta_c 峰值: %.4f (在 cj_threshold = %.2f)\n', max_delta_c, optimal_cj);

% 找出转变区域
threshold_idx = find(delta_c > max_delta_c * 0.5, 1, 'first');
if ~isempty(threshold_idx)
    critical_cj = cj_thresholds(threshold_idx);
    fprintf('临界阈值 (Δc > 50%%峰值): cj_threshold ≈ %.2f\n', critical_cj);
end

%% 7. 快速可视化
fprintf('\n生成快速预览图...\n');

figure('Name', 'Delta_c 参数扫描结果', 'Position', [100, 100, 1200, 400]);

% 子图1: c1和c2随参数变化
subplot(1, 3, 1);
hold on;
errorbar(cj_thresholds, c1_mean, c1_sem, 'b.-', 'LineWidth', 1.5, 'DisplayName', 'c_1');
errorbar(cj_thresholds, c2_mean, c2_sem, 'r.-', 'LineWidth', 1.5, 'DisplayName', 'c_2');
xlabel('c_j threshold');
ylabel('平均级联规模');
title('级联规模 vs 运动显著性阈值');
legend('Location', 'best');
grid on;
xlim([cj_threshold_min, cj_threshold_max]);

% 子图2: delta_c随参数变化
subplot(1, 3, 2);
plot(cj_thresholds, delta_c, 'k.-', 'LineWidth', 2);
xlabel('c_j threshold');
ylabel('\Delta c = c_2 - c_1');
title('集群敏感性 vs 运动显著性阈值');
grid on;
xlim([cj_threshold_min, cj_threshold_max]);
% 标记峰值
hold on;
plot(optimal_cj, max_delta_c, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
text(optimal_cj, max_delta_c, sprintf('  峰值: %.4f\n  cj = %.2f', max_delta_c, optimal_cj), ...
    'VerticalAlignment', 'bottom');

% 子图3: 数据分布箱线图（选择几个代表性参数点）
subplot(1, 3, 3);
representative_indices = round(linspace(1, num_params, min(8, num_params)));

% 预计算总数据量以进行预分配
total_data_points = 0;
for i = 1:length(representative_indices)
    idx = representative_indices(i);
    total_data_points = total_data_points + sum(~isnan(c1_raw(idx, :))) + sum(~isnan(c2_raw(idx, :)));
end

% 预分配数组
boxplot_data = zeros(1, total_data_points);
boxplot_labels = cell(1, total_data_points);
data_idx = 0;

% 填充数据
for i = 1:length(representative_indices)
    idx = representative_indices(i);
    c1_data = c1_raw(idx, ~isnan(c1_raw(idx, :)));
    c2_data = c2_raw(idx, ~isnan(c2_raw(idx, :)));

    % 添加c1数据
    n_c1 = length(c1_data);
    if n_c1 > 0
        boxplot_data(data_idx+1:data_idx+n_c1) = c1_data;
        boxplot_labels(data_idx+1:data_idx+n_c1) = repmat({sprintf('c1\n%.1f', cj_thresholds(idx))}, 1, n_c1);
        data_idx = data_idx + n_c1;
    end

    % 添加c2数据
    n_c2 = length(c2_data);
    if n_c2 > 0
        boxplot_data(data_idx+1:data_idx+n_c2) = c2_data;
        boxplot_labels(data_idx+1:data_idx+n_c2) = repmat({sprintf('c2\n%.1f', cj_thresholds(idx))}, 1, n_c2);
        data_idx = data_idx + n_c2;
    end
end

% 调整到实际使用的长度
boxplot_data = boxplot_data(1:data_idx);
boxplot_labels = boxplot_labels(1:data_idx);

boxplot(boxplot_data, boxplot_labels);
ylabel('级联规模');
title('代表性参数点的数据分布');
set(gca, 'XTickLabelRotation', 0);

sgtitle(sprintf('Delta_c 参数扫描结果 (N=%d, %d次重复)', params.N, num_runs));

% 保存图像
savefig(sprintf('data/delta_c_scan_%s.fig', timestamp));
print(sprintf('data/delta_c_scan_%s.png', timestamp), '-dpng', '-r300');

fprintf('图像已保存\n');
fprintf('\n实验完成！\n');