% 分支比统计可视化脚本（单次外源脉冲级联）
% =========================================================================
% 功能描述：
%   本脚本用于可视化粒子系统中外源脉冲触发的级联传播过程，并计算
%   平均分支比。分支比是指每个被激活的粒子平均能够激活的新粒子数量，
%   是衡量级联传播能力的重要指标。
%
% 主要功能：
%   1. 模拟粒子系统在不同运动显著性阈值下的行为
%   2. 追踪外源脉冲触发的级联传播链
%   3. 实时计算和可视化平均分支比随时间的变化
%   4. 显示父节点和子节点的数量统计
%   5. 记录并显示传播事件的时间线
%
% 输入参数：
%   - cj_threshold: 运动显著性阈值，用户通过命令行输入
%
% 输出：
%   - 实时可视化界面，包含粒子状态、分支比曲线、数量统计和事件记录
%
% 依赖文件：
%   - ParticleSimulationWithExternalPulse.m: 粒子仿真类
%
% 作者：系统自动生成
% 创建日期：2025年
% 版本：MATLAB 2025a兼容
% =========================================================================

clc;        % 清除命令行窗口
clear;      % 清除工作空间变量
close all;  % 关闭所有图形窗口

%% 1. 图形默认设置
% 设置全局图形属性，确保所有图形具有一致的外观
set(0, 'DefaultAxesFontSize', 10);      % 设置坐标轴字体大小
set(0, 'DefaultAxesFontName', 'Arial');  % 设置坐标轴字体类型
set(0, 'DefaultTextFontSize', 10);       % 设置文本字体大小
set(0, 'DefaultTextFontName', 'Arial');  % 设置文本字体类型
set(0, 'DefaultLineLineWidth', 1.5);     % 设置线条默认宽度

%% 2. 参数设置
% 用户交互输入和系统参数初始化
fprintf('=== 分支比可视化 ===\n');
% 获取用户输入的运动显著性阈值，该阈值决定了粒子激活的难易程度
cj_threshold = input('请输入运动显著性阈值 (默认 2.0): ');
if isempty(cj_threshold)
    cj_threshold = 4.0;  % 如果用户未输入，使用默认值
end

% 粒子系统基本参数
params.N = 200;                    % 粒子总数1
params.rho = 1;                    % 粒子密度参数
params.v0 = 1;                     % 粒子基础速度
params.angleUpdateParameter = 10;  % 角度更新参数，影响粒子转向强度
params.angleNoiseIntensity = 0;    % 角度噪声强度，0表示无随机噪声
params.T_max = 500;                % 最大仿真时间步数
params.dt = 0.1;                   % 时间步长
params.radius = 5;                 % 粒子交互半径
params.deac_threshold = 0.1745;    % 去激活阈值（弧度，约10度）
params.cj_threshold = cj_threshold;% 运动显著性阈值

% 仿真环境参数
params.fieldSize = 50;             % 仿真区域大小（正方形区域）
params.initDirection = pi/4;       % 初始运动方向（45度）
params.useFixedField = true;       % 是否使用固定边界

% 级联传播控制参数
params.stabilization_steps = 150;  % 系统稳定期步数，在此期间不施加外源脉冲
params.forced_turn_duration = 200; % 强制转向持续时间
params.cascade_end_threshold = 10;  % 级联结束阈值（连续无激活步数）
params.external_pulse_count = 1;   % 外源脉冲激活的粒子数量

track_steps_after_trigger = 100;   % 外源触发后最多统计的时间步数

% 显示设置的实验参数，便于用户确认
fprintf('\n实验参数:\n');
fprintf('  • cj_threshold = %.2f\n', params.cj_threshold);
fprintf('  • 外源初发个体数 = %d\n', params.external_pulse_count);
fprintf('  • 稳定期 = %d 步\n', params.stabilization_steps);

%% 3. 创建仿真对象
% 使用预设参数创建粒子仿真对象，该类实现了包含外源脉冲的粒子系统
simulation = ParticleSimulationWithExternalPulse(params);
%% 4. 创建可视化界面
% 创建主图形窗口，包含多个子图用于显示不同类型的信息
fig = figure('Name', sprintf('分支比可视化 (cj=%.2f)', params.cj_threshold), ...
    'NumberTitle', 'off', 'Color', 'white', 'Position', [80, 60, 1600, 900]);

% 创建5个子图，分别用于显示不同信息
ax_particles = subplot('Position', [0.05 0.35 0.4 0.55]);  % 左侧：粒子状态显示
ax_b = subplot('Position', [0.5 0.65 0.45 0.25]);        % 右上：分支比曲线
ax_counts = subplot('Position', [0.5 0.38 0.45 0.2]);     % 右中：数量统计
ax_events = subplot('Position', [0.5 0.1 0.45 0.18]);     % 右下：事件统计
ax_info = subplot('Position', [0.05 0.07 0.4 0.22]);      % 左下：详细信息

%% 5. 初始化粒子展示
% 配置粒子状态显示区域
subplot(ax_particles);
hold on;
axis([0 params.fieldSize 0 params.fieldSize]);  % 设置坐标轴范围
axis square;                                     % 保持正方形比例
grid on;                                         % 显示网格
xlabel('X');                                     % X轴标签
ylabel('Y');                                     % Y轴标签
title('粒子状态');                               % 图标题

% 创建粒子散点图，显示所有粒子的位置
particles_plot = scatter(simulation.positions(:,1), simulation.positions(:,2), ...
    40, repmat([0.6 0.6 0.6], params.N, 1), 'filled', 'MarkerFaceAlpha', 0.8);

% 创建箭头图，显示粒子的运动方向
arrows_plot = quiver(simulation.positions(:,1), simulation.positions(:,2), ...
    0.3*cos(simulation.theta), 0.3*sin(simulation.theta), 0, ...
    'Color', [0.4 0.4 0.4 0.6], 'LineWidth', 0.9, 'MaxHeadSize', 0.5);

% 创建外源激活粒子的特殊标记
external_plot = scatter([], [], 90, [0.1 0.8 0.1], 'o', 'LineWidth', 2);
legend({'粒子', '方向', '外源激活'}, 'Location', 'northeast');
hold off;

%% 6. 初始化统计图
% 配置分支比曲线图
subplot(ax_b);
hold on;
xlabel('时间步');
ylabel('平均分支比 b');
title('平均分支比随时间演化');
b_plot = plot(NaN, NaN, 'b-', 'LineWidth', 1.6);  % 初始化空曲线
yline(1, '--r', 'b = 1', 'LineWidth', 1.1);       % 添加参考线b=1
ylim([0, 2]);                                      % 设置Y轴范围
hold off;

% 配置父节点与子节点数量统计图
subplot(ax_counts);
hold on;
xlabel('时间步');
ylabel('数量');
title('父节点与累计子节点');
parent_plot = plot(NaN, NaN, '-', 'Color', [0.2 0.5 0.8], 'LineWidth', 1.8, ...
    'DisplayName', '父节点数');
child_plot = plot(NaN, NaN, '-', 'Color', [0.9 0.4 0.1], 'LineWidth', 1.8, ...
    'DisplayName', '累计子节点数');
legend('Location', 'northwest');
hold off;

% 配置每步新增事件统计图
subplot(ax_events);
hold on;
xlabel('时间步');
ylabel('事件数');
title('每步新增父/子节点');
new_parent_plot = plot(NaN, NaN, 'o-', 'Color', [0.2 0.6 0.8], ...
    'LineWidth', 1.3, 'MarkerSize', 4, 'DisplayName', '新增父节点');
new_child_plot = plot(NaN, NaN, 's-', 'Color', [0.9 0.4 0.2], ...
    'LineWidth', 1.3, 'MarkerSize', 4, 'DisplayName', '新增子节点');
legend('Location', 'northwest');
hold off;

% 配置信息显示区域
subplot(ax_info);
axis off;  % 关闭坐标轴
info_text = text(0.03, 0.95, '等待仿真开始...', 'FontSize', 11, ...
    'FontWeight', 'bold', 'VerticalAlignment', 'top');
detail_text = text(0.03, 0.55, '', 'FontSize', 10, 'VerticalAlignment', 'top');
event_text = text(0.52, 0.55, '', 'FontSize', 10, 'VerticalAlignment', 'top');

%% 7. 数据结构初始化
% 初始化用于存储历史数据的数组，采用预分配内存以提高性能
avg_b_history = NaN(params.T_max, 1);      % 平均分支比历史记录
parent_history = NaN(params.T_max, 1);     % 父节点数量历史记录
child_history = NaN(params.T_max, 1);      % 子节点数量历史记录
new_parent_history = zeros(params.T_max, 1); % 新增父节点历史记录
new_child_history = zeros(params.T_max, 1); % 新增子节点历史记录
time_vec = (0:params.T_max)' * params.dt;   % 时间向量
global_heading_series = NaN(params.T_max + 1, 1); % 全局平均方向序列
global_heading_series(1) = compute_global_heading(simulation.theta, params.v0);

% 初始化粒子状态跟踪数组
parent_flags = false(params.N, 1);         % 标记粒子是否为父节点
children_count = zeros(params.N, 1);       % 记录每个粒子的子节点数量

% 初始化控制变量
counting_enabled = false;                  % 是否开始统计分支比
seed_flags = false(params.N, 1);           % 标记外源激活的种子粒子
branching_events = strings(10, 1);         % 存储最近的分支事件记录

% 输出仿真开始信息
fprintf('\n开始仿真...\n');
child_total = 0;    % 累计子节点总数
final_b = 0;        % 最终分支比
tracking_deadline = params.T_max;  % 触发后允许统计的最晚时间步

%% 8. 主仿真循环
% 遍历每个时间步，进行粒子状态更新和分支比统计
for t_step = 1:params.T_max
    % 保存前一时刻的激活状态，用于检测新激活的粒子
    prev_active = simulation.isActive;
    % 执行一步仿真
    simulation.step();

    % 获取当前仿真状态
    positions = simulation.positions;           % 粒子位置
    theta = simulation.theta;                   % 粒子运动方向
    is_active = simulation.isActive;            % 当前激活状态
    ever_activated = simulation.everActivated;  % 历史激活状态
    is_external = simulation.isExternallyActivated; % 外源激活状态
    global_heading_series(t_step + 1) = compute_global_heading(theta, params.v0);

    % 检查是否开始统计分支比（外源脉冲触发后）
    if ~counting_enabled && simulation.external_pulse_triggered
        counting_enabled = true;            % 开始统计
        seed_flags = false(params.N, 1);    % 重置种子标记
        seeds = simulation.getExternallyActivatedIndices(); % 获取外源激活粒子索引
        if ~isempty(seeds)
            seed_flags(unique(seeds)) = true; % 标记种子粒子
        end
        fprintf('t = %d: 外源脉冲触发，开始统计分支。\n', t_step);
        tracking_deadline = min(params.T_max, t_step + track_steps_after_trigger);
    end

    % 检测新激活的粒子（当前激活但前一时刻未激活）
    newly_active = is_active & ~prev_active;
    new_parents = 0;   % 当前步骤新增的父节点数
    new_children = 0;  % 当前步骤新增的子节点数

    % 仅在统计窗口内记录分支信息
    if counting_enabled && t_step <= tracking_deadline
        if any(newly_active)
            indices = find(newly_active);  % 获取新激活粒子的索引
            for idx = indices'  % 遍历每个新激活的粒子
                % 如果该粒子尚未被标记为父节点，则标记并计数
                if ~parent_flags(idx)
                    parent_flags(idx) = true;
                    new_parents = new_parents + 1;
                end
                
                % 获取激活源信息
                src = simulation.src_ids{idx};
                if ~isempty(src)
                    % 如果有激活源，说明是粒子间传播
                    parent_idx = round(src(1));
                    if parent_idx >= 1 && parent_idx <= params.N
                        children_count(parent_idx) = children_count(parent_idx) + 1;
                        new_children = new_children + 1;
                        parent_flags(parent_idx) = true;
                        event_str = sprintf('t=%d: %d → %d', t_step, parent_idx, idx);
                    else
                        % 激活源索引无效，视为外源激活
                        event_str = sprintf('t=%d: 外源激活 → %d', t_step, idx);
                    end
                else
                    % 无激活源，直接外源激活
                    event_str = sprintf('t=%d: 外源激活 → %d', t_step, idx);
                end
                % 更新分支事件记录（保持最新10条）
                branching_events = [event_str; branching_events(1:end-1)];
            end
        end
    else
        % 未进入统计窗口时清空事件显示，避免显示过期信息
        branching_events = [""; branching_events(1:end-1)];
    end

    % 计算当前分支比统计
    if counting_enabled && t_step <= tracking_deadline
        parent_count = sum(parent_flags);  % 当前父节点总数
        child_total = sum(children_count); % 当前子节点总数
    else
        parent_count = 0;
        child_total = 0;
    end
    if parent_count == 0
        avg_b = 0;  % 避免除零错误
    else
        avg_b = child_total / parent_count;  % 平均分支比 = 子节点数/父节点数
    end
    final_b = avg_b;  % 保存当前分支比作为最终值

    % 记录历史数据
    avg_b_history(t_step) = avg_b;
    parent_history(t_step) = parent_count;
    child_history(t_step) = child_total;
    new_parent_history(t_step) = new_parents;
    new_child_history(t_step) = new_children;

    %% 9. 可视化更新（每2步或最后一步或级联结束时更新）
    % 控制可视化更新频率，避免过于频繁的刷新影响性能
    if mod(t_step, 2) == 0 || t_step == params.T_max || ~simulation.cascade_active
        % 更新粒子状态显示
        subplot(ax_particles);
        % 初始化所有粒子的颜色和大小
        colors = repmat([0.6 0.6 0.6], params.N, 1);  % 默认灰色
        sizes = 35 * ones(params.N, 1);               % 默认大小
        
        % 标记曾经激活但当前未激活的粒子（红色）
        ever_only = ever_activated & ~is_active;
        colors(ever_only, :) = repmat([0.85 0.2 0.2], sum(ever_only), 1);
        sizes(ever_only) = 45;
        
        % 标记当前激活的粒子（黄色）
        active_idx = is_active;
        colors(active_idx, :) = repmat([1.0 0.85 0.2], sum(active_idx), 1);
        sizes(active_idx) = 55;
        
        % 更新粒子图数据
        set(particles_plot, 'XData', positions(:,1), 'YData', positions(:,2), ...
            'CData', colors, 'SizeData', sizes);
        % 更新方向箭头
        set(arrows_plot, 'XData', positions(:,1), 'YData', positions(:,2), ...
            'UData', 0.3*cos(theta), 'VData', 0.3*sin(theta));
        % 更新外源激活粒子标记
        set(external_plot, 'XData', positions(is_external,1), 'YData', positions(is_external,2));

        % 更新分支比曲线
        subplot(ax_b);
        set(b_plot, 'XData', 1:t_step, 'YData', avg_b_history(1:t_step));
        % 动态调整Y轴范围，确保曲线完整显示
        max_b = max(avg_b_history(1:t_step), [], 'omitnan');
        if isempty(max_b) || isnan(max_b)
            max_b = 0;
        end
        ylim([0, max(2, max_b * 1.1)]);

        % 更新数量统计图
        subplot(ax_counts);
        set(parent_plot, 'XData', 1:t_step, 'YData', parent_history(1:t_step));
        set(child_plot, 'XData', 1:t_step, 'YData', child_history(1:t_step));
        % 动态调整Y轴范围
        max_child = max(child_history(1:t_step), [], 'omitnan');
        if isempty(max_child) || isnan(max_child)
            max_child = 0;
        end
        ylim([0, max(1, max_child * 1.1)]);

        % 更新事件统计图
        subplot(ax_events);
        set(new_parent_plot, 'XData', 1:t_step, 'YData', new_parent_history(1:t_step));
        set(new_child_plot, 'XData', 1:t_step, 'YData', new_child_history(1:t_step));
        % 动态调整Y轴范围
        ylim([0, max(1, max(new_child_history(1:t_step))*1.2)]);

        % 更新信息显示区域
        subplot(ax_info);
        % 构建状态信息文本
        status_lines = {
            sprintf('时间步: %d / %d', t_step, params.T_max);
            sprintf('平均分支比 b: %.3f', avg_b);
            sprintf('父节点数: %d', parent_count);
            sprintf('累计子节点数: %d', child_total);
            sprintf('当前外源激活: %d', sum(is_external));
            sprintf('累计激活比例: %.2f%%', sum(ever_activated)/params.N*100)
        };
        set(info_text, 'String', strjoin(status_lines, '\n'));
        % 更新详细信息
        set(detail_text, 'String', sprintf('新增父节点: %d\n新增子节点: %d', ...
            new_parents, new_children));
        % 更新事件记录
        event_str = strjoin(branching_events(~strcmp(branching_events, "")), '\n');
        if isempty(event_str)
            event_str = '尚无传播事件';
        end
        set(event_text, 'String', sprintf('最新传播事件:\n%s', event_str));

        % 刷新图形显示
        drawnow;
    end

    %% 10. 级联结束检测
    % 检查级联是否已经结束或达到预设统计窗口
    if counting_enabled && (~simulation.cascade_active || t_step >= tracking_deadline)
        if t_step >= tracking_deadline && simulation.cascade_active
            fprintf('t = %d: 达到预设统计窗口，停止统计。\n', t_step);
        else
            fprintf('t = %d: 级联结束，停止统计。\n', t_step);
        end
        break;
    end
end

final_step = min(t_step, params.T_max);

%% 11. 基于高阶拓扑逻辑的持久性估计
heading_segment = global_heading_series(1:final_step + 1);
unwrapped_heading = unwrap(heading_segment);
heading_diff = unwrapped_heading - unwrapped_heading(1);
msd_heading = heading_diff .^ 2;
burn_in_index = max(2, min(final_step + 1, params.stabilization_steps + 1));
fit_time = time_vec(burn_in_index:final_step + 1);
fit_msd = msd_heading(burn_in_index:final_step + 1);

fit_curve = NaN(size(fit_time));
if numel(fit_time) >= 2 && any(abs(diff(fit_msd)) > eps)
    t_shift = fit_time - fit_time(1);
    msd_shift = fit_msd - fit_msd(1);
    smooth_window = max(5, floor(numel(msd_shift) * 0.1));
    if smooth_window > 1
        msd_shift = smoothdata(msd_shift, 'movmean', smooth_window);
    end
    slope = lsqnonneg(t_shift(:), msd_shift(:));
    fit_curve = slope * t_shift + fit_msd(1);
    if slope > 0
        D_A = slope;
        if slope <= 1e-6
            persistence_P = Inf;
        else
            persistence_P = 1 / sqrt(D_A);
        end
    else
        D_A = 0;
        persistence_P = Inf;
    end
else
    D_A = NaN;
    persistence_P = NaN;
end

if ~isnan(persistence_P)
    if isinf(persistence_P)
        fprintf('基于高阶拓扑逻辑的持久性 P → ∞ (D_A ≈ 0)\n');
    else
        fprintf('基于高阶拓扑逻辑的持久性 P = %.4f (D_A = %.4e)\n', persistence_P, D_A);
    end
else
    fprintf('持久性估计失败：有效拟合样本不足。\n');
end

persistence_results = struct();
persistence_results.description = 'High-order inspired persistence metrics for branching visualization';
persistence_results.time = time_vec(1:final_step + 1);
persistence_results.global_heading = heading_segment;
persistence_results.msd = msd_heading(1:final_step + 1);
persistence_results.fit_time = fit_time;
persistence_results.fit_msd = fit_msd;
persistence_results.fit_curve = fit_curve;
persistence_results.burn_in_index = burn_in_index;
persistence_results.D_A = D_A;
persistence_results.P = persistence_P;
assignin('base', 'branching_persistence_results', persistence_results);

%% 12. 数据导出为CSV
% 将重要的统计数据导出为CSV文件，便于后续分析
fprintf('正在导出数据为CSV文件...\n');

% 创建数据矩阵，包含所有历史记录
data_matrix = [avg_b_history, parent_history, child_history, ...
               new_parent_history, new_child_history];

% 移除NaN值（未使用的步数）
valid_steps = ~isnan(avg_b_history);
valid_data = data_matrix(valid_steps, :);

% 创建时间步向量
time_steps = find(valid_steps);

% 创建带有表头的完整数据矩阵
csv_data = [time_steps, valid_data];

% 定义CSV文件名，包含参数信息
csv_filename = sprintf('branching_ratio_data_cj_%.2f_%s.csv', ...
    params.cj_threshold, datetime("now", "Format", "yyyyMMdd_HHmmss"));

% 定义列标题
headers = {'TimeStep', 'AvgBranchingRatio', 'ParentCount', 'ChildCount', ...
           'NewParentCount', 'NewChildCount'};

% 使用writematrix导出数据（MATLAB 2025a推荐方法）
try
    writematrix(csv_data, csv_filename);
    fprintf('数据已成功导出到: %s\n', csv_filename);
    
    % 也创建一个包含表头的版本，便于阅读
    csv_filename_with_headers = sprintf('branching_ratio_data_cj_%.2f_%s_with_headers.csv', ...
        params.cj_threshold, datetime("now", "Format", "yyyyMMdd_HHmmss"));
    
    % 创建表头行
    header_line = strjoin(headers, ',');
    
    % 将数据转换为字符串矩阵
    data_strings = num2str(csv_data, '%.6f');
    
    % 写入带表头的CSV文件
    fid = fopen(csv_filename_with_headers, 'w');
    fprintf(fid, '%s\n', header_line);
    for i = 1:size(data_strings, 1)
        fprintf(fid, '%s\n', strjoin(data_strings(i, :), ','));
    end
    fclose(fid);
    
    fprintf('带表头的数据已导出到: %s\n', csv_filename_with_headers);
    
catch ME
    fprintf('CSV导出出错: %s\n', ME.message);
    % 备用方案：使用writematrix
    try
        writematrix(csv_data, csv_filename);
        fprintf('使用备用方法成功导出到: %s\n', csv_filename);
    catch
        fprintf('CSV导出失败，请检查文件权限。\n');
    end
end

%% 13. 仿真结束
% 输出最终统计结果
fprintf('仿真结束。最终平均分支比 b = %.3f\n', final_b);
if ~isnan(persistence_P)
    if isinf(persistence_P)
        fprintf('最终持久性指标 P → ∞，扩散系数 D_A ≈ 0\n');
    else
        fprintf('最终持久性指标 P = %.4f，扩散系数 D_A = %.4e\n', persistence_P, D_A);
    end
end

%% ========================================================================
%% 辅助函数
%% ========================================================================
function heading = compute_global_heading(theta, v0)
% compute_global_heading 计算全局平均速度方向
    vx = v0 * cos(theta);
    vy = v0 * sin(theta);
    heading = atan2(mean(vy), mean(vx));
end
