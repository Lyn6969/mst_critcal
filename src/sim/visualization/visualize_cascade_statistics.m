% 增强版级联统计可视化脚本
% 在run_external_pulse_test基础上增加级联统计信息的实时显示
clc;
clear;
close all;

%% 1. 设置图形默认属性
set(0, 'DefaultAxesFontSize', 10);
set(0, 'DefaultAxesFontName', 'Arial');
set(0, 'DefaultTextFontSize', 10);
set(0, 'DefaultTextFontName', 'Arial');
set(0, 'DefaultLineLineWidth', 1.5);

%% 2. 实验模式选择
fprintf('=== 级联统计可视化系统 ===\n');
fprintf('选择实验模式：\n');
fprintf('1 - c1实验（1个初发个体）\n');
fprintf('2 - c2实验（2个初发个体）\n');
fprintf('3 - 对比实验（先c1后c2）\n');


experiment_mode = input('请输入选择（默认为1）: ');
if isempty(experiment_mode)
    experiment_mode = 1;
end

%% 3. 参数设置
params.N = 200;                      % 粒子数量（适中以便观察）
params.rho = 1;                     % 粒子密度
params.v0 = 1;                      % 粒子速度
params.angleUpdateParameter = 10;    % 角度更新参数
params.angleNoiseIntensity = 0;   % 较小噪声便3
% 于观察级联
params.T_max = 500;                 % 最大仿真时间
params.dt = 0.1;                    % 时间步长
params.cj_threshold = 0.5;            % 激活阈值
params.radius = 5;                  % 邻居查找半径
params.deac_threshold = 0.1745;     % 取消激活阈值

% 固定场地参数
params.fieldSize = 50;              % 方形场地边长
params.initDirection = pi/4;        % 45度初始方向
params.useFixedField = true;        % 启用固定场地模式

% 外源激活参数
params.stabilization_steps = 100;    % 稳定期
params.forced_turn_duration = 200;   % 独立状态持续时间
params.cascade_end_threshold = 5;   % 级联结束判断阈值

% 根据实验模式设置初发个体数
switch experiment_mode
    case 1
        params.external_pulse_count = 1;
        experiment_label = 'c1实验';
    case 2
        params.external_pulse_count = 2;
        experiment_label = 'c2实验';
    case 3
        params.external_pulse_count = 1;  % 先从c1开始
        experiment_label = '对比实验';
    otherwise
        params.external_pulse_count = 1;
        experiment_label = 'c1实验';
end

fprintf('\n当前模式：%s\n', experiment_label);
fprintf('粒子数量: %d\n', params.N);
fprintf('初发个体数: %d\n', params.external_pulse_count);
fprintf('稳定期: %d 步\n', params.stabilization_steps);

%% 4. 创建仿真对象
simulation = ParticleSimulationWithExternalPulse(params);

%% 5. 创建增强图形界面
main_figure = figure('Name', ['级联统计可视化 - ' experiment_label], ...
    'NumberTitle', 'off', ...
    'Position', [50, 50, 1800, 900], ...
    'Color', 'white');

% === 设置6子图布局 ===
% 左侧大图：粒子运动可视化
p1 = subplot('Position', [0.03 0.35 0.35 0.55]);  % 粒子运动

% 右上方：级联统计图表
p2 = subplot('Position', [0.42 0.55 0.25 0.35]);  % 级联规模追踪
p3 = subplot('Position', [0.70 0.55 0.25 0.35]);  % 激活传播速率

% 右下方：系统状态图表
p4 = subplot('Position', [0.42 0.10 0.25 0.35]);  % 阶参数演化
p5 = subplot('Position', [0.70 0.10 0.25 0.35]);  % 激活个体分类

% 底部：级联状态信息面板
p6 = subplot('Position', [0.03 0.05 0.35 0.20]);   % 统计信息文本

%% 6. 初始化左侧粒子运动显示
subplot(p1);
hold on;
ax1 = gca;
ax1.Box = 'on';
ax1.LineWidth = 1.2;
ax1.GridAlpha = 0.15;
ax1.GridLineStyle = ':';

% 设置坐标轴
axis_range = [0 params.fieldSize 0 params.fieldSize];
axis(axis_range);
xlabel('X Position', 'FontWeight', 'bold');
ylabel('Y Position', 'FontWeight', 'bold');
title('粒子状态可视化', 'FontWeight', 'bold', 'FontSize', 12);
grid on;
axis square;

% 创建粒子散点图（使用不同颜色表示状态）
particles_plot = scatter(simulation.positions(:,1), simulation.positions(:,2), ...
    40, [0.5 0.5 0.5], 'filled', 'MarkerFaceAlpha', 0.8);

% 创建状态标记
active_plot = scatter([], [], 60, [1 0.8 0], 'filled');           % 黄色：当前激活
ever_plot = scatter([], [], 45, [0.8 0.2 0.2], 'filled');        % 红色：曾经激活
external_plot = scatter([], [], 80, [0.1 0.8 0.1], 'o', 'LineWidth', 3); % 绿色：外源激活

% 添加图例
legend([particles_plot, active_plot, ever_plot, external_plot], ...
    {'未激活', '当前激活', '曾经激活', '外源激活'}, ...
    'Location', 'northeast', 'FontSize', 9);

% 创建方向箭头（较小尺寸）
u = 0.3 * cos(simulation.theta);
v = 0.3 * sin(simulation.theta);
arrows_plot = quiver(simulation.positions(:,1), simulation.positions(:,2), ...
    u, v, 0, 'Color', [0.3 0.3 0.3 0.5], 'LineWidth', 0.8, 'MaxHeadSize', 0.5);

%% 7. 初始化级联规模追踪图（右上1）
subplot(p2);
hold on;
ax2 = gca;
ax2.Box = 'on';
ax2.LineWidth = 1.0;
ax2.GridAlpha = 0.15;
xlabel('时间步', 'FontWeight', 'bold');
ylabel('级联规模 (比例)', 'FontWeight', 'bold');
title('级联规模实时追踪', 'FontWeight', 'bold', 'FontSize', 11);
grid on;
ylim([0, 1]);
xlim([0, params.T_max]);

% 级联规模线
cascade_size_plot = plot(NaN, NaN, 'Color', [0.8 0.2 0.2], 'LineWidth', 2);
current_active_plot = plot(NaN, NaN, 'Color', [1 0.8 0], 'LineWidth', 1.5, 'LineStyle', '--');

% 添加触发线
xline(params.stabilization_steps + 1, '--r', '触发', 'LineWidth', 1.5);
legend('累计激活(everActivated)', '当前激活(isActive)', 'Location', 'best', 'FontSize', 8);
hold off;

%% 8. 初始化激活传播速率图（右上2）
subplot(p3);
hold on;
ax3 = gca;
ax3.Box = 'on';
ax3.LineWidth = 1.0;
ax3.GridAlpha = 0.15;
xlabel('时间步', 'FontWeight', 'bold');
ylabel('新激活个体数/步', 'FontWeight', 'bold');
title('级联传播速率', 'FontWeight', 'bold', 'FontSize', 11);
grid on;
ylim([0, max(5, params.N/10)]);
xlim([0, params.T_max]);

% 传播速率柱状图
spread_rate_plot = bar(NaN, NaN, 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'none');
xline(params.stabilization_steps + 1, '--r', '触发', 'LineWidth', 1.5);
hold off;

%% 9. 初始化阶参数演化图（右下1）
subplot(p4);
hold on;
ax4 = gca;
ax4.Box = 'on';
ax4.LineWidth = 1.0;
ax4.GridAlpha = 0.15;
xlabel('时间步', 'FontWeight', 'bold');
ylabel('阶参数', 'FontWeight', 'bold');
title('系统序参数演化', 'FontWeight', 'bold', 'FontSize', 11);
grid on;
ylim([0, 1]);
xlim([0, params.T_max]);

order_param_plot = plot(NaN, NaN, 'Color', [0.2 0.4 0.8], 'LineWidth', 2);
xline(params.stabilization_steps + 1, '--r', '触发', 'LineWidth', 1.5);
hold off;

%% 10. 初始化激活个体分类图（右下2）
subplot(p5);
hold on;
ax5 = gca;
ax5.Box = 'on';
ax5.LineWidth = 1.0;
ax5.GridAlpha = 0.15;
xlabel('时间步', 'FontWeight', 'bold');
ylabel('个体数', 'FontWeight', 'bold');
title('激活状态分类统计', 'FontWeight', 'bold', 'FontSize', 11);
grid on;
ylim([0, params.N]);
xlim([0, params.T_max]);

% 堆叠面积图数据初始化
time_data = zeros(1, params.T_max);
inactive_data = zeros(1, params.T_max);
active_data = zeros(1, params.T_max);
ever_active_data = zeros(1, params.T_max);
area_data_count = 0;  % 记录堆叠面积图的实际采样点

% 创建初始面积图：使用重复时间点确保 X 与 Y 维度匹配
initial_time = [0 0];
initial_stack = [params.N 0 0; params.N 0 0];
area_plot = area(initial_time, initial_stack, 'LineWidth', 0.5);
area_plot(1).FaceColor = [0.7 0.7 0.7];  % 灰色：未激活
area_plot(2).FaceColor = [1 0.8 0];      % 黄色：当前激活
area_plot(3).FaceColor = [0.8 0.2 0.2];  % 红色：曾经但非当前激活

xline(params.stabilization_steps + 1, '--r', '触发', 'LineWidth', 1.5);
legend('未激活', '当前激活', '曾激活(非当前)', 'Location', 'best', 'FontSize', 8);
hold off;

%% 11. 初始化统计信息面板（底部）
subplot(p6);
axis off;
info_text = text(0.05, 0.9, '等待仿真开始...', ...
    'FontSize', 11, 'FontWeight', 'bold', 'VerticalAlignment', 'top');

% 创建多行信息显示
status_text = text(0.05, 0.5, '', 'FontSize', 10, 'VerticalAlignment', 'top');
stats_text = text(0.55, 0.5, '', 'FontSize', 10, 'VerticalAlignment', 'top');

%% 12. 数据存储初始化
cascade_sizes = zeros(params.T_max, 1);    % 级联规模历史
current_actives = zeros(params.T_max, 1);  % 当前激活数历史
spread_rates = zeros(params.T_max, 1);     % 传播速率历史
previous_ever_count = 0;                   % 上一步的累计激活数

%% 13. 主仿真循环
tic;
cascade_triggered = false;
cascade_complete = false;
final_c_value = NaN;

for t_step = 1:params.T_max
    % 执行仿真步骤
    simulation.step();

    % 获取当前状态
    positions = simulation.positions;
    theta = simulation.theta;
    is_active = simulation.isActive;
    ever_activated = simulation.everActivated;
    is_external = simulation.isExternallyActivated;

    % 计算统计量
    current_active_count = sum(is_active);
    ever_active_count = sum(ever_activated);
    cascade_size = ever_active_count / params.N;

    % 计算传播速率
    new_activations = ever_active_count - previous_ever_count;
    previous_ever_count = ever_active_count;

    % 存储历史数据
    cascade_sizes(t_step) = cascade_size;
    current_actives(t_step) = current_active_count / params.N;
    spread_rates(t_step) = new_activations;

    % 检测级联触发和结束
    if t_step == params.stabilization_steps + 1
        cascade_triggered = true;
        fprintf('\n步骤 %d: 级联触发！初发个体数 = %d\n', t_step, params.external_pulse_count);
    end

    if cascade_triggered && ~cascade_complete && simulation.isCascadeComplete()
        cascade_complete = true;
        final_c_value = cascade_size;
        fprintf('步骤 %d: 级联结束！最终c值 = %.4f\n', t_step, final_c_value);
    end

    % === 更新可视化（每2步更新一次以提高性能）===
    if mod(t_step, 2) == 0 || t_step == params.T_max || cascade_complete
        % 1. 更新粒子位置和颜色
        subplot(p1);

        % 根据状态设置粒子颜色
        particle_colors = ones(params.N, 3) * 0.5;  % 默认灰色
        particle_sizes = ones(params.N, 1) * 40;

        % 设置颜色：优先级从低到高
        inactive_idx = ~is_active & ~ever_activated;
        particle_colors(inactive_idx, :) = repmat([0.6 0.6 0.6], sum(inactive_idx), 1);

        ever_only_idx = ever_activated & ~is_active;
        particle_colors(ever_only_idx, :) = repmat([0.8 0.2 0.2], sum(ever_only_idx), 1);
        particle_sizes(ever_only_idx) = 45;

        active_idx = is_active;
        particle_colors(active_idx, :) = repmat([1 0.8 0], sum(active_idx), 1);
        particle_sizes(active_idx) = 50;

        % 更新主粒子图
        set(particles_plot, 'XData', positions(:,1), 'YData', positions(:,2), ...
            'CData', particle_colors, 'SizeData', particle_sizes);

        % 更新外源激活标记
        external_idx = find(is_external);
        if ~isempty(external_idx)
            set(external_plot, 'XData', positions(external_idx, 1), ...
                'YData', positions(external_idx, 2));
        else
            set(external_plot, 'XData', [], 'YData', []);
        end

        % 更新箭头
        u = 0.3 * cos(theta);
        v = 0.3 * sin(theta);
        set(arrows_plot, 'XData', positions(:,1), 'YData', positions(:,2), ...
            'UData', u, 'VData', v);

        % 2. 更新级联规模图
        subplot(p2);
        set(cascade_size_plot, 'XData', 1:t_step, 'YData', cascade_sizes(1:t_step));
        set(current_active_plot, 'XData', 1:t_step, 'YData', current_actives(1:t_step));

        % 3. 更新传播速率图
        subplot(p3);
        if t_step > 1
            set(spread_rate_plot, 'XData', 1:t_step, 'YData', spread_rates(1:t_step));
        end

        % 4. 更新阶参数图
        subplot(p4);
        set(order_param_plot, 'XData', 1:t_step, 'YData', simulation.order_parameter(1:t_step));

        % 5. 更新激活分类图
        subplot(p5);
        area_data_count = area_data_count + 1;
        time_data(area_data_count) = t_step;
        inactive_count = sum(~is_active & ~ever_activated);
        active_only_count = sum(is_active & ~ever_activated) + sum(is_active & ever_activated);
        ever_only_count = sum(ever_activated & ~is_active);

        % 累加数据
        inactive_data(area_data_count) = inactive_count;
        active_data(area_data_count) = active_only_count;
        ever_active_data(area_data_count) = ever_only_count;

        % 更新面积图
        if area_data_count > 1
            recent_time = time_data(1:area_data_count);
            recent_stack = [inactive_data(1:area_data_count); active_data(1:area_data_count); ever_active_data(1:area_data_count)];
            delete(area_plot);
            area_plot = area(recent_time, recent_stack', 'LineWidth', 0.5);
            area_plot(1).FaceColor = [0.7 0.7 0.7];
            area_plot(2).FaceColor = [1 0.8 0];
            area_plot(3).FaceColor = [0.8 0.2 0.2];
        end

        % 6. 更新统计信息面板
        subplot(p6);

        % 主状态信息
        if cascade_complete
            main_status = sprintf('级联状态: 已完成 (步骤 %d)', t_step);
        elseif cascade_triggered
            main_status = sprintf('级联状态: 进行中 (步骤 %d)', t_step);
        else
            main_status = sprintf('级联状态: 稳定期 (步骤 %d/%d)', t_step, params.stabilization_steps);
        end

        set(info_text, 'String', sprintf('%s\n实验模式: %s', main_status, experiment_label));

        % 处理级联活跃状态显示
        if simulation.cascade_active
            cascade_status_str = '是';
        else
            cascade_status_str = '否';
        end

        % 详细统计
        status_str = sprintf(['当前统计:\n' ...
                            '  • 累计激活(everActivated): %d/%d (%.2f%%)\n' ...
                            '  • 当前激活(isActive): %d/%d (%.2f%%)\n' ...
                            '  • 外源激活: %d个\n' ...
                            '  • 级联活跃: %s'], ...
                            ever_active_count, params.N, cascade_size*100, ...
                            current_active_count, params.N, current_active_count/params.N*100, ...
                            sum(is_external), cascade_status_str);

        set(status_text, 'String', status_str);

        % c值信息
        if cascade_complete
            c_str = sprintf(['c值计算:\n' ...
                           '  • c%d = %.4f\n' ...
                           '  • 级联持续: %d 步\n' ...
                           '  • 最大传播速率: %d 个/步'], ...
                           params.external_pulse_count, final_c_value, ...
                           t_step - params.stabilization_steps - 1, ...
                           max(spread_rates(1:t_step)));
        else
            c_str = sprintf(['实时c值:\n' ...
                           '  • 当前c%d = %.4f\n' ...
                           '  • 新激活: %d 个\n' ...
                           '  • 阶参数: %.4f'], ...
                           params.external_pulse_count, cascade_size, ...
                           new_activations, ...
                           simulation.order_parameter(t_step));
        end

        set(stats_text, 'String', c_str);

        drawnow limitrate;
    end

    % 级联结束后提前终止
    if cascade_complete && t_step > params.stabilization_steps + 50
        fprintf('级联已稳定，提前结束仿真\n');
        break;
    end
end

simulation_time = toc;

%% 14. 对比实验模式（如果选择了模式3）
if experiment_mode == 3
    fprintf('\n=== 开始c2实验 ===\n');

    % 保存c1结果
    c1_value = final_c_value;

    % 重新初始化为c2实验
    simulation.external_pulse_count = 2;
    simulation.resetCascadeTracking();
    simulation.initializeParticles();
    simulation.current_step = 0;

    % 运行c2实验（简化版，不更新图形）
    fprintf('运行c2实验...\n');
    for step = 1:params.stabilization_steps
        simulation.step();
    end

    % 触发级联并运行到结束
    for step = 1:200
        simulation.step();
        if simulation.isCascadeComplete()
            break;
        end
    end

    c2_value = simulation.getCascadeSize();
    delta_c = c2_value - c1_value;

    % 显示对比结果
    fprintf('\n=== 对比实验结果 ===\n');
    fprintf('c1 (1个初发): %.4f\n', c1_value);
    fprintf('c2 (2个初发): %.4f\n', c2_value);
    fprintf('Δc = c2 - c1 = %.4f\n', delta_c);

    % 在图形上显示对比结果
    subplot(p6);
    comparison_str = sprintf(['\n对比结果:\n' ...
                            'c1 = %.4f, c2 = %.4f\n' ...
                            'Δc = %.4f (敏感性指标)'], ...
                            c1_value, c2_value, delta_c);
    text(0.05, 0.1, comparison_str, 'FontSize', 11, 'FontWeight', 'bold', ...
        'Color', [0.8 0 0], 'VerticalAlignment', 'bottom');
end

%% 15. 最终总结
fprintf('\n=== 仿真完成 ===\n');
fprintf('总用时: %.2f 秒\n', simulation_time);
if ~isnan(final_c_value)
    fprintf('最终c%d值: %.4f\n', params.external_pulse_count, final_c_value);
    fprintf('级联规模: %d/%d 个体被激活\n', sum(simulation.everActivated), params.N);
end

fprintf('\n可视化功能说明：\n');
fprintf('• 左侧大图：粒子状态实时显示（灰=未激活，黄=当前激活，红=曾激活）\n');
fprintf('• 右上1：级联规模追踪（everActivated累计）\n');
fprintf('• 右上2：传播速率（每步新激活数）\n');
fprintf('• 右下1：系统序参数演化\n');
fprintf('• 右下2：激活状态分类统计\n');
fprintf('• 底部：实时统计信息和c值计算\n');

