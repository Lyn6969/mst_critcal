% 扩散系数与持久性交互式演示脚本
% =========================================================================
% 功能简介：
%   - 以单次粒子群仿真为基础，实时展示相对质心的均方位移 MSD、扩散系数 D_r
%     与持久性指标 P = 1 / sqrt(D_r) 的估计过程。
%   - 结合 run_persistence_scan 中的扩散系数拟合逻辑与
%     visualize_branching_ratio 的交互式可视化风格，帮助研究者直观理解
%     参数变化对群体空间扩散与持久性的影响。
%
% 使用方式：
%   - 运行脚本后按提示输入运动显著性阈值与角度噪声强度。
%   - 图形界面左侧展示粒子空间布置，右侧依次展示：
%       1. 相对质心均方位移 (MSD) 与线性拟合曲线
%       2. 相对质心扩散系数 D_r 随时间演化
%       3. 持久性指标 P 随时间演化
%   - 底部信息面板实时给出统计窗口、拟合点数与当前估计值。
%
% 依赖：
%   - ParticleSimulation.m (需已在 MATLAB 路径中)
%
% 作者：李亚男
% 日期：2025年
% MATLAB 2025a 兼容
% =========================================================================

clc;
clear;
close all;

%% 1. 全局图形设置 ---------------------------------------------------------
set(0, 'DefaultAxesFontSize', 10);
set(0, 'DefaultAxesFontName', 'Arial');
set(0, 'DefaultTextFontSize', 10);
set(0, 'DefaultTextFontName', 'Arial');
set(0, 'DefaultLineLineWidth', 1.4);

%% 2. 交互输入与参数准备 ---------------------------------------------------
fprintf('=== 扩散系数与持久性交互式演示 ===\n');

cj_threshold = input('请输入运动显著性阈值 (默认 2.0): ');
if isempty(cj_threshold)
    cj_threshold = 0.1;
end

angle_noise = input('请输入角度噪声强度 (默认 0.2): ');
if isempty(angle_noise)
    angle_noise = 0.01;
end

fprintf('\n实验参数：\n');
fprintf('  • cj_threshold = %.2f\n', cj_threshold);
fprintf('  • angleNoiseIntensity = %.2f\n', angle_noise);

% 仿真基础参数，沿用 run_persistence_scan 的默认配置
params = struct();
params.N = 200;
params.rho = 1;
params.v0 = 1;
params.angleUpdateParameter = 10;
params.angleNoiseIntensity = angle_noise;
params.T_max = 600;
params.dt = 0.1;
params.radius = 5;
params.deac_threshold = 0.1745;
params.cj_threshold = cj_threshold;
params.fieldSize = 50;
params.initDirection = pi/4;
params.useFixedField = true;

% 扩散系数拟合相关参数
burn_in_ratio = 0.2;          % 丢弃初始瞬态比例
min_fit_points = 30;          % 线性拟合所需最少数据点

% 创建仿真对象
simulation = ParticleSimulation(params);

T = simulation.T_max;
dt = simulation.dt;
time_vec = (0:T)' * dt;

% 初始化统计数组
initial_positions = simulation.positions;   % 保存初始位置以计算 MSD
initial_centroid = mean(initial_positions, 1);
prev_positions = initial_positions;
prev_centroid = initial_centroid;
initial_offsets = initial_positions - initial_centroid;  % 记录初始相对位置
msd_history = zeros(T + 1, 1);
D_history = NaN(T + 1, 1);
P_history = NaN(T + 1, 1);
step_disp_history = zeros(T + 1, 1);
msd_history(1) = 0;
step_disp_history(1) = 0;
centroid_history = zeros(T + 1, 2);
centroid_history(1, :) = initial_centroid;

%% 3. 图形界面搭建 ---------------------------------------------------------
fig = figure('Name', sprintf('扩散与持久性演示 (cj=%.2f, noise=%.2f)', ...
    cj_threshold, angle_noise), 'NumberTitle', 'off', 'Color', 'white', ...
    'Position', [100, 60, 1550, 900]);

ax_particles = subplot('Position', [0.05 0.36 0.38 0.6]);
ax_info = subplot('Position', [0.05 0.08 0.38 0.22]);
ax_msd = subplot('Position', [0.48 0.66 0.47 0.28]);
ax_D = subplot('Position', [0.48 0.37 0.47 0.22]);
ax_P = subplot('Position', [0.48 0.08 0.47 0.22]);

% 粒子散点与方向箭头
subplot(ax_particles);
hold on;
axis([0 params.fieldSize 0 params.fieldSize]);
axis square;
grid on;
xlabel('X');
ylabel('Y');
title('粒子空间分布');
particle_colors = repmat([0.6 0.6 0.6], params.N, 1);
particle_sizes = 40 * ones(params.N, 1);
particles_plot = scatter(simulation.positions(:,1), simulation.positions(:,2), ...
    particle_sizes, particle_colors, 'filled', 'MarkerFaceAlpha', 0.8);
arrows_plot = quiver(simulation.positions(:,1), simulation.positions(:,2), ...
    0.35*cos(simulation.theta), 0.35*sin(simulation.theta), 0, ...
    'Color', [0.3 0.3 0.3 0.6], 'LineWidth', 0.9, 'MaxHeadSize', 0.45);
hold off;

% 信息文本区域
subplot(ax_info);
axis off;
info_text = text(0.02, 0.95, '初始化中...', 'FontSize', 11, ...
    'FontWeight', 'bold', 'VerticalAlignment', 'top');
detail_text = text(0.02, 0.62, '', 'FontSize', 10, 'VerticalAlignment', 'top');
fit_text = text(0.52, 0.62, '', 'FontSize', 10, 'VerticalAlignment', 'top');
status_text = text(0.02, 0.28, '', 'FontSize', 10, 'VerticalAlignment', 'top');

% MSD 曲线
subplot(ax_msd);
hold on;
xlabel('时间 t');
ylabel('⟨Δr^2⟩');
title('平均位置均方位移与线性拟合');
msd_plot = plot(NaN, NaN, 'Color', [0.1 0.4 0.8], 'LineWidth', 1.8, ...
    'DisplayName', 'MSD');
fit_plot = plot(NaN, NaN, '--', 'Color', [0.85 0.2 0.2], 'LineWidth', 1.6, ...
    'DisplayName', '线性拟合');
legend('Location', 'northwest');
hold off;

% D_r 曲线
subplot(ax_D);
hold on;
xlabel('时间步');
ylabel('D_r');
title('位置扩散系数估计');
D_plot = plot(NaN, NaN, '-', 'Color', [0.2 0.6 0.4], 'LineWidth', 1.6);
ylim([0, 1]);
hold off;

% 持久性曲线
subplot(ax_P);
hold on;
xlabel('时间步');
ylabel('P = 1/\surd D_r');
title('持久性指标估计');
P_plot = plot(NaN, NaN, '-', 'Color', [0.55 0.35 0.8], 'LineWidth', 1.6);
ylim([0, 5]);
hold off;

% 初始信息展示
set(info_text, 'String', sprintf(['时间步: 0 / %d\n' ...
    '位置扩散系数 D_r: 未定义\n' ...
    '持久性 P: 未定义'], T));

%% 4. 主循环：实时仿真与统计 ------------------------------------------------
for t_step = 1:T
    if ~ishandle(fig)
        fprintf('检测到窗口关闭，提前结束仿真。\n');
        break;
    end

    prev_positions = simulation.positions;
    prev_centroid = centroid_history(t_step, :);
    simulation.step();
    centroid = mean(simulation.positions, 1);
    centroid_history(t_step + 1, :) = centroid;

    % 更新相对质心的均方位移
    centered_pos = simulation.positions - centroid;
    rel_disp = centered_pos - initial_offsets;
    squared_disp = sum(rel_disp.^2, 2);
    msd_history(t_step + 1) = mean(squared_disp, 'omitnan');

    % 记录一步位移（质心移动与个体相对位移）
    centroid_shift = norm(centroid - prev_centroid);
    step_motion = vecnorm(simulation.positions - prev_positions, 2, 2);
    step_disp_history(t_step + 1) = mean(step_motion, 'omitnan');

    % 线性拟合估计 D_r
    burn_in_index = max(2, floor((t_step + 1) * burn_in_ratio));
    fit_indices = burn_in_index:(t_step + 1);
    D_current = NaN;
    P_current = NaN;
    coeffs = [];

    if numel(fit_indices) >= 2 && numel(fit_indices) >= min_fit_points
        x = time_vec(fit_indices);
        y = msd_history(fit_indices);
        x_shift = x - x(1);
        y_shift = y - y(1);
        if any(x_shift > 0) && any(abs(y_shift) > eps)
            smooth_window = max(5, floor(numel(y_shift) * 0.1));
            if smooth_window > 1
                y_shift = smoothdata(y_shift, 'movmean', smooth_window);
            end
            slope = lsqnonneg(x_shift(:), y_shift(:));
            intercept = y(1) - slope * x(1);
            coeffs = [slope, intercept];
            if slope > 0
                D_current = slope;
                if slope <= 1e-6
                    P_current = Inf;
                else
                    P_current = 1 / sqrt(D_current);
                end
            else
                D_current = 0;
                P_current = Inf;
            end
        end
    end

    D_history(t_step + 1) = D_current;
    P_history(t_step + 1) = P_current;

    %% 可视化更新（每 2 步刷新一次以平衡性能）
    if mod(t_step, 2) == 0 || t_step == T
        % 更新粒子状态
        subplot(ax_particles);
        colors = repmat([0.6 0.6 0.6], params.N, 1);
        sizes = 40 * ones(params.N, 1);
        set(particles_plot, 'XData', simulation.positions(:,1), ...
            'YData', simulation.positions(:,2), 'CData', colors, ...
            'SizeData', sizes);
        set(arrows_plot, 'XData', simulation.positions(:,1), ...
            'YData', simulation.positions(:,2), ...
            'UData', 0.35*cos(simulation.theta), ...
            'VData', 0.35*sin(simulation.theta));

        % 更新 MSD 曲线
        subplot(ax_msd);
        set(msd_plot, 'XData', time_vec(1:t_step + 1), ...
            'YData', msd_history(1:t_step + 1));
        if ~isempty(coeffs)
            fit_x = time_vec(fit_indices);
            fit_y = polyval(coeffs, fit_x);
            set(fit_plot, 'XData', fit_x, 'YData', fit_y);
        else
            set(fit_plot, 'XData', NaN, 'YData', NaN);
        end
        max_msd = max(msd_history(1:t_step + 1), [], 'omitnan');
        if isempty(max_msd) || isnan(max_msd)
            max_msd = 1;
        end
        ylim([0, max(1e-3, max_msd * 1.1)]);

        % 更新 D_r 曲线
        subplot(ax_D);
        set(D_plot, 'XData', 0:t_step, 'YData', D_history(1:t_step + 1));
        max_D = max(D_history(1:t_step + 1), [], 'omitnan');
        if isempty(max_D) || isnan(max_D)
            max_D = 1e-2;
        end
        ylim([0, max(1e-3, max_D * 1.2)]);

        % 更新 P 曲线
        subplot(ax_P);
        set(P_plot, 'XData', 0:t_step, 'YData', P_history(1:t_step + 1));
        finite_P = P_history(1:t_step + 1);
        finite_P = finite_P(isfinite(finite_P));
        max_P = max(finite_P, [], 'omitnan');
        if isempty(max_P) || isnan(max_P)
            max_P = 1;
        end
        ylim([0, max(1, max_P * 1.2)]);

        % 更新文本信息
        subplot(ax_info);
        D_display = '未定义';
        P_display = '未定义';
        if ~isnan(D_current)
            D_display = sprintf('%.4f', D_current);
            if D_current == 0
                D_display = '≈0';
            end
        end
        if ~isnan(P_current)
            if isinf(P_current)
                P_display = '∞';
            else
                P_display = sprintf('%.4f', P_current);
            end
        end

        base_lines = {
            sprintf('时间步: %d / %d', t_step, T);
            sprintf('相对质心扩散系数 D_r: %s', D_display);
            sprintf('持久性 P: %s', P_display)
        };
        set(info_text, 'String', strjoin(base_lines, '\n'));

        detail_lines = {
            sprintf('时间分辨率 dt: %.3f', dt);
            sprintf('拟合起点索引: %d', burn_in_index);
            sprintf('拟合样本数: %d', numel(fit_indices));
            sprintf('平均相对位移 (上一时间步): %.3f', step_disp_history(t_step + 1))
        };
        set(detail_text, 'String', strjoin(detail_lines, '\n'));

        if ~isempty(coeffs)
            fit_lines = {
                sprintf('非负最小二乘: slope = %.4e', coeffs(1));
                sprintf('当前 MSD(t): %.4f', msd_history(t_step + 1))
            };
        else
            fit_lines = {'线性拟合暂不可用', ''};
            set(fit_plot, 'XData', NaN, 'YData', NaN);
        end
        set(fit_text, 'String', strjoin(fit_lines, '\n'));

        status_lines = {
            sprintf('粒子数: %d', params.N);
            sprintf('噪声强度: %.2f', params.angleNoiseIntensity);
            sprintf('运动显著性阈值: %.2f', params.cj_threshold)
        };
        set(status_text, 'String', strjoin(status_lines, '\n'));

        drawnow limitrate;
    end
end

%% 5. 结果输出 -------------------------------------------------------------
valid_indices = ~isnan(D_history);
if any(valid_indices)
    last_idx = find(valid_indices, 1, 'last');
else
    last_idx = numel(time_vec);
end
results = struct();
results.description = 'Single-run diffusion and persistence visualization';
results.params = params;
results.burn_in_ratio = burn_in_ratio;
results.min_fit_points = min_fit_points;
results.time = time_vec(1:last_idx);
results.initial_positions = initial_positions;
results.initial_centroid = initial_centroid;
results.centroid_history = centroid_history(1:last_idx, :);
results.step_disp = step_disp_history(1:last_idx);
results.msd = msd_history(1:last_idx);
results.D_history = D_history(1:last_idx);
results.P_history = P_history(1:last_idx);

assignin('base', 'diffusion_persistence_results', results);

fprintf('\n仿真结束。结果已写入工作区变量 diffusion_persistence_results。\n');
