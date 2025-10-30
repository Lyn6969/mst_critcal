% plot_cascade_trajectories.m - c1/c2 轨迹对比图生成脚本
%
% 功能说明：
%   针对 1 个与 2 个随机惊吓个体的实验，记录整个仿真周期的移动轨迹，
%   并在最终位置上分别突出显示级联统计使用到的个体与惊吓个体。
%   图片输出至项目 pic 目录，格式与 formalpic 目录其他脚本保持一致。

clear; clc; close all;

%% -------------------- 可调参数 --------------------
CJ_THRESHOLD = 0.6;              % 运动显著性阈值（可根据需要调整）
RNG_SEED_C1 = 2025;              % c1 实验随机数种子
RNG_SEED_C2 = 3025;              % c2 实验随机数种子
EXTRA_STEPS_AFTER_CASCADE = 20;  % 级联完成后继续仿真的步数（保证位置稳定）

%% -------------------- 图像与配色配置 --------------------
FIG_WIDTH_TRAJ = 500;
FIG_HEIGHT_TRAJ = 500;
FIG_WIDTH_CASCADE = 1000;
FIG_HEIGHT_CASCADE = 200;
AXIS_LINE_WIDTH = 1.5;
TICK_FONT_SIZE = 12;
LABEL_FONT_SIZE = 13;
FONT_NAME = 'Arial';
LABEL_FONT_WEIGHT = 'Bold';
TICK_FONT_WEIGHT = 'Bold';
LEGEND_FONT_SIZE = 12;

TRAJ_LINE_WIDTH = 1.0;
SHOCK_LINE_WIDTH = 1.8;

COLOR_ORANGE = [1.0 0.55 0.0];
COLOR_RED = [0.85 0.1 0.1];
COLOR_GREY = [0.65 0.65 0.65];
CASCADE_C1_COLOR = [0.18 0.36 0.78];
CASCADE_C2_COLOR = CASCADE_C1_COLOR;

AGENT_SCALE = 1.0;          % 个体绘制尺寸缩放因子
HEAD_RADIUS = 0.25;
TAIL_LENGTH = 0.45;
TAIL_WIDTH = 0.12;
HEAD_RESOLUTION = 40;
BASE_FACE_ALPHA = 0.6;
HIGHLIGHT_FACE_ALPHA = 0.9;
CASCADE_LINE_WIDTH = 2.0;

BLUE_GRADIENT_START = [0.85 0.92 1.0];
BLUE_GRADIENT_END = [0.00 0.10 0.45];
RED_GRADIENT_START = [1.00 0.86 0.86];
RED_GRADIENT_END = [0.55 0.00 0.00];
BLUE_TRAJ_ALPHA = 0.2;     % 普通个体轨迹透明度
RED_TRAJ_ALPHA = 0.6;      % 信息个体轨迹透明度

%% -------------------- 仿真公共参数 --------------------
params.N = 200;
params.rho = 1;
params.v0 = 1;
params.angleUpdateParameter = 10;
params.angleNoiseIntensity = 0;
params.T_max = 400;
params.dt = 0.1;
params.cj_threshold = CJ_THRESHOLD;
params.radius = 5;
params.deac_threshold = 0.1745;
params.fieldSize = 40;
params.initDirection = pi/4;
params.useFixedField = true;
params.stabilization_steps = 100;
params.forced_turn_duration = 200;
params.cascade_end_threshold = 5;

%% -------------------- 路径准备 --------------------
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(script_dir));   % mst_critcal 所在目录
pic_dir = fullfile(project_root, 'pic');
if ~exist(pic_dir, 'dir')
    mkdir(pic_dir);
end

%% -------------------- 运行仿真（c1 与 c2） --------------------
rng(RNG_SEED_C1);
result_c1 = simulate_cascade(params, 1, EXTRA_STEPS_AFTER_CASCADE);

rng(RNG_SEED_C2);
result_c2 = simulate_cascade(params, 2, EXTRA_STEPS_AFTER_CASCADE);

%% -------------------- 绘制与导出 --------------------
fig1 = figure('Position', [100, 100, FIG_WIDTH_TRAJ, FIG_HEIGHT_TRAJ], 'Color', 'white');
ax1 = axes('Parent', fig1);
plot_trajectories(ax1, result_c1, '', ...
    COLOR_GREY, COLOR_ORANGE, COLOR_RED, ...
    TRAJ_LINE_WIDTH, SHOCK_LINE_WIDTH, ...
    HEAD_RADIUS * AGENT_SCALE, TAIL_LENGTH * AGENT_SCALE, TAIL_WIDTH * AGENT_SCALE, HEAD_RESOLUTION, ...
    BASE_FACE_ALPHA, HIGHLIGHT_FACE_ALPHA, ...
    BLUE_GRADIENT_START, BLUE_GRADIENT_END, RED_GRADIENT_START, RED_GRADIENT_END, ...
    BLUE_TRAJ_ALPHA, RED_TRAJ_ALPHA, ...
    AXIS_LINE_WIDTH, FONT_NAME, TICK_FONT_SIZE, LABEL_FONT_SIZE, ...
    TICK_FONT_WEIGHT, LABEL_FONT_WEIGHT);

fig2 = figure('Position', [150, 150, FIG_WIDTH_TRAJ, FIG_HEIGHT_TRAJ], 'Color', 'white');
ax2 = axes('Parent', fig2);
plot_trajectories(ax2, result_c2, '', ...
    COLOR_GREY, COLOR_ORANGE, COLOR_RED, ...
    TRAJ_LINE_WIDTH, SHOCK_LINE_WIDTH, ...
    HEAD_RADIUS * AGENT_SCALE, TAIL_LENGTH * AGENT_SCALE, TAIL_WIDTH * AGENT_SCALE, HEAD_RESOLUTION, ...
    BASE_FACE_ALPHA, HIGHLIGHT_FACE_ALPHA, ...
    BLUE_GRADIENT_START, BLUE_GRADIENT_END, RED_GRADIENT_START, RED_GRADIENT_END, ...
    BLUE_TRAJ_ALPHA, RED_TRAJ_ALPHA, ...
    AXIS_LINE_WIDTH, FONT_NAME, TICK_FONT_SIZE, LABEL_FONT_SIZE, ...
    TICK_FONT_WEIGHT, LABEL_FONT_WEIGHT);

fig3 = figure('Position', [200, 200, FIG_WIDTH_CASCADE, FIG_HEIGHT_CASCADE], 'Color', 'white');
ax3 = axes('Parent', fig3);
hold(ax3, 'on');

time_steps_c1 = result_c1.time_steps;
cascade_c1 = result_c1.cascade_history;
time_steps_c2 = result_c2.time_steps;
cascade_c2 = result_c2.cascade_history;

plot(ax3, time_steps_c1, cascade_c1, 'LineWidth', CASCADE_LINE_WIDTH, ...
    'Color', CASCADE_C1_COLOR, 'DisplayName', 'c_1');
plot(ax3, time_steps_c2, cascade_c2, 'LineWidth', CASCADE_LINE_WIDTH, ...
    'Color', CASCADE_C2_COLOR, 'LineStyle', '--', 'DisplayName', 'c_2');

if isfield(result_c1, 'trigger_step') && ~isempty(result_c1.trigger_step)
    xline(ax3, result_c1.trigger_step, '--', 'Color', [0.95 0.75 0.1], 'LineWidth', 2.4, 'HandleVisibility', 'off');
end

ax3.LineWidth = AXIS_LINE_WIDTH;
ax3.FontName = FONT_NAME;
ax3.FontSize = TICK_FONT_SIZE;
ax3.FontWeight = TICK_FONT_WEIGHT;
ax3.TickDir = 'in';
grid(ax3, 'off');
box(ax3, 'on');

xlabel(ax3, 'Time', 'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);
ylabel(ax3, 'Cascade Size', 'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);
legend(ax3, 'Location', 'northwest', 'FontSize', LEGEND_FONT_SIZE, 'Interpreter', 'tex');

max_time = max([time_steps_c1(end), time_steps_c2(end)]);
if max_time <= 0
    max_time = 1;
end
xlim(ax3, [0, max_time]);

max_cascade = max([cascade_c1; cascade_c2]);
upper_limit = min(1, max_cascade * 1.1 + 1e-3);
if upper_limit <= 0.05
    upper_limit = 0.05;
end
ylim(ax3, [0, upper_limit]);

hold(ax3, 'off');

% 输出路径
c1_pdf = fullfile(pic_dir, 'cascade_trajectory_c1.pdf');
c2_pdf = fullfile(pic_dir, 'cascade_trajectory_c2.pdf');
cascade_pdf = fullfile(pic_dir, 'cascade_size_timeseries.pdf');

exportgraphics(fig1, c1_pdf, 'ContentType', 'vector');
exportgraphics(fig2, c2_pdf, 'ContentType', 'vector');
exportgraphics(fig3, cascade_pdf, 'ContentType', 'vector');

fprintf('c1 trajectory figure saved to: %s\n', c1_pdf);
fprintf('c2 trajectory figure saved to: %s\n', c2_pdf);
fprintf('Cascade-size timeline saved to: %s\n', cascade_pdf);

%% -------------------- 辅助函数定义 --------------------
function result = simulate_cascade(params, external_count, extra_steps_after_cascade)
% simulate_cascade - 针对给定初发个体数量运行仿真并记录轨迹
%
% 输入参数：
%   params - 仿真参数结构体
%   external_count - 初发个体数量（1 或 2）
%   extra_steps_after_cascade - 级联完成后继续运行的步数
%
% 输出结果：
%   result.positions_history - [step x N x 2] 的轨迹数据
%   result.final_positions   - 最终时刻的粒子位置
%   result.cascade_mask      - everActivated 布尔向量
%   result.external_indices  - 惊吓个体的索引
%   result.field_size        - 场地边长

    params.external_pulse_count = external_count;
    simulation = ParticleSimulationWithExternalPulse(params);

    % 初始位置整体平移至 (10,10) 附近
    offset = [10, 10];
    simulation.positions = bsxfun(@plus, simulation.positions, offset);
    simulation.positions = min(max(simulation.positions, 0), params.fieldSize);
    simulation.previousPositions = simulation.positions;

    max_steps = params.T_max;
    num_agents = params.N;

    positions_history = zeros(max_steps + 1, num_agents, 2);
    positions_history(1, :, :) = simulation.positions;
    theta_history = zeros(max_steps + 1, num_agents);
    theta_history(1, :) = simulation.theta.';
    cascade_history = zeros(max_steps + 1, 1);
    cascade_history(1) = sum(simulation.everActivated) / simulation.N;

    external_indices = [];
    step_counter = 0;
    cascade_complete_logged = false;

    for step = 1:max_steps
        simulation.step();
        step_counter = step_counter + 1;
        positions_history(step_counter + 1, :, :) = simulation.positions;
        theta_history(step_counter + 1, :) = simulation.theta.';
        cascade_history(step_counter + 1) = sum(simulation.everActivated) / simulation.N;

        if simulation.external_pulse_triggered && isempty(external_indices)
            external_indices = find(simulation.external_activation_timer > 0 | simulation.isExternallyActivated);
            if isempty(external_indices)
                external_indices = find(simulation.everActivated);
            end
        end

        if simulation.isCascadeComplete()
            cascade_complete_logged = true;
        end

        if cascade_complete_logged && step > params.stabilization_steps + 50
            for extra = 1:extra_steps_after_cascade
                if step_counter >= max_steps
                    break;
                end
                simulation.step();
                step_counter = step_counter + 1;
                positions_history(step_counter + 1, :, :) = simulation.positions;
                theta_history(step_counter + 1, :) = simulation.theta.';
                cascade_history(step_counter + 1) = sum(simulation.everActivated) / simulation.N;
            end
            break;
        end
    end

    positions_history = positions_history(1:step_counter + 1, :, :);
    theta_history = theta_history(1:step_counter + 1, :);
    cascade_history = cascade_history(1:step_counter + 1, :);

    result.positions_history = positions_history;
    result.final_positions = squeeze(positions_history(end, :, :));
    result.theta_history = theta_history;
    result.cascade_history = cascade_history;
    result.time_vector = (0:step_counter)' * params.dt;
    result.time_steps = (0:step_counter)';
    result.trigger_step = params.stabilization_steps + 1;
    result.cascade_mask = simulation.everActivated;
    result.external_indices = external_indices(:)';
    result.field_size = params.fieldSize;
    prev_positions = squeeze(positions_history(end - 1, :, :));
    delta_move = result.final_positions - prev_positions;
    move_norm = vecnorm(delta_move, 2, 2);
    movement_angles = atan2(delta_move(:, 2), delta_move(:, 1));
    zero_motion_mask = move_norm < 1e-6;
    if any(zero_motion_mask)
        fallback_angles = theta_history(end, zero_motion_mask).';
        movement_angles(zero_motion_mask) = fallback_angles;
    end
    result.final_theta = movement_angles;
end

function plot_trajectories(ax, result, title_text, ...
    color_base, color_orange, color_red, ...
    traj_line_width, shock_line_width, ...
    head_radius, tail_length, tail_width, head_resolution, ...
    base_face_alpha, highlight_face_alpha, ...
    blue_grad_start, blue_grad_end, red_grad_start, red_grad_end, ...
    blue_traj_alpha, red_traj_alpha, ...
    axis_line_width, font_name, tick_font_size, label_font_size, ...
    tick_font_weight, label_font_weight)
% plot_trajectories - 绘制轨迹与最终位置标注

    hold(ax, 'on');
    num_agents = size(result.positions_history, 2);
    final_theta = result.final_theta;

    % 先绘制普通个体轨迹（蓝色渐变）
    normal_indices = setdiff(1:num_agents, result.external_indices);
    for idx = normal_indices
        x_traj = squeeze(result.positions_history(:, idx, 1));
        y_traj = squeeze(result.positions_history(:, idx, 2));

        if any(isnan(x_traj)) || any(isnan(y_traj))
            continue;
        end

        draw_gradient_line(ax, x_traj, y_traj, blue_grad_start, blue_grad_end, traj_line_width, blue_traj_alpha);
    end

    final_pos = result.final_positions;
    % 基础形状：普通个体（灰色）
    for idx = 1:num_agents
        draw_agent_shape(ax, final_pos(idx, :), final_theta(idx), color_base, ...
            head_radius, tail_length, tail_width, head_resolution, base_face_alpha);
    end

    % 级联个体（橙色，覆盖灰色）
    cascade_idx = find(result.cascade_mask);
    for idx = cascade_idx(:)'
        draw_agent_shape(ax, final_pos(idx, :), final_theta(idx), color_orange, ...
            head_radius, tail_length, tail_width, head_resolution, highlight_face_alpha);
    end

    % 惊吓个体轨迹（红色渐变），位于最上层
    for idx = result.external_indices
        x_traj = squeeze(result.positions_history(:, idx, 1));
        y_traj = squeeze(result.positions_history(:, idx, 2));
        if any(isnan(x_traj)) || any(isnan(y_traj))
            continue;
        end
        draw_gradient_line(ax, x_traj, y_traj, red_grad_start, red_grad_end, shock_line_width, red_traj_alpha);
    end

    % 惊吓个体（红色，最上层）
    shock_idx = result.external_indices;
    for idx = shock_idx
        draw_agent_shape(ax, final_pos(idx, :), final_theta(idx), color_red, ...
            head_radius, tail_length, tail_width, head_resolution, 1.0);
    end

    axis(ax, [0 result.field_size 0 result.field_size]);
    axis(ax, 'square');
    box(ax, 'on');
    ax.LineWidth = axis_line_width;
    ax.FontName = font_name;
    ax.FontSize = tick_font_size;
    ax.FontWeight = tick_font_weight;
    ax.TickDir = 'in';

    xlabel(ax, 'x', 'FontName', font_name, 'FontSize', label_font_size, 'FontWeight', label_font_weight);
    ylabel(ax, 'y', 'FontName', font_name, 'FontSize', label_font_size, 'FontWeight', label_font_weight);
    if ~isempty(title_text)
        title(ax, title_text, 'FontName', font_name, 'FontSize', label_font_size, 'FontWeight', label_font_weight);
    else
        title(ax, '');
    end
end

function draw_agent_shape(ax, position, heading, color, ...
    head_radius, tail_length, tail_width, head_resolution, face_alpha)
% draw_agent_shape - 依据角度绘制粒子形状（尾巴 + 头部）

    if any(isnan(position)) || isnan(heading)
        return;
    end

    theta = heading;
    tail_corners = [
        0, -tail_width/2;
        tail_length, -tail_width/2;
        tail_length, tail_width/2;
        0, tail_width/2
        ];

    R = [cos(theta), -sin(theta); sin(theta), cos(theta)];
    rotated_tail = (R * tail_corners.').' + position;

    patch('Parent', ax, ...
        'XData', rotated_tail(:, 1), ...
        'YData', rotated_tail(:, 2), ...
        'FaceColor', color, ...
        'EdgeColor', color, ...
        'FaceAlpha', face_alpha, ...
        'EdgeAlpha', face_alpha);

    t = linspace(0, 2*pi, head_resolution);
    head_points = [head_radius * cos(t); head_radius * sin(t)];
    rotated_head = (R * head_points).' + position;

    patch('Parent', ax, ...
        'XData', rotated_head(:, 1), ...
        'YData', rotated_head(:, 2), ...
        'FaceColor', color, ...
        'EdgeColor', color, ...
        'FaceAlpha', face_alpha, ...
        'EdgeAlpha', face_alpha);
end

function draw_gradient_line(ax, x, y, color_start, color_end, line_width, edge_alpha)
% draw_gradient_line - 使用颜色渐变绘制轨迹线

    if numel(x) < 2 || numel(y) < 2
        return;
    end

    x = x(:);
    y = y(:);

    if any(isnan(x)) || any(isnan(y))
        return;
    end

    if nargin < 7 || isempty(edge_alpha)
        edge_alpha = 1.0;
    end
    edge_alpha = min(max(edge_alpha, 0), 1);

    num_points = numel(x);
    t = linspace(0, 1, num_points - 1).';
    colors = (1 - t) .* color_start + t .* color_end;

    for idx = 1:(num_points - 1)
        patch('Parent', ax, ...
            'XData', [x(idx), x(idx + 1)], ...
            'YData', [y(idx), y(idx + 1)], ...
            'ZData', [0, 0], ...
            'FaceColor', 'none', ...
            'EdgeColor', colors(idx, :), ...
            'LineWidth', line_width, ...
            'EdgeAlpha', edge_alpha);
    end
end
