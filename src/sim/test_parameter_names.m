% 快速测试脚本：验证参数名称是否正确
% 只测试单次实验，确保所有参数名称正确

clc;
clear;
close all;

fprintf('=== 参数名称验证测试 ===\n');

% 设置参数（与run_delta_c_parameter_scan.m相同）
params.N = 200;
params.rho = 1;
params.v0 = 1;
params.angleUpdateParameter = 10;
params.angleNoiseIntensity = 0;
params.T_max = 400;
params.dt = 0.1;
params.radius = 5;
params.deac_threshold = 0.1745;
params.cj_threshold = 1.0;
params.fieldSize = 50;
params.initDirection = pi/4;
params.useFixedField = true;
params.stabilization_steps = 200;
params.forced_turn_duration = 200;

try
    % 创建仿真对象
    fprintf('创建仿真对象...\n');
    sim = ParticleSimulationWithExternalPulse(params);
    fprintf('✓ 仿真对象创建成功！\n');

    % 设置外源激活个体数
    sim.external_pulse_count = 1;

    % 运行单次实验
    fprintf('运行单次c1实验...\n');
    cascade_size = sim.runSingleExperiment(1);
    fprintf('✓ c1实验成功！级联规模：%.4f\n', cascade_size);

    % 重置并运行c2实验
    sim.resetCascadeTracking();
    sim.external_pulse_count = 2;
    fprintf('运行单次c2实验...\n');
    cascade_size = sim.runSingleExperiment(2);
    fprintf('✓ c2实验成功！级联规模：%.4f\n', cascade_size);

    fprintf('\n所有参数名称验证通过！可以安全运行完整实验。\n');

catch ME
    fprintf('\n✗ 错误：%s\n', ME.message);
    fprintf('错误位置：%s (第%d行)\n', ME.stack(1).name, ME.stack(1).line);
    fprintf('\n请检查参数名称是否正确。\n');
    rethrow(ME);
end

fprintf('\n=== 测试完成 ===\n');