% run_fix_persistence_scaling - 批量调用 fix_persistence_scaling 的示例脚本
%
% 使用步骤：
%   1. 根据实际数据位置调整 target_patterns（支持通配符）。
%   2. dryRunOnly=true 时先预览，不会写回文件；确认无误后改为 false 再运行。
%   3. 如需关闭备份，将 doBackup 设为 false。
%
% 备注：脚本会自动把项目 src 目录加入路径，便于直接调用工具函数。

clear; clc;

%% 1) 待处理文件通配符（按需修改）
% 当前针对 persistence_scan 结果目录 20251105_221002
target_patterns = { ...
    fullfile('data', 'experiments', 'persistence_scan', '20251105_221002', '*.mat') ...
};

%% 2) 运行选项
dryRunOnly = false;        % 先预览，不写盘；确认后改为 false
doBackup = true;          % 是否在写盘前生成 .pre_pfix 备份
backupSuffix = '.pre_pfix';
verboseLog = true;        % 是否输出详细日志

%% 3) 路径配置（上溯两级定位项目根）
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(script_dir));
addpath(genpath(fullfile(project_root, 'src')));

%% 4) 调用修正函数
fix_persistence_scaling(target_patterns, ...
    'DryRun', dryRunOnly, ...
    'Backup', doBackup, ...
    'BackupSuffix', backupSuffix, ...
    'Verbose', verboseLog);

%% 5) 提示
if dryRunOnly
    fprintf('当前为 DryRun 预览模式，如需写盘请将 dryRunOnly 置为 false 后重新运行。\n');
end
