load('saliency_scan_20251028_231609.mat');
max_P = max(P_mean, [], 'omitnan');
P_mean_norm = P_mean / max_P;
P_std_norm = P_std / max_P;
save('saliency_scan_20251028_231609.mat', 'P_mean_norm', 'P_std_norm', 'max_P', '-append');
%%
%[text] 画一下响应性和噪声的图
%%

% === 可视化最近一次响应性噪声×阈值扫描结果 ===
% 自动定位 data/experiments/responsiveness_noise_cj_scan 中最新时间戳目录，
% 载入 results.mat 并绘制 M_T–η 热力图，仅显示，不写入文件。

project_root = detect_project_root();
data_dir = fullfile(project_root, 'data', 'experiments', 'responsiveness_noise_cj_scan');

if ~isfolder(data_dir)
    error('未找到响应性扫描数据目录: %s', data_dir);
end

dir_info = dir(data_dir);
dir_info = dir_info([dir_info.isdir]);
dir_names = setdiff({dir_info.name}, {'.', '..'});
if isempty(dir_names)
    error('目录 %s 中没有任何响应性扫描结果。', data_dir);
end

timestamps = datetime(dir_names, 'InputFormat', 'yyyyMMdd_HHmmss');
[~, newest_idx] = max(timestamps);
latest_dir = fullfile(data_dir, dir_names{newest_idx});
mat_file = fullfile(latest_dir, 'results.mat');

if ~isfile(mat_file)
    error('在 %s 中找不到 results.mat，请确认扫描是否成功保存。', latest_dir);
end

load(mat_file, 'results');

figure('Color', 'white', 'Position', [160, 100, 900, 540]);
imagesc(results.cj_thresholds, results.eta_levels, results.R_mean);
axis xy;
xlabel('M_T');
ylabel('\eta');
title('Responsiveness R Mean');
cb = colorbar;
cb.Label.String = 'R';
grid on;

%%






function project_root = detect_project_root()
candidates = {};

script_path = mfilename('fullpath');
if ~isempty(script_path)
candidates{end+1} = fileparts(script_path); %#ok<AGROW>
end

try
active_file = matlab.desktop.editor.getActiveFilename;
if ~isempty(active_file)
candidates{end+1} = fileparts(active_file); %#ok<AGROW>
end
catch

end

try
current_folder = pwd;
if ~isempty(current_folder)
candidates{end+1} = current_folder; %#ok<AGROW>
end
catch

end

anchor_script = which('run_responsiveness_noise_cj_scan');
if ~isempty(anchor_script)
candidates{end+1} = fileparts(anchor_script); %#ok<AGROW>
end

candidates = unique(candidates);

project_root = '';
for i = 1:numel(candidates)
root_candidate = search_up_for_project(candidates{i});
if ~isempty(root_candidate)
project_root = root_candidate;
break;
end
end

if isempty(project_root)
error('无法定位项目根目录，请确认已将 mst_critcal 项目加入路径。');
end
end

function root = search_up_for_project(start_dir)
root = '';
if isempty(start_dir)
return;
end

current_dir = start_dir;
max_levels = 15;
for level = 1:max_levels
[parent_dir, name, ext] = fileparts(current_dir);
dir_name = [name ext];

is_target = strcmpi(dir_name, 'mst_critcal');
has_marker = isfile(fullfile(current_dir, 'AGENTS.md')) && ...
isfolder(fullfile(current_dir, 'data')) && ...
isfolder(fullfile(current_dir, 'src'));

if is_target && has_marker
root = current_dir;
return;
end

if isempty(parent_dir) || strcmp(parent_dir, current_dir)
return;
end
current_dir = parent_dir;
end
end

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline","rightPanelPercent":29.3}
%---
