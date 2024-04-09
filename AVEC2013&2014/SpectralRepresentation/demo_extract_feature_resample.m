% Song, Siyang, Shashank Jaiswal, Linlin Shen, and Michel Valstar
% Spectral Representation of Behaviour Primitives for Depression Analysis.
% IEEE Transactions on Affective Computing (2020)
% Email: siyang.song@nottingham.ac.uk

clear all;clc

%% setting

Primitive_num = 29; % the number of behaviour signals

N = 80; % Choosing TOP-N frequency (N < fre_resolution/2)

fre_resolution = 256; % sampling frequency 

file_name = 'example_data.mat';


%% pre_processing

t_length = N*Primitive_num; % set feature length for amp/phase map

raw_data = load(file_name); % load data

raw_data = raw_data.example_data;

raw_data = raw_data';

processed_data = preprocess(raw_data); % substracting median values, ypu can customized your own preprocess method here

processed_data = processed_data';

%% feature extraction

sta_fea = getVideoFeature(processed_data); % compute statistics features

[amp_map, phase_map] = fourier_transform_resample(processed_data, N,fre_resolution);% 2-D amplitude map and phase map generation

amp_flat_data = flat_data(amp_map,t_length, N);% 1-D amplitude feature generation

phase_flat_data = flat_data(phase_map,t_length, N);% 1-D phase feature generation




















