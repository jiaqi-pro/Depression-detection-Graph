% Song, Siyang, Shashank Jaiswal, Linlin Shen, and Michel Valstar
% Spectral Representation of Behaviour Primitives for Depression Analysis.
% IEEE Transactions on Affective Computing (2020)
% Email: siyang.song@nottingham.ac.uk
function [raw_data,num_keep_frame,raw_num_keep_frame] = cut_videos(raw_data,fre_resolution)

len = size(raw_data,1);
raw_num_keep_frame = floor(len/fre_resolution);
num_keep_frame = raw_num_keep_frame*fre_resolution;
num_delete = len - num_keep_frame;

num_delete_start = ceil(num_delete/2);
num_delete_end = floor(num_delete/2);

raw_data([1:num_delete_start,end-num_delete_end+1:end],:)=[];

end

