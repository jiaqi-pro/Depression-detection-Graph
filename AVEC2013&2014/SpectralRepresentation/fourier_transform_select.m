% Song, Siyang, Shashank Jaiswal, Linlin Shen, and Michel Valstar
% Spectral Representation of Behaviour Primitives for Depression Analysis.
% IEEE Transactions on Affective Computing (2020)
% Email: siyang.song@nottingham.ac.uk
function [ amp_map_return, phase_map_return ] = fourier_transform_select( all_data, N, num_multiple,fre_resolution)


[channel_num, length] = size(all_data);
%temp_contain = zeros(channel_num,length);
amp_map = zeros(channel_num,length);
phase_map = zeros(channel_num,length);
common_temp_amp = zeros(channel_num,fre_resolution);
common_temp_pha = zeros(channel_num,fre_resolution);


for i = 1:channel_num
    
    temp_contain = fft(all_data(i,:));
    amp_map(i,:) = abs(temp_contain/length);
    phase_map(i,:) = angle(temp_contain);
    
end


for j = 1:fre_resolution
    
    common_temp_amp(:,j) = amp_map(:,1+(j-1)*num_multiple);
    common_temp_pha(:,j) = phase_map(:,1+(j-1)*num_multiple);    
   
end


amp_map_return = [amp_map(:,length/2+1),common_temp_amp(:,1:N-1)];
phase_map_return = [phase_map(:,length/2+1),common_temp_pha(:,1:N-1)];



end

