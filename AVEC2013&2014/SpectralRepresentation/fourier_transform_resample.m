% Song, Siyang, Shashank Jaiswal, Linlin Shen, and Michel Valstar
% Spectral Representation of Behaviour Primitives for Depression Analysis.
% IEEE Transactions on Affective Computing (2020)
% Email: siyang.song@nottingham.ac.uk
% input:
%--all_data: Multi-channel time-series facial behaviour primitives data
%--num_fre: sampling frequency
%--N: final used TOP-N frequencies
% output:
%--amp_map_return: amplitude spectrum map
%--phase_map_return: phase spectrum map

function [amp_map_return, phase_map_return] = fourier_transform_resample(all_data, N, num_fre)

[channel_num, length] = size(all_data);
amp_map = zeros(channel_num,num_fre);
phase_map = zeros(channel_num,num_fre);


for i = 1:channel_num
    
    temp_contain = fft(all_data(i,:));
    
    if mod(length,2) == 0
        
        temp_contain = temp_contain(:,1:length/2+1);
        
    else
        
        temp_contain = temp_contain(:,1:(length+1)/2);
        
    end
    
    temp_resample_data = resample(temp_contain,num_fre,size(temp_contain,2));
    amp_map(i,:) = abs(temp_resample_data)/length;
    phase_map(i,:) = angle(temp_resample_data);
    
end

amp_map_return = amp_map(:,1:N);
phase_map_return = phase_map(:,1:N);


end

