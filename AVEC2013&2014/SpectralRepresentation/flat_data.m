function [ flat_data ] = flat_data( data, t_length, N)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

flat_data = zeros(1,t_length);

row_num = size(data,1);

for i = 1:row_num
    
    flat_data(1,(i-1)*N+1:i*N) = data(i,:);
    
end


end

