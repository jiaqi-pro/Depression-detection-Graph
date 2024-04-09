% Song, Siyang, Shashank Jaiswal, Linlin Shen, and Michel Valstar
% Spectral Representation of Behaviour Primitives for Depression Analysis.
% IEEE Transactions on Affective Computing (2020)
% Email: siyang.song@nottingham.ac.uk

function out_data = preprocess( raw_data )

[num, length] = size(raw_data);

out_data = zeros(num, length);

for i = 1:num
    
    out_data(i,:) = raw_data(i,:) - median(raw_data(i,:));
     
end



end

