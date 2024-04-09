% Song, Siyang, Shashank Jaiswal, Linlin Shen, and Michel Valstar
% Spectral Representation of Behaviour Primitives for Depression Analysis.
% IEEE Transactions on Affective Computing (2020)
% Email: siyang.song@nottingham.ac.uk

function videoFea = getVideoFeature( feaVec )

feaNum = size(feaVec,1);
videoFea = zeros(1,feaNum*12-6*6);
for i = 1:feaNum
    %if i <= feaNum-6
        videoFea(1,(i-1)*12 + 1) = mean(feaVec(i,:));
        videoFea(1,(i-1)*12 + 2) = std(feaVec(i,:));
        videoFea(1,(i-1)*12 + 3) = max(feaVec(i,:));
        videoFea(1,(i-1)*12 + 4) = min(feaVec(i,:));
        videoFea(1,(i-1)*12 + 5) = mean(diff(feaVec(i,:)));
        videoFea(1,(i-1)*12 + 6) = std(diff(feaVec(i,:)));
        videoFea(1,(i-1)*12 + 7) = max(diff(feaVec(i,:)));
        videoFea(1,(i-1)*12 + 8) = min(diff(feaVec(i,:)));
        a = feaVec(i,1:end-2);
        b = feaVec(i,3:end);
        videoFea(1,(i-1)*12 + 9) = mean(a-b);
        videoFea(1,(i-1)*12 + 10) = std(a-b);
        videoFea(1,(i-1)*12 + 11) = max(a-b);
        videoFea(1,(i-1)*12 + 12) = min(a-b);
%     else
%         videoFea(1,(feaNum-6)*12+ (i-feaNum+5)*6 + 1) = mean(feaVec(i,:));
%         videoFea(1,(feaNum-6)*12+ (i-feaNum+5)*6 + 2) = std(feaVec(i,:));
%         videoFea(1,(feaNum-6)*12+ (i-feaNum+5)*6 + 3) = mean(diff(feaVec(i,:)));
%         videoFea(1,(feaNum-6)*12+ (i-feaNum+5)*6 + 4) = std(diff(feaVec(i,:)));
%         a = feaVec(i,1:end-2);
%         b = feaVec(i,3:end);
%         videoFea(1,(feaNum-6)*12+ (i-feaNum+5)*6 + 5) = mean(a-b);
%         videoFea(1,(feaNum-6)*12+ (i-feaNum+5)*6 + 6) = std(a-b);
    %end
end

end