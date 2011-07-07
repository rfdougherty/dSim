function [D,md,fa,l1,l2,l3] = dSimFitTensor(mrSignal, delta, Delta, dwGrads)
%
% [D,md,fa,l1,l2,l3] = dSimFitTensor(mrSignal, delta, Delta, dwGrads)
% 
% E.g.,
% % load the data:
% dsimOutputFile
% % fit a measurment:
% [D,md,fa,l1,l2,l3] = dSimFitTensor(mrSig{1}, delta{1}, Delta{1}, dwGrads{1})
% ad = l1;
% rd = l2+l3;
%

g = 42576.0; % kHz/T = (cycles/millisecond)/T
G = sqrt(sum(dwGrads.^2));
gradDir = dwGrads./repmat(G,3,1);
gradDir(isnan(gradDir)) = 0; 
bvals = (2.*pi.*g).^2 * (G.*1e-9).^2 .* delta.^2 .* (Delta-delta./3);

dwInds = bvals>0;
% Fit the Stejskal-Tanner equation: S(b) = S(0) exp(-b ADC),
% where S(b) is the image acquired at non-zero b-value, and S(0) is
% the image acquired at b=0. Thus, we can find ADC with the
% following:
%   ADC = -1/b * log( S(b) / S(0)
% But, to avoid divide-by-zero, we need to add a small offset to
% S(0). We also need to add a small offset to avoid log(0).
offset = 1e-6;
logB0 = mean(log(mrSignal(~dwInds)+offset));
logDw = log(mrSignal(dwInds)+offset);
adc = -1./bvals(dwInds).*(logDw-logB0);

% The bvecs are all unit vectors pointed in three space.

% Compute the diffusion tensor D using a least-squares fit.
% See, e.g., http://sirl.stanford.edu/dti/maj/
bv = gradDir(:,dwInds)';
m = [bv(:,1).^2 bv(:,2).^2 bv(:,3).^2 2*bv(:,1).*bv(:,2) 2*bv(:,1).*bv(:,3) 2*bv(:,2).*bv(:,3)];
coef = pinv(m)*adc';
D = [coef(1) coef(4) coef(5); coef(4) coef(2) coef(6); coef(5) coef(6) coef(3)];
[vec,val] = eig(D);
val = diag(val);
md = mean(val);
l1 = val(3);
l2 = val(2);
l3 = val(1);
fa = sqrt(3/2).*(sqrt(sum((val-md).^2,1))./norm(val));
return;

    
