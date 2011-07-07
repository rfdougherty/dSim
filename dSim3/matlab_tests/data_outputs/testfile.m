
% * * * Fri Jul  9 16:33:37 2010
if (exist('m','var')), m=m+1; else, m=1; end
s=0;

%%%%%%%%%%%%%%%%% 
% Input parameters 
%%%%%%%%%%%%%%%%%
numSpins(m) = 5000;
useGpu(m) = 1;
fibersFile{m} = 'sim.triangles';
useDisplay(m) = 1;
stepsPerUpdate(m) = 10
gyroMagneticRatio(m) = 42576
timeStep(m) = 0.005
extraAdc(m) = 2.1
intraAdc(m) = 2.1
myelinAdc(m) = 0.1
spaceScale(m) = 20
permeability(m) = 1e-06


%%%%%%%%%%%%%%%%% 
% Simulation data 
%%%%%%%%%%%%%%%%%
s=s+1;
delta{m}(s) = 10;
Delta{m}(s) = 12.0669;
readOut{m}(s) = 1;
dwGrads{m}(:,s) = [0,0,0];
mrSig{m}(s) = 0.749661;
s=s+1;
delta{m}(s) = 10;
Delta{m}(s) = 12.0669;
readOut{m}(s) = 1;
dwGrads{m}(:,s) = [80,80,0];
mrSig{m}(s) = 0.212253;
s=s+1;
delta{m}(s) = 10;
Delta{m}(s) = 12.0669;
readOut{m}(s) = 1;
dwGrads{m}(:,s) = [-80,-80,0];
mrSig{m}(s) = 0.206954;
s=s+1;
delta{m}(s) = 10;
Delta{m}(s) = 12.0669;
readOut{m}(s) = 1;
dwGrads{m}(:,s) = [0,80,80];
mrSig{m}(s) = 0.202545;
s=s+1;
delta{m}(s) = 10;
Delta{m}(s) = 12.0669;
readOut{m}(s) = 1;
dwGrads{m}(:,s) = [0,-80,-80];
mrSig{m}(s) = 0.191305;
s=s+1;
delta{m}(s) = 10;
Delta{m}(s) = 12.0669;
readOut{m}(s) = 1;
dwGrads{m}(:,s) = [80,0,80];
mrSig{m}(s) = 0.204571;
s=s+1;
delta{m}(s) = 10;
Delta{m}(s) = 12.0669;
readOut{m}(s) = 1;
dwGrads{m}(:,s) = [-80,0,-80];
mrSig{m}(s) = 0.207616;
s=s+1;
delta{m}(s) = 10;
Delta{m}(s) = 12.0669;
readOut{m}(s) = 1;
dwGrads{m}(:,s) = [-80,80,0];
mrSig{m}(s) = 0.21401;
s=s+1;
delta{m}(s) = 10;
Delta{m}(s) = 12.0669;
readOut{m}(s) = 1;
dwGrads{m}(:,s) = [80,-80,0];
mrSig{m}(s) = 0.221752;
s=s+1;
delta{m}(s) = 10;
Delta{m}(s) = 12.0669;
readOut{m}(s) = 1;
dwGrads{m}(:,s) = [0,-80,80];
mrSig{m}(s) = 0.215402;
s=s+1;
delta{m}(s) = 10;
Delta{m}(s) = 12.0669;
readOut{m}(s) = 1;
dwGrads{m}(:,s) = [0,80,-80];
mrSig{m}(s) = 0.217902;
s=s+1;
delta{m}(s) = 10;
Delta{m}(s) = 12.0669;
readOut{m}(s) = 1;
dwGrads{m}(:,s) = [80,0,-80];
mrSig{m}(s) = 0.21096;
s=s+1;
delta{m}(s) = 10;
Delta{m}(s) = 12.0669;
readOut{m}(s) = 1;
dwGrads{m}(:,s) = [-80,0,80];
mrSig{m}(s) = 0.211865;
