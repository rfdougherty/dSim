% Generate q-space sampling grid

g = 42576.0; % kHz/T = (cycles/millisecond)/T

g = 2*pi*g;

readout = 1.0; % readout time (in ms, currently has no effect on simulation)

% q = g * delta * G
% tau = (Delta-delta./3);
% b = g^2 * delta^2 * tau;
% 

% Values from Hagmann et. al. 2008:
delta = 32.5; % msec (23.1 w/ 40mT = b=1.0), 33.6 for bvals up to 3
Delta = 43.5; % msec
Gmax = 80.0; % mT/m * 1e-3/1e+6 = 1e-9
Bmax = 9.0;

% Values from Van Wedeen:
delta = 60; % msec (23.1 w/ 40mT = b=1.0), 33.6 for bvals up to 3
Delta = 66; % msec
Gmax = 40.0; % mT/m * 1e-3/1e+6 = 1e-9
Bmax = 17.0;
Qmax = 1/10.0; % in micrometers^-1
Qmin = 1/50.0; % in micrometers^-1
nTotal = 512; % van Wedeen does 512

% sample q on a uniform grid 
n = ceil(nTotal.^(1/3));
v = linspace(Qmin,Qmax,n);
[qx,qy,qz] = ndgrid(v,v,v);

Q = [qx(:), qy(:), qz(:)];

fid = fopen('sim_qspace.grads','w');
%fid = 1;
for(ii=1:size(Q,1))
    G = Q(ii,:)./(g*delta).*1e9;
    fprintf(fid,'%0.2f %0.2f %0.2f %0.1f %0.1f %0.1f\n\n',delta, Delta, readout, G);
end
fprintf(fid,'1 0 0 0\n');
if(fid~=1), fclose(fid); end
   
% to analyze data:
addpath data; 
m=0; qspace_vanwedeen
s=reshape(mrSig{1},[8 8 8]);
dsi = abs(fftshift(fftn(s)));
showMontage(dsi);
figure; isosurface(dsi,1); axis equal off;

m=0; qspace_crossing
s=reshape(mrSig{1},[8 8 8]);
dsi = abs(fftshift(fftn(s)));
showMontage(dsi);
figure; isosurface(dsi,1); axis equal off;

%bval = g.^2 * (G.*1e-9).^2 .* delta.^2 .* (Delta-delta./3)

%bval = [0.0:0.2:3.0];
%Delta = bval/g.^2 * (G*1e-9).^2 * delta.^2) + delta/3;
% G from b:
%G = sqrt(bval./(g.^2 * delta.^2 .* (Delta-delta./3)))*1e9;




