% mylin radius fraction
mrf = 0.8;
% axons spacing (in micrometers)
s = [0:0.05:1];
r = [0.2:0.1:5];

[rg,sg] = meshgrid(r,s);

% myelin area fraction for a single axon
ma = (pi.*rg.^2 - pi.*(mrf.*rg).^2)./(pi.*rg.^2);

% myelin volume fraction is myelin area times the packing density.
% Since we assume eveyrything is constant in the 3rd dimension,
% this gives us the myelin volume fraction.
% number of axons that fit into a 2r+s*2r+s square. For hex-packing,
% remember that in one dimension, the circles are spaced on-center 
% at 2*r, but in the other dim, on-center spacing is 2.*r * sqrt(3)./2.
% Thus, we have 2*r+s * 2.*r+s * sqrt(3)./2 = (2.*r+s).^2.*sqrt(3)/2
len = 2.*rg+sg;
%n = (len ./ (2.*r+s)) .*  (len ./ ((2.*r+s) .* sqrt(3)./2))
n = len.^2 ./ (len.^2 .* sqrt(3)./2);
% density of a single circle in a square:
a = pi.*rg.^2 ./ len.^2;
% density of a hex-packed array:
packDensity = n.*a;
% max density is hex-pack- 1/6*pi*sqrt(3) (or pi/sqrt(12));
%packDensity = pi./sqrt(12);
mvf = ma.*packDensity;

%plot(r,mvf(2,:));
% surf(rg,sg,mvf);
% colormap(autumn);
% xlabel('axon radius (\mum)');
% ylabel('axon spacing (\mum)');
% zlabel('mylelin fraction');


% Do a surface plot of the diffusivity simulation results

addpath /home/bob/svn/dSim/data
addpath /home/bob/svn/dSim/analysis
sizeSpaceSim
for(ii=1:m)
   [D,md(ii),fa(ii),l1(ii),l2(ii),l3(ii)] = dSimFitTensor(mrSig{ii}, delta{ii}, Delta{ii}, dwGrads{ii});
end

axSpace = unique(radiusSpace);
axRad = unique(radiusMean);

[rg,sg] = meshgrid(axRad,axSpace);
adg = zeros(size(rg));
rdg = zeros(size(rg));
for(ii=1:numel(axSpace))
    tmp = radiusSpace==axSpace(ii);
    adg(ii,:) = l1(tmp);
    rdg(ii,:) = (l2(tmp)+l3(tmp))./2;
end
figure(43);
surf(rg,sg,rdg);
colormap(autumn);
xlabel('axon radius (\mum)');
ylabel('axon spacing (\mum)');
zlabel('radial diffusivity (\mum^2/msec)');
figure(42);
[c,h] = contour(rg,sg,rdg); clabel(c,h); colorbar;
xlabel('axon radius (\mum)');
ylabel('axon spacing (\mum)');

ma = (pi.*rg.^2 - pi.*(mrf.*rg).^2)./(pi.*rg.^2);
len = 2.*rg+sg;
n = len.^2 ./ (len.^2 .* sqrt(3)./2);
a = pi.*rg.^2 ./ len.^2;
packDensity = n.*a;
mvf = ma.*packDensity;

figure(44)
surf(rg,sg,mvf);
colormap(winter);
xlabel('axon radius (\mum)');
ylabel('axon spacing (\mum)');
zlabel('mylelin fraction');


mrf = 0.8;
% Some sources put mrf at 0.6
r = [0.1:0.1:5];
bpf=0.18;
mvf = bpf.*1.5;
s = (sqrt(2*sqrt(3)).*r.*sqrt((pi-pi.*mrf.^2).*mvf)-2*sqrt(3).*r.*mvf)./(sqrt(3).*mvf);
%figure(3);plot(r,s);

figure(3);
s = [0:0.01:1];
r = (sqrt(2*sqrt(3)).*s.*sqrt(pi.*mvf)-2*sqrt(3).*s.*mvf)./(4*sqrt(3).*mvf+2*pi.*mrf.^2-2*pi);
if(all(r<=0)), r=-r; end
plot(s,r);
ylabel('axon radius (\mum)');
xlabel('axon spacing (\mum)');
grid on;

figure(42);
rdVals = [0.1:0.1:0.8];
[c,h] = contour(rg,sg,rdg,rdVals);
xlabel('axon radius (\mum)');
ylabel('axon spacing (\mum)');
h = clabel(c,h);
grid on;
s = axSpace;
mrf = 0.8;
%bpf = [0.16:0.005:0.22];
bpf = [0.1600 0.1700 0.1800 0.1900 0.1950 0.2000 0.2025 0.2050 0.2075 0.2100 0.2125 0.2200]
mvf = bpf.*1.5;
p = round(numel(s)/2);
for(ii=1:numel(mvf))
    r = (sqrt(2.*sqrt(3)).*s.*sqrt(pi.*mvf(ii))-2.*sqrt(3).*s.*mvf(ii))./(4.*sqrt(3).*mvf(ii)+2.*pi.*mrf.^2-2.*pi);
    if(all(r<=0)), r=-r; end
    hold on; plot(r,s,'k-.'); hold off;
    %max(r)
    %text(r(p),s(p),sprintf('%0.3f',bpf(ii)));
end



