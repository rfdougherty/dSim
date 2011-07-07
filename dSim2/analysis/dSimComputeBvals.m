
g = 42576.0; % kHz/T = (cycles/millisecond)/T
delta = 18.0; % msec (23.1 w/ 40mT = b=1.0), 33.6 for bvals up to 3
Delta = delta+1; % msec
maxG = 36.0; % mT/m * 1e-3/1e+6 = 1e-9
% For an optimized DTI sequence, we can get an effective G of srt(2)*maxG:
G = sqrt(2)*maxG;
readout = 1.0; % readout time (in ms, currently has no effect on simulation)

%bval = (2.*pi.*g).^2 * (G.*1e-9).^2 .* delta.^2 .* (Delta-delta./3)

%bval = [0.0:0.2:3.0];
bval = .8; % msec/micrometer^2 (8000 sec/mm^2 / 1000)
Delta = bval/((2*pi*g).^2 * (G*1e-9).^2 * delta.^2) + delta/3

% On our GE Signa, we typically run at 118ms / slice for a 0.8 bvalue. 
% Assuming delta=18, Delta=19 -> DWGrad time = 37
% Thus, there is ~81msec of overhead for RF, imaging grads, read-out, etc.
sliceTime = Delta+delta+81

%G = sqrt(bval./((2*pi*g).^2 * delta.^2 * (Delta-delta/3))) * 1e9

dirs = [0.0 0.0 0.0
1.0 1.0 0.0
-1.0 -1.0 0.0
0.0 1.0 1.0
0.0 -1.0 -1.0
1.0 0.0 1.0
-1.0 0.0 -1.0
-1.0 1.0 0.0
1.0 -1.0 0.0
0.0 -1.0 1.0
0.0 1.0 -1.0
1.0 0.0 -1.0
-1.0 0.0 1.0];

val = Delta;
%fid = fopen('sim_bval_x.grads','w');
fid = 1;
for(ii=1:length(val))
   for(jj=1:size(dirs,1))
      tDelta = Delta(min(ii,numel(Delta)));
      tdelta = delta(min(ii,numel(delta)));
      tG = G(min(ii,numel(G)));
      curDir = dirs(jj,:);
      if(norm(curDir)~=0), curDir = curDir./norm(curDir); end
      gd = curDir*tG;
      if(tDelta<=tdelta), error('Delta <= delta!'); end
      fprintf(fid,'%0.2f %0.2f %0.2f %0.1f %0.1f %0.1f\n\n',tdelta, tDelta, readout, gd);
   end
end
fprintf(fid,'1 0 0 0\n');
if(fid~=1), fclose(fid); end
   
bval = [0.0:0.2:3.0];
%adc = -1./bval.*log(mrSig);
x = linspace(min(bval),max(bval),100);

% Some calculations based on simulated data
realAdc = [1.0, 1.5, 2.0, 3.0];
mrSig(1,:)=[ 1 0.82379276 0.67760694 0.55308419 0.45511615 0.36737433 0.30707473 0.24743466 0.19704604 0.15964846 0.14241035 0.11226655 0.094162464 0.077935413 0.052232079 0.051570382 ];
mrSig(2,:)=[ 1 0.74140388 0.556804 0.41690576 0.30791491 0.23128697 0.16611511 0.12058327 0.097443894 0.064243935 0.0473058 0.046698023 0.018840995 0.019442644 0.017009614 0.0080275508 ];
mrSig(3,:)=[ 1 0.6730634 0.45978948 0.30851313 0.20255643 0.14032035 0.098095991 0.058910791 0.042505149 0.025675239 0.015245737 0.0013691077 0.0036870737 0.0046120863 0.0019113389 0.0059886952 ];
%mrSig(4,:)=[ 1 0.55820304 0.31377298 0.17507426 0.10885275 0.050084911 0.032552209 0.013371097 0.0061536254 0.0060313847 0.0038876233 0.0041224631 0.0038909595 0.0061003547 0.0080938321 0.0071779415 ];
mrSig(4,:)=[ 1 0.5701856 0.32396975 0.17697942 0.10140479 0.062190071 0.031057538 0.012961613 0.011603279 0.00512328 0.0071840067 0.0040644575 0.0023037274 0.0067513171 0.0044212546 0.006316951 ]
for(ii=1:numel(realAdc)), y(ii,:) = exp(-x.*realAdc(ii)); end
figure(1); 
plot(bval,mrSig(1,:),'ko',bval,mrSig(2,:),'kx',bval,mrSig(3,:),'k^',bval,mrSig(4,:),'ks');
hold on;plot(x,y,'k--'); hold off;
set(gca,'fontSize',14);
xlabel('b-value (msec/\mum^2)'); ylabel('MR Signal (arb)');
for(ii=1:numel(realAdc)),l{ii}=sprintf('%0.1f \\mum^2/msec',realAdc(ii)); end
set(gca,'fontSize',10); legend(l);
mrUtilResizeFigure(gcf,400,300);
mrUtilPrintFigure('figs/fig1',gcf,300)

% Simulations with fibers (1um diameter, 1.1um on-center grid)
clear mrSig realAdc l y;
mrSig(1,:)=[ 1 0.67873657 0.46713457 0.31064206 0.21449305 0.141403 0.09570501 0.065886401 0.038594779 0.032231063 0.01311334 0.0079717273 0.010546354 0.0046422514 0.0076927114 0.0061849123 ];
mrSig(2,:))=[ 1 0.97262692 0.94850171 0.92418569 0.90471697 0.88710332 0.86843604 0.85677266 0.84454256 0.83369595 0.82847869 0.81246316 0.80277097 0.79331726 0.78814322 0.78258944 ]
adc = -1./repmat(bval,size(mrSig,1),1).*log(mrSig);
realAdc = mean(adc(:,2:end),2);
for(ii=1:numel(realAdc)), y(ii,:) = exp(-x.*realAdc(ii)); end
plot(bval,mrSig(1,:),'ko',bval,mrSig(2,:),'kx');
hold on;plot(x,y,'k--'); hold off;
xlabel('b-value (msec/\mum^2)'); ylabel('MR Signal (arb))');
for(ii=1:numel(realAdc)),l{ii}=sprintf('ADC %0.2f \\mum^2/msec',realAdc(ii)); end
legend(l)

% Simulations with fibers (1um diameter, 1.2um on-center grid)
mrSig=[ 1 0.93457294 0.88164335 0.83606726 0.80038756 0.76906937 0.74079782 0.72028148 0.70129269 0.69062424 0.67288637 0.66582614 0.64951563 0.65035224 0.63961411 0.64147276 ]

% Simulations with fibers (1um diameter, 1.3um on-center grid)
mrSig=[ 1 0.90944231 0.82919765 0.77158868 0.72165376 0.68143135 0.64855933 0.62619835 0.61091018 0.59356439 0.5775333 0.56620991 0.55234373 0.55220199 0.55066979 0.53503668 ]

% Simulations with fibers (1um diameter, 1.4um on-center grid)
mrSig=[ 1 0.88377118 0.79188955 0.71979553 0.65530503 0.62337607 0.58032221 0.55105519 0.53000563 0.51270384 0.49333444 0.48974898 0.48356023 0.47993496 0.46290013 0.46304598 ]


% Simulations with fibers (2um diameter, 2.1um on-center grid)
clear mrSig realAdc l y;
mrSig(1,:)=[ 1 0.67866373 0.46834278 0.30782083 0.20631631 0.14006171 0.093297414 0.067509927 0.045984779 0.027269214 0.012291844 0.014782659 0.0071455115 0.0090901218 0.008396579 0.0048593995 ];
mrSig(2,:)=[ 1 0.98876506 0.97869891 0.96785474 0.95993835 0.95115232 0.94347805 0.93648589 0.92869538 0.92098635 0.9150598 0.91117024 0.90583104 0.90093362 0.89488113 0.89229631 ];
adc = -1./repmat(bval,size(mrSig,1),1).*log(mrSig);
realAdc = mean(adc(:,2:end),2);
for(ii=1:numel(realAdc)), y(ii,:) = exp(-x.*realAdc(ii)); end
plot(bval,mrSig(1,:),'ko',bval,mrSig(2,:),'kx');
hold on;plot(x,y,'k--'); hold off;
xlabel('b-value (msec/\mum^2)'); ylabel('MR Signal (arb))');
for(ii=1:numel(realAdc)),l{ii}=sprintf('ADC %0.2f \\mum^2/msec',realAdc(ii)); end
legend(l)

% Simulations with fibers (3um diameter, 3.1um on-center grid)
mrSig=[ 1 0.99155891 0.9836697 0.97760314 0.969679 0.96191585 0.95849085 0.95321274 0.94640124 0.94118464 0.93680501 0.93516409 0.9283545 0.92490613 0.91994464 0.9168303 ];

% Simulations with fibers (6um diameter, 6.1um on-center grid)
mrSig=[ 1 0.99409217 0.98866224 0.98228866 0.97923887 0.97493273 0.96994823 0.96503317 0.96154088 0.95802838 0.95340276 0.95115834 0.94942153 0.94495338 0.94273192 0.93971008 ];
% with timestep=0.001
mrSig=[ 1 0.99109727 0.98413968 0.97518826 0.96797055 0.96423537 0.95791584 0.95300454 0.94789809 0.94327378 0.93741006 0.93456572 0.93286604 0.92404109 0.92324579 0.9260323 ];
% with timestep=0.001, 9um on-center grid
mrSig=[ 1 0.8559317 0.74689126 0.66506582 0.59790432 0.55317241 0.52161264 0.49710912 0.47320107 0.44821715 0.44631034 0.43240112 0.42757514 0.41897941 0.42197251 0.41185614 ]

% Simulations with fibers (12um diameter, 12.1um on-center grid)
mrSig=[ 1 0.99548084 0.99087799 0.98727417 0.98300147 0.98052305 0.97575104 0.97556114 0.97058356 0.96961766 0.96585917 0.96325469 0.96124744 0.95904845 0.95705986 0.95730251 ];
