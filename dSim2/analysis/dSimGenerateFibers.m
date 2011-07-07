function dSimGenerateFibers(outFile, alpha, beta, density, spaceScale, maxIter)
%
% dSimGenerateFibers([outFile='sim.fibers'], alpha, beta, [density=.1], [spaceScale=100], [maxIter=5000])
%
% density is in fibers/um
% spaceScale is the voxel size in um
%

if(~exist('outFile','var')||isempty(outFile))
    outFile = 'sim.fibers';
end
if(exist(outFile,'file'))
    [f,p] = uiputfile('*.fibers', 'Save fibers...', outFile);
    if(isnumeric(f)), disp('Canceled.'); return; end
    outFile = fullfile(p,f);
end
if(~exist('density','var')||isempty(density))
    density = 0.1;
end
if(~exist('spaceScale','var')||isempty(spaceScale))
    spaceScale = 100;
end
if(~exist('pad','var')||isempty(pad))
    pad = 0.05;
end
if(~exist('maxIter','var')||isempty(maxIter))
    maxIter = 5000;
end

% Double the space scale because voxels go from -1 to 1.
spaceScale = spaceScale*2;
minRadius = 0.1;
maxRadius = 5.0;

nFibers = spaceScale^2*density;
% lay down the fibers on a uniform grid
fibers = rand(2,nFibers)*spaceScale-spaceScale/2;

radii = getRadii(fibers, maxRadius);

figure(87); subplot(1,2,1); plot(fibers(1,:),fibers(2,:),'.'); axis equal; hold on;
line([fibers(1,:)-radii;fibers(1,:)+radii],[fibers(2,:); fibers(2,:)]);
line([fibers(1,:); fibers(1,:)],[fibers(2,:)-radii;fibers(2,:)+radii]);
hold off;

% WORK HERE:
% We could grow the radii if we jiggle the positions somewhat to get a
% denser packing.

g = gamfit(radii);
err = abs(g - [alpha beta])./[alpha beta];
tol = [0.01 0.01];

step = .5;
iter = 0;

x = linspace(0.125,4.875,20);
n = hist(radii,x);
targetPdf = gampdf(x,alpha,beta);
targetPdf = targetPdf./max(targetPdf);
y = gampdf(x, g(1), g(2));
figure(87); subplot(1,2,2); plot(x,n,'.',x,y./max(y).*max(n),'-');
title(sprintf('alpha = %0.2f, beta = %0.2f',g(1),g(2)));
while(any(err>tol) && iter<maxIter)
    tmpFibers = fibers + rand(2,nFibers)*step*2-step;
    radii = getRadii(tmpFibers, maxRadius);
    radii(radii>maxRadius) = maxRadius;
    g = gamfit(radii);
    tmpErr = abs(g - [alpha beta])./[alpha beta];
    if(sum(tmpErr)<sum(err))
        err = tmpErr;
        fibers = tmpFibers;
        n = hist(radii,x);
        y = gampdf(x, g(1), g(2));
        figure(87); clf;
        subplot(1,2,1); plot(fibers(1,:),fibers(2,:),'.'); axis equal; hold on;
        line([fibers(1,:)-radii;fibers(1,:)+radii],[fibers(2,:); fibers(2,:)]);
        line([fibers(1,:); fibers(1,:)],[fibers(2,:)-radii;fibers(2,:)+radii]);
        hold off;
        subplot(1,2,2); 
        plot(x, n, 'm.', x, y./max(y).*max(n), 'r-', x, targetPdf.*max(n),'k-');
        title(sprintf('iter = %d, alpha = %0.2f, beta = %0.2f',iter,g(1),g(2)));
        refresh(87);
    end
    iter = iter+1;
end

tooSmall = radii<minRadius;
fibers = fibers(:,~tooSmall);
radii = radii(~tooSmall);

figure(87); subplot(1,2,1); plot(fibers(1,:),fibers(2,:),'.'); axis equal; hold on;
line([fibers(1,:)-radii;fibers(1,:)+radii],[fibers(2,:); fibers(2,:)]);
line([fibers(1,:); fibers(1,:)],[fibers(2,:)-radii;fibers(2,:)+radii]);


% Save fibers
fid = fopen(outFile,'w');
for(ii=1:numel(radii))
    fprintf(fid,'%0.2f %0.2f INF %0.2f\n', fibers(1,ii), fibers(2,ii), radii(ii));
end
fclose(fid);

return;


function radii = getRadii(fibers, maxRadius)
% Grow the fibers to a radius that is half the distance to it's nearest
% neighbor. 
% Note- we should modify nearpoints so that it can take a single array and
% find the nearest pairs of points.

nFibers = size(fibers,2);
%nearest = zeros(1,nFibers);
radii = zeros(1,nFibers);
far = [9e9, 9e9];
for(ii=1:nFibers)
%     dsq = (fibers(1,ii)-fibers(1,:)).^2 + (fibers(2,ii)-fibers(2,:)).^2;
%     dsq(dsq==0) = spaceScale;
%     nearest(ii) = find(dsq==min(dsq));
%     dist(ii) = sqrt(dsq(nearest(ii)));
    tmp = fibers;
    tmp(:,ii) = far;
    [indices, bestSqDist] = nearpoints2d(fibers, tmp);
    %nearest(ii) = indices(ii);
    radii(ii) = sqrt(bestSqDist(ii))./2;
end
radii(radii>maxRadius) = maxRadius;
% G = sparse([1:nFibers],nearest,dist);

return;
