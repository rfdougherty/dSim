function dSimGenerateCallosalFibers(densityScale, doSegs)

spaceScale = 100;
outBase = '/home/bob/svn/dSim/fibers_new/';
if(~exist(outBase,'dir')), mkdir(outBase); end
spacing = 0.02;

maxIter = 100000;

if(~exist('doSegs','var'))
    doSegs = [1:5];
end

for(ds = densityScale)
    [fiberDiams,segNames] = dSimGenerateCCFibers(spaceScale*2, ds);
    numSegs = size(fiberDiams,1);
    for(seg=doSegs)
        fname = sprintf('%s_%03d.fibers', segNames{seg}, round(ds*100));
        disp(['*** Processing ' fname '...']);
        % Sort the radii so that we lay down the largest fibers first.
        r = sort(fiberDiams{seg}./2,'descend');
        coords = zeros(numel(r),2);
        % Lay down the first (largest) fiber. We select coords from a uniform
        % random distribution that keeps the fiber completely within the box.
        coords(1,:) = rand(1,2).*2.*(spaceScale-r(1))-(spaceScale-r(1));
        % Now do the rest:
        n = numel(r);
        for(ii=2:n)
            dist = zeros(ii-1,1);
            iter = 0;
            while(min(dist)<=r(ii)+spacing && iter<maxIter)
                newCoord = rand(1,2).*2.*(spaceScale-r(ii))-(spaceScale-r(ii));
                dist = sqrt((newCoord(1)-coords(1:ii-1,1)).^2 + (newCoord(2)-coords(1:ii-1,2)).^2);
                dist = dist - r(1:ii-1);
                iter = iter+1;
            end
            if(iter==maxIter)
                warning(['Exceeded maxIter for ' fname '!']);
                newCoord = [NaN NaN];
                break; 
            end
            coords(ii,:) = newCoord;
            if(mod(ii,ceil(n/10))==0), fprintf('      finished %d of %d fibers...\n',ii,n); end
        end
        if(0)
            figure(87); plot(coords(:,1),coords(:,2),'.'); axis equal; hold on;
            line([coords(:,1)-r, coords(:,1)+r]', [coords(:,2), coords(:,2)]');
            line([coords(:,1), coords(:,1)]', [coords(:,2)-r,coords(:,2)+r]');
            hold off;
        end
        if(all(~isnan(coords(:))))
            fid = fopen(fullfile(outBase,fname),'wt');
            for(ii=1:numel(r))
                fprintf(fid,'%0.2f %0.2f INF %0.2f\n', coords(ii,1), coords(ii,2), r(ii));
            end
            fclose(fid);
        end
    end
end

return;

