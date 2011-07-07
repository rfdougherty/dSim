clear all

axonRadiusFraction = 0.7;

bd = '/white/u5/bob/svn/dSim/';

outDir = fullfile(bd,'densitySim');

if(~exist('dSimFitTensor.m','file')), addpath(fullfile(bd,'analysis')); end
if(~exist('densityTest.m','file')), addpath(fullfile(bd,'data')); end

densityTest

segs = {'genu','antBody','midBody','postBody','splenium'};
for(jj=1:numel(segs))
    inds = strmatch(['fibers/' segs{jj}],fibersFile);

    for(ii=1:numel(inds))
        [D,md{jj}(ii),fa{jj}(ii),l1{jj}(ii),l2{jj}(ii),l3{jj}(ii)] = dSimFitTensor(mrSig{inds(ii)}, delta{inds(ii)}, Delta{inds(ii)}, dwGrads{inds(ii)});
        f = dSimLoadFibers(fullfile(bd,fibersFile{inds(ii)}));
        density{jj}(ii) = size(f,1)./spaceScale(inds(ii)).^2;
        myelinVolumeFraction{jj}(ii) = sum(pi.*f(:,4).^2 - pi.*(f(:,4).*axonRadiusFraction).^2)./(spaceScale(inds(ii)).*2).^2;
        if(ii==1), fibers{ii} = f; end
    end
end

figure(866);
for(jj=1:5)
    subplot(4,5,jj);
    f = fibers{jj};
    f = f(f(:,1)<=5&f(:,1)<=-5&f(:,2)<=5&f(:,2)<=-5,:);
    for(ii=1:size(f,1))
        r = f(ii,4)*2; x = f(ii,1)-r/2; y = f(ii,2)-r/2;
        rectangle('Position',[x,y,r,r],'Curvature',[1,1],'FaceColor',[0.2 0.2 0.2]);
        r = r*0.7; x = f(ii,1)-r/2; y = f(ii,2)-r/2;
        rectangle('Position',[x,y,r,r],'Curvature',[1,1],'FaceColor','w');
    end
    title(segs{jj});
    axis equal tight off;
end

for(jj=1:5)
    subplot(4,5,jj+5); 
    plot(density{jj}, l1{jj}, 'rs', density{jj}, l2{jj}, 'go', density{jj}, l3{jj}, 'b^');
    xlabel('density (fibers/\mum^2)');
    ylabel('ADC (\mum^2/msec)');

    subplot(4,5,jj+10);
    plot(density{jj}, fa{jj}, 'ko');
    xlabel('density (fibers/\mum^2)');
    ylabel('FA');

    subplot(4,5,jj+15);
    x = myelinVolumeFraction{jj};
    y = fa{jj};
    plot(x, y, 'ko');
    p = polyfit(x, y, 1);
    hold on; plot([min(x),max(x)], [min(x),max(x)]*p(1)+p(2),'k-'); hold off;
    xlabel('myelin volume');
    ylabel('FA');
    title(sprintf('fa=mvf*%0.2f+%0.2f',p(1), p(2)));
end
mrUtilResizeFigure(866, 900, 700);

fn = fullfile(outDir,'allDensity');
mrUtilPrintFigure([fn '.eps'],866);
unix(['pstoimg -antialias -aaliastext -density 300 -type png -crop a -trans -out ' fn '.png ' fn '.eps']);



