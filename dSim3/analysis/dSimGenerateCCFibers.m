function [fiberDiams,segNames] = dSimGenerateCCFibers(boxSize, densityScale)
%
% Usage: [fiberDiams,segNames] = dSimGenerateCCFibers(boxSize, densityScale)
%
% Generate a matrix of fiber diameters given a box size. 
% Each row of the matrix corresponds to one of the five
% parts of the corpus callosum.  It will also plot the 
% resulting histograms.
% 
% HISTORY:
% 2009.07.28 Nikola Stikov wrote it.
% 2009.07.30 Aviv: now also accounts for tissue shrinkage in the Aboitiz data.
%            density = density.*inv(1.5^2); for the box
%            fiberDiams = fiberDiams.*1.5;  for the axon sizes

showHist = false;

rand('seed', 0);

shrinkageFactor = 1.5;

if(~exist('densityScale','var')||isempty(densityScale))
    densityScale = 1.0;
end


segNames = {'genu', 'antBody', 'midBody', 'postBody', 'splenium'};

% The code below generates the target histograms
% copied from the Aboitiz paper

target = zeros(5, 46);

% From the Aboitiz paper

numBins = [19 30 44 31 27]; % The number of bins in the distribution before it goes to 0

bins = linspace(0, 9, 46);

% Target histograms from electron microscopy

target(1, 1:numBins(1)) = [0 2.1 13 25.8 17.9 12.8 7.7 4.0 6.7 1.9 1.2 1.2 1.1 .7 .7 .5 0 0 .5];
target(2, 1:numBins(2)) = [0 2.5 19.2 22.2 18.4 14.3 7.2 4 4 2.8 1.8 1.4 1.1 0 2.3 1.2 0.9 .9 .9 0 .9 0 0 0 0 0 .8 0 0 .9];
target(3, 1:numBins(3)) = [0 1.6 13.5 17.3 20.5 11.4 10.7 5.6 2.6 3.7 1.8 2.3 0.5 3 .9 1.1 1.2 0 1.2 0 1.6 0 .5 .9 0 0 0 .9 0 0 .7 .7 0 0 .7 0 0 0 0 0 0 0 0 .4];
target(4, 1:numBins(4)) = [0 1 5.6 12 18.1 14.1 10.4 8.9 6.3 5.9 4.2 2.6 1.7 .9 1 1.2 1.2 1.3 0 1.4 1 1.5 .8 0 1.2 1 0 0 0 0 1];
target(5, 1:numBins(5)) = [0 2.1 7.3 15.9 16.5 18.2 15.6 4.6 6.6 4.8 3.2 4.4 2.0 1.8 0 1.4 0 0 .7 0 .8 0 0 0 0 0 .9];

% Target density from light microscopy

density = [.39 .35 .33 .3 .37]; %number of Fibers per unite area, does not include fibers <.4um, adjusted below
density = densityScale .* density .* 1./shrinkageFactor^2; %aviv edits for the shrinkage in the box!
maxNumFibers = ceil(max(density+.1)*boxSize^2); %Making sure it is big enough to fit all fibers
fiberDiams = cell(5, 1);

for ii=1:5
    target(ii,:) = target(ii, :)/sum(target(ii,:)); %To make sure that pdf sums to one
    density2umFibers = target(ii, 2)/sum(target(ii, 3:end)) * density(ii); %density of 2um Fibers per um^2 
    density(ii) = density(ii);% + density2umFibers; %Add all small fibers that were not included in density
    totalFibers(ii) = density(ii)*boxSize^2; %Adjust number of fibers by the size of the box
    targetFibers(ii, :) = round(target(ii,:) * totalFibers(ii)); %target diameters per bin
    
    % Now let's generate the diameters that give a distribution like in
    % targetFibers.  Assume that the size associated with a bin is the
    % minimum size of fibers in that bin
    
    diams = [];
    for jj=2:numel (bins) %not using the zero bin
        binDiams = bins(jj) + .2*rand(targetFibers(ii, jj), 1); %Generate the fibers in that bin according to a uniform distribution
        diams = cat(1, diams, binDiams);
    end
    
    fiberDiams{ii} = diams.*shrinkageFactor;
    
    if(showHist)
        figure(869); subplot(1,5,ii); hist(diams(diams>0), bins+.1);
    end
%     eval(['coverage_' segNames{ii} '= sum((diams/2).^2*pi)/boxSize^2'])
%     eval(['MVF_' segNames{ii} ' = sum((diams/2).^2*pi - (.7*diams/2).^2*pi)/boxSize^2'])
%     sprintf('%g percent of total area', coverage); 
%     
end

return;


% Sample code:
arf = 0.7;

spaceScale = 100;
diams = dSimGenerateCCFibers(spaceScale);
for(ii=1:numel(diams))
    axRad = diams{ii}./2;
    density(ii) = sum(pi.*axRad.^2) ./ spaceScale^2;
    mvf(ii) = sum(pi.*axRad.^2 - pi.*(axRad.*arf).^2) ./ spaceScale^2;
end
density
mvf

% These are the names of the fiber groups
% Genu, anterior body, mid body, posterior body, splenium


% Below is the code for gamma distributions

% for ii=1:5
% 
%     g = gamfit(bins, [], [], target(ii, :));
%     result = gampdf(bins, g(1), g(2));
%     result = result./max(result)*max(target(ii, :))/100;
%     figure; plot(bins, result)
%     hold on; bar(bins, target(ii, :)/100)
%     
%     outName = ['sim.fibers' segNames{ii}]
% 
%     dSimGenerateFibers(outName, g(1), g(2), density(ii), 50, 5000);
%     clear result;
% 
% end

% % Now that they are generated, let's check if the distributions fit the
% % original date
% 
% for ii=1:5
%     g = gamfit(bins, [], [], target(ii, :));
%     result = gampdf(bins, g(1), g(2));
%     outName = ['sim.fibers' segNames{ii}];
%     [xLoc yLoc zLoc radius] = textread(outName,'%f%f%s%f');
%     radHist = hist(radius, bins);
%     figure; 
%     hold on; plot(bins, radHist, 'b'); plot(bins, result/max(result)*max(radHist), 'k');
%     legend ('obtained', 'target');
%     
% end
