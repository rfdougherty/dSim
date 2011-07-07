clear all

dSimDir = '/home/bob/svn/dSim/trunk/';
addpath(fullfile(dSimDir,'analysis'),fullfile(dSimDir,'data'));
radiusTest

for(ii=1:m)
   [D,md(ii),fa(ii),l1(ii),l2(ii),l3(ii)] = dSimFitTensor(mrSig{ii}, delta{ii}, Delta{ii}, dwGrads{ii});
end


uSpace = fliplr(unique(radiusSpace));

figure; axis;
set(gca,'FontSize',14);
xlab = 'Axon Radius (\mum)';
lt = 'k-';
symb = {'o','s','d','v','p','x','^','+','h','*','>'};
for(ii=1:numel(uSpace))
   inds = uSpace(ii)==radiusSpace;
   subplot(2,3,1);
   hold on;
   plot(radiusMean(inds),fa(inds),[lt symb{ii}]);
   hold off;
   subplot(2,3,2);
   hold on;
   plot(radiusMean(inds),md(inds),[lt symb{ii}]);
   hold off;
   subplot(2,3,4);
   hold on;
   plot(radiusMean(inds),l1(inds),[lt symb{ii}]);
   hold off;
   subplot(2,3,5);
   hold on;
   plot(radiusMean(inds),l2(inds),[lt symb{ii}]);
   hold off;
   subplot(2,3,6);
   hold on;
   plot(radiusMean(inds),l3(inds),[lt symb{ii}]);
   hold off;
   
   leg{ii} = sprintf('%0.2f\\mum',uSpace(ii));
end
subplot(2,3,1);
ylabel('Fractional Anisotropy');
xlabel(xlab);
subplot(2,3,2);
ylabel('Mean Diffusivity (\mum^2/msec)');
xlabel(xlab);
lh = legend(leg);
set(lh,'FontSize',10);
set(get(lh,'Title'),'String','Spacing');

subplot(2,3,4);
ylabel('\lambda_1 Diffusivity (\mum^2/msec)');
xlabel(xlab);
axis([0 5 1.1 2.1]);
subplot(2,3,5);
ylabel('\lambda_2 Diffusivity (\mum^2/msec)');
xlabel(xlab);
axis([0 5 0.0 1.0]);
subplot(2,3,6);
ylabel('\lambda_3 Diffusivity (\mum^2/msec)');
xlabel(xlab);
axis([0 5 0.0 1.0]);



set(gcf,'Position',[60  60  800  460]);
set(lh,'Position',[0.75 0.49 0.12 0.44]);

outName = fullfile(dSimDir,'doc','figs','radius');
mrUtilPrintFigure([outName '.eps']);

% pstoimg is in the latex2html package (e.g., apt-get install 
unix(['pstoimg -antialias -aaliastext -density 300 -type png -crop a -trans -out ' outName '.png ' outName '.eps']);

