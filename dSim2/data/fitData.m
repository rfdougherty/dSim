
% load the data
m = 0;
radiusTest

% process it
gradsFile = '../sim.grads';
for(ii=1:m)
   [D,md(ii),fa(ii),ad(ii),rd(ii)] = dSimFitTensor(mrSig{ii}, gradsFile);
end

uSpace = fliplr(unique(radiusSpace));

figure; axis;
set(gca,'FontSize',14);
lt = 'k-';
symb = {'o','s','d','v','p','x','^','+','h','*','>'};
for(ii=1:numel(uSpace))
   hold on;
   inds = uSpace(ii)==radiusSpace;
   plot(radiusMean(inds),rd(inds),[lt symb{ii}]);
   hold off;
   leg{ii} = sprintf('%0.2f\\mum',uSpace(ii));
end
lh = legend(leg);
set(lh,'FontSize',10);
set(get(lh,'Title'),'String','Spacing');

ylabel('Radial Diffusivity (\mum^2/msec)');
xlabel('Axon radius (\mum)');
set(gcf,'Position',[1000  700  460  350]);
mrUtilPrintFigure('radius_spacing_RD');
