clear all

cd /home/bob/svn/dSim/trunk/data
gpuTest

figNum = figure;

cpuInds = useGpu([1:14])==0;
plot(log10(numParticles(cpuInds)), log10(totalTime(cpuInds)), 'o-b');
gpuInds = ~cpuInds;
hold on; 
plot(log10(numParticles(gpuInds)), log10(totalTime(gpuInds)), 's-r');

% resize the figure
p = get(figNum,'Position');
set(figNum, 'Position', [p(1), p(2), width, height]);

axis([1.6 5.6 0 4.1]);
x = get(gca,'xtick');
y = get(gca,'ytick');
set(gca,'xticklabel',strcat('10^',num2str(round(x'))));
set(gca,'yticklabel',num2str(round(10.^y')));

xlabel('Number of spins');
ylabel('Time (seconds)');

fileName = 'gpuParticlesTime.eps';
set(figNum, 'PaperUnits', 'inches','PaperOrientation','portrait');
set(figNum, 'PaperPositionMode', 'auto');
pgPos = get(figNum,'PaperPosition');
pgSize = [pgPos(3)-pgPos(1) pgPos(4)-pgPos(2)];
print(figNum, '-depsc', '-tiff', '-cmyk', '-loose', '-r300', '-painters', fileName);
% pstoimg is in the latex2html package (e.g., apt-get install 
unix('pstoimg -antialias -aaliastext -density 300 -type png -crop a -trans -out gpuParticlesTime.png gpuParticlesTime.eps');
