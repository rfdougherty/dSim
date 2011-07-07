% Load the particles
d = load ('dumpedParticlePositions.txt','-ascii');
printf("loaded the file");
n=32768;
p = [1:32768];
xstd = reshape(p,32768,1);
ystd=xstd;
zstd=xstd;
printf("initialized x,y,z std");
for i = 1:32768
xstd(i) = std (d(:,(i-1)*3+1));
endfor
printf("computed xstd");
for i = 1:32768
ystd(i) = std (d(:,(i-1)*3+2));
endfor
printf("computed ystd");
for i = 1:32768
zstd(i) = std (d(:,(i-1)*3+3));
endfor
printf("computed zstd");
printf("script over");
