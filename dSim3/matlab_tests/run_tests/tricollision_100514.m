clear all
close all
clc

oPos = [0.735727,0.46366,0.110058];
pos = [0.738849,0.47971,0.11884];

% Triangle 269
v1 = [0.605839, 0.401189, 0.168516];
v2 = [0.774693, 0.502321, 0.0838178];
v3 = [0.795898, 0.475108, 0.126296];


allv = [v1;v2;v3];
meanv = mean(allv,1)
%     0.7255    0.4595    0.1262
maxxyz = max(allv)
%     0.7959    0.5023    0.1685
minxyz = min(allv)
%     0.6058    0.4012    0.0838

allv = [allv ; v1];

v = v2 - v1;
u = v3 - v1;
C = cross(v,u)

uv = dot(u,v)

%ray = [meanv-0.5*C; meanv+0.5*C]
ray = [oPos;pos];

figure(1)
hold on
plot3(allv(:,1),allv(:,2),allv(:,3));
plot3(ray(:,1),ray(:,2),ray(:,3));
hold off
axis equal


