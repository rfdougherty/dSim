
clear all
close all
clc

oPosx = -0.00854051; oPosy = -0.418389; oPosz = -0.635162;
posx = -0.00918993; posy = -0.419038; posz = -0.644214;
x1 = 0.0357;y1 = -0.4258;z1 = -0.7714; 
x2 = -0.0308;y2= -0.4167;z2 = -0.5751; 
x3 = -0.0036;y3 = -0.3571;z3 = -0.5949; 
nx = -0.0118797;ny = 0.00402266;nz = -0.00421092;
intPx = -0.00881566; intPy = -0.418664; intPz = -0.638997;





r = (nx*(x1-oPosx)+ny*(y1-oPosy)+nz*(z1-oPosz))/(nx*(posx-oPosx)+ny*(posy-oPosy)+nz*(posz-oPosz))
int2Px = oPosx + r*(posx-oPosx)
int2Py = oPosy + r*(posy-oPosy)
int2Pz = oPosz + r*(posz-oPosz)

ux = x3-x1
uy = y3-y1
uz = z3-z1
vx = x2-x1
vy = y2-y1
vz = z2-z1
wx = intPx-x1; wy = intPy-y1; wz = intPz-z1;

uu = ux^2 + uy^2 + uz^2
vv = vx^2 + vy^2 + vz^2
uv = ux*vx + uy*vy + uz*vz
wu = wx*ux + wy*uy + wz*uz
wv = wx*vx + wy*vy + wz*vz

s = (uv*wv-vv*wu)/(uv*uv-uu*vv)
t = (uv*wu-uu*wv)/(uv*uv-uu*vv)
s+t

figure(1)
hold on
plot3([x1,x2,x3,x1]',[y1,y2,y3,y1]',[z1,z2,z3,z1]');
plot3([oPosx,intPx],[oPosy,intPy],[oPosz,intPz],'r');
plot3([intPx,posx],[intPy,posy],[intPz,posz],'g');
plot3([x1,x1+nx],[y1,y1+ny],[z1,z1+nz],'m');
hold off
axis equal
grid on