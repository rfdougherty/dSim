clear all
close all
clc

v = [0 0 1]';
theta = -pi/4;
ksi = pi/2;
alpha = atan(sqrt(2));
R = rotationMatrix(theta,ksi,alpha);

v_rot = R*v

figure(1)
hold on
plot3([0,v(1)],[0,v(2)],[0,v(3)]);
plot3([0,v_rot(1)],[0,v_rot(2)],[0,v_rot(3)]);
hold off
xlabel('x');
ylabel('y');
zlabel('z');
axis equal
axis([-1 1 -1 1 -1 1]);
grid on
