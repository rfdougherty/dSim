function [fiber_r, X, Y, Z, triangles] = dSimGetFiberTriangles(points,M,N,r,closed)
%
% [fiber_r, X, Y, Z, triangles] = dSimGetFiberTriangles(points,M,N,r,closed)
%
%points = [0.0 -0.5 -1.0; 0.0 0.0 0.0; 1.0 0.5 0.0];
%M = 15;         % Number of points that define spline

fiber=points'; 
fiber_r=dSimFiberResample(fiber, M, 'N');


%N = 10;         % Number of points that define cross section
theta = [0:2*pi/N:2*pi*(N-1)/N];
%r = 0.1;
xbase = r*cos(theta);
ybase = r*sin(theta);
base = [xbase;ybase];


[X,Y,Z] = dSimExtrude(base,fiber_r);

% Fix ends of fiber

% r_max_start = max((-1-Z(:,2))./(Z(:,1)-Z(:,2)));
% r_max_end = max((1-X(:,end-1))./(X(:,end)-X(:,end-1)));
% 
% X(:,1) = X(:,2) + 1.01*r_max_start*(X(:,1)-X(:,2));
% Y(:,1) = Y(:,2) + 1.01*r_max_start*(Y(:,1)-Y(:,2));
% Z(:,1) = Z(:,2) + 1.01*r_max_start*(Z(:,1)-Z(:,2));
% fiber_r(1,1) = fiber_r(1,2) + 1.01*r_max_start*(fiber_r(1,1)-fiber_r(1,2));
% fiber_r(2,1) = fiber_r(2,2) + 1.01*r_max_start*(fiber_r(2,1)-fiber_r(2,2));
% fiber_r(3,1) = fiber_r(3,2) + 1.01*r_max_start*(fiber_r(3,1)-fiber_r(3,2));
% 
% X(:,end) = X(:,end-1) + 1.01*r_max_end*(X(:,end)-X(:,end-1));
% Y(:,end) = Y(:,end-1) + 1.01*r_max_end*(Y(:,end)-Y(:,end-1));
% Z(:,end) = Z(:,end-1) + 1.01*r_max_end*(Z(:,end)-Z(:,end-1));
% fiber_r(1,end) = fiber_r(1,end-1) + 1.01*r_max_end*(fiber_r(1,end)-fiber_r(1,end-1));
% fiber_r(2,end) = fiber_r(2,end-1) + 1.01*r_max_end*(fiber_r(2,end)-fiber_r(2,end-1));
% fiber_r(3,end) = fiber_r(3,end-1) + 1.01*r_max_end*(fiber_r(3,end)-fiber_r(3,end-1));



% Find volume of fiber
volsum = 0;
for (m=1:M-1)
    volsum = volsum + r^2*pi*sqrt((fiber_r(1,m+1)-fiber_r(1,m))^2+(fiber_r(2,m+1)-fiber_r(2,m))^2+(fiber_r(3,m+1)-fiber_r(3,m))^2);
end
%volsum

Ntriangles = 2*N*(M-1);
triangles = zeros(Ntriangles,9);

% Create the triangle mesh from the collection of vertices (X,Y,Z)
for (m = 1:M-1)
    for (n = 1:N-1)
        triangles(2*((m-1)*N+(n-1))+1,:) = [X(n,m) Y(n,m) Z(n,m) X(n,m+1) Y(n,m+1) Z(n,m+1) ...
                                        X(n+1,m) Y(n+1,m) Z(n+1,m)];
        triangles(2*((m-1)*N+(n-1))+2,:) = [X(n+1,m) Y(n+1,m) Z(n+1,m) X(n,m+1) Y(n,m+1) Z(n,m+1) ...
                                  X(n+1,m+1) Y(n+1,m+1) Z(n+1,m+1)];
                              
    end
        % n = N is a special case

        triangles(2*(m*N-1)+1,:) = [X(N,m) Y(N,m) Z(N,m) X(N,m+1) Y(N,m+1) Z(N,m+1) ...
                                        X(1,m) Y(1,m) Z(1,m)];
        triangles(2*(m*N-1)+2,:) = [X(1,m) Y(1,m) Z(1,m) X(N,m+1) Y(N,m+1) Z(N,m+1) ...
                                  X(1,m+1) Y(1,m+1) Z(1,m+1)];
                              
end



% The triangle mesh is now open at the ends - we close the mesh with
% startTriangles and endTriangles

startTriangles = zeros(N,9);
for (n = 1:N-1)
    %n
    startTriangles(n,:) = [fiber_r(1,1) fiber_r(2,1) fiber_r(3,1) ...
                           X(n,1) Y(n,1) Z(n,1) X(n+1,1) Y(n+1,1) Z(n+1,1)];                              
end
% n = N is a special case
%N
startTriangles(N,:) = [fiber_r(1,1) fiber_r(2,1) fiber_r(3,1) ...
                           X(N,1) Y(N,1) Z(N,1) X(1,1) Y(1,1) Z(1,1)];
                    
                       
endTriangles = zeros(N,9);
for (n = 1:N-1)
    endTriangles(n,:) = [fiber_r(1,end) fiber_r(2,end) fiber_r(3,end) ...
                           X(n,end) Y(n,end) Z(n,end) X(n+1,end) Y(n+1,end) Z(n+1,end)];
end
% n = N is a special case
endTriangles(N,:) = [fiber_r(1,end) fiber_r(2,end) fiber_r(3,end) ...
                           X(N,end) Y(N,end) Z(N,end) X(1,end) Y(1,end) Z(1,end)];

                       
% Add the three meshes into one mesh, triangles2
triangles2 = [startTriangles; endTriangles; triangles];

% Find the normal to each triangle
% Note - have stopped using this
normals = zeros(Ntriangles,3);
for (k = 1:Ntriangles)
    a1 = triangles(k,1)-triangles(k,4);
    a2 = triangles(k,2)-triangles(k,5);
    a3 = triangles(k,3)-triangles(k,6);
    b1 = triangles(k,1)-triangles(k,7);
    b2 = triangles(k,2)-triangles(k,8);
    b3 = triangles(k,3)-triangles(k,9);
    
    normals(k,1) = a2*b3-a3*b2;
    normals(k,2) = a3*b1-a1*b3;
    normals(k,3) = a1*b2-a2*b1;
end

% Return closed mesh if prompted
if closed
    triangles = triangles2;
end
