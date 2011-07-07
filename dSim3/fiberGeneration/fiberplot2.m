clear all
close all
clc


points = [0.0 0.0 -1.0; 0.0 0.0 0.0; 0.0 0.0 1.0];
M = 15;         % Number of points that define spline
N = 10;         % Number of points that define cross section
r = 0.1;        % Radius of cross section
closed = true;
f = 0.8;

totalTriangles = [];
totalVertices = [];
axonTri = [];
myelinTri = [];
triAxons = [];          % Have stopped using this
triMyelin = [];         % Have stopped using this
n = 0;
maxTriMembrane = 0;

% Construct axon no. n
[fiber_rc, Xc, Yc, Zc, trianglesc] = fiberTriangles(points,M,N,f*r,closed);

[T,V] = makeTriStd(trianglesc);
axonTri = [axonTri ; [size(totalTriangles,1):size(totalTriangles,1)+size(T,1)-1]' ];
totalTriangles = [totalTriangles ; T - 1 + size(totalVertices,1)];
totalVertices = [totalVertices ; V];
triAxons = [triAxons ; n*ones(size(T,1),1)];
if (size(axonTri,1) > maxTriMembrane)
    maxTriMembrane = size(axonTri,1)
end


% Construct myelin no. n
[fiber_rm, Xm, Ym, Zm, trianglesm] = fiberTriangles(points,M,N,r,closed);

[T,V] = makeTriStd(trianglesm);
myelinTri = [myelinTri ; [size(totalTriangles,1):size(totalTriangles,1)+size(T,1)-1]' ];
totalTriangles = [totalTriangles ; T - 1 + size(totalVertices,1)];
totalVertices = [totalVertices ; V];
triMyelin = [triMyelin ; n*ones(size(T,1),1)];
if (size(myelinTri,1) > maxTriMembrane)
    maxTriMembrane = size(myelinTri,1)
end

n = n+1;
%allTriangles = [axonTriangles ; myelinTriangles];

%round(totalVertices*1e6)/1e6;

%A = [ 1 2 3; 4 5 6; 7 8 9];
%nVertices = 304;

%fid = fopen('testOutput.txt','w');
fid = fopen('sim.triangles','w');
fprintf(fid,'Number_of_vertices\n');
fprintf(fid,'%0.0i\n',size(totalVertices,1));
fprintf(fid,'Number_of_triangles\n');
fprintf(fid,'%0.0i\n',size(totalTriangles,1));
fprintf(fid,'Number_of_membrane_types\n');
fprintf(fid,'%0.0i\n',2);
fprintf(fid,'Number_of_fibers\n');
fprintf(fid,'%0.0i\n',n);
fprintf(fid,'Max_triangles_on_membrane\n');
fprintf(fid,'%0.0i\n',maxTriMembrane);
fprintf(fid,'\n');
fprintf(fid,'V\n');
fprintf(fid,'%9.4g %9.4g %9.4g\n', totalVertices');
fprintf(fid,'T\n');
fprintf(fid,'%5.1i %5.1i %5.1i\n', totalTriangles');
% Start looping through n
fprintf(fid,['A',num2str(n-1),'\n']);
fprintf(fid,'%5.1i\n',axonTri');
fprintf(fid,['M',num2str(n-1),'\n']);
fprintf(fid,'%5.1i\n',myelinTri');
% End looping through n

fclose(fid);



%save axons.txt axonTri -ascii
%save myelin.txt myelinTri -ascii
%save triangles.txt totalTriangles -ascii
%save vertices.txt totalVertices -ascii

%save tri2Axons.txt triAxons -ascii
%save tri2Myelin.txt triMyelin -ascii


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Testing all sorts of stuff
% 
testPoint = [0.8,0.5,0];

oPos2pos = [0.896499,0.509502,0.112588 ; 0.895659,0.512044,0.11633];
collPoint2endpos = [0.89599,0.511042,0.114855 ; 0.896601,0.512875,0.116864];

teststartPos0 = [0.782598,0.44772,0.140468];
testendPos0 =   [0.783494,0.446243,0.141397];
teststartPos1 = [0.77721,0.46475,0.151569];
testendPos1 =   [0.783101,0.446132,0.139433];
teststartPos2 = [0.777083,0.46453,0.150818];
testendPos2 =   [0.780902,0.47115,0.173341];
testray0 = [teststartPos0 ; testendPos0];
testray1 = [teststartPos1 ; testendPos1];
testray2 = [teststartPos2 ; testendPos2];

triIndex = 272;
v1Index = totalTriangles(triIndex+1,1);
v2Index = totalTriangles(triIndex+1,2);
v3Index = totalTriangles(triIndex+1,3);
v1 = totalVertices(v1Index+1,:);
v2 = totalVertices(v2Index+1,:);
v3 = totalVertices(v3Index+1,:);
tri = [v1;v2;v3;v1];

triIndex2 = 571;
v1Index2 = totalTriangles(triIndex2+1,1);
v2Index2 = totalTriangles(triIndex2+1,2);
v3Index2 = totalTriangles(triIndex2+1,3);
v12 = totalVertices(v1Index2+1,:);
v22 = totalVertices(v2Index2+1,:);
v32 = totalVertices(v3Index2+1,:);
tri2 = [v12;v22;v32;v12];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%
% Plot results
%

figure(1)
plot3(fiber_rm(1,:),fiber_rm(2,:),fiber_rm(3,:),'bx-', ...
points(:,1),points(:,2),points(:,3),'ro--');
legend('Resampled', 'Original');
axis([-1 1 -1 1 -1 1]);
xlabel('x');
ylabel('y');
zlabel('z');
grid on

figure(2);
hold on
surf(Xm,Ym,Zm,'FaceAlpha',0.2);
surf(Xc,Yc,Zc,'FaceAlpha',0.4);
% plot3(teststartPos0(1),teststartPos0(2),teststartPos0(3),'bx');
% plot3(testendPos0(1),testendPos0(2),testendPos0(3),'bo');
% plot3(testray0(:,1),testray0(:,2),testray0(:,3),'b');
% plot3(teststartPos1(1),teststartPos1(2),teststartPos1(3),'gx');
% plot3(testendPos1(1),testendPos1(2),testendPos1(3),'go');
% plot3(testray1(:,1),testray1(:,2),testray1(:,3),'g');
% plot3(teststartPos2(1),teststartPos2(2),teststartPos2(3),'kx');
% plot3(testendPos2(1),testendPos2(2),testendPos2(3),'ko');
% plot3(testray2(:,1),testray2(:,2),testray2(:,3),'k');
% plot3(tri(:,1),tri(:,2),tri(:,3),'b');
% plot3(tri2(:,1),tri2(:,2),tri2(:,3),'c');
axis equal;
axis([-1 1 -1 1 -1 1]);
xlabel('x');
ylabel('y');
zlabel('z');
grid on
hold off


% T = [-1.0000,0,0,0.9426;
%       0,0,1.0000,-0.4945;
%       0,-1.0000,0,9.1603;
%       0,0,0,1.0000];
% view(T);