addpath('analysis');

fibersFile    = 'fibers/splenium_083.fibers';
fiberMeshFile = 'splenium_083.fmf';
g = 0.6;

fibers = dSimLoadFibers(fibersFile);

numFibers = size(fibers,1);

%spaceScale = max(fibers(:,1));
spaceScale = 100;
fov = spaceScale*2;

fvf = sum(pi*fibers(:,4).^2)./fov^2

od = fibers(:,4)./spaceScale;
id = od .* g;

S = [spaceScale 0 0; 0 spaceScale 0; 0 0 spaceScale];   % Matrix for scaling the fibers in x, y or z direction
% %R = rotationMatrix(-pi/4,pi/2,atan(sqrt(2)));        % Matrix for rotating the fibers - see function doc
R = [1,0,0;0,1,0;0,0,1];
for (fiber=1:numFibers)
    points{fiber} = (R*S*points{fiber}')';
    r(fiber) = ss*r(fiber);
end

M = 2;                  % Number of points that define spline
N = 12;                  % Number of points that define cross section
closed = true;          % Whether we close the fibers at the ends

totalTriangles = [];
totalVertices = [];
axonTri = [];
myelinTri = [];
%triAxons = [];          % Have stopped using this
%triMyelin = [];         % Have stopped using this
%n = 0;
maxTriMembrane = 0;


% Construct axon no. n
for (n=0:numFibers-1)
    [fiber_rc{n+1}, Xc{n+1}, Yc{n+1}, Zc{n+1}, trianglesc] = fiberTriangles(points{n+1},M,N,f(n+1)*r(n+1),closed);

    [T,V] = makeTriStd(trianglesc);
    %axonTri = [axonTri ; [size(totalTriangles,1):size(totalTriangles,1)+size(T,1)-1]' ];
    axonTri{n+1} = [size(totalTriangles,1):size(totalTriangles,1)+size(T,1)-1]';
    totalTriangles = [totalTriangles ; T - 1 + size(totalVertices,1)];
    totalVertices = [totalVertices ; V];
    %triAxons = [triAxons ; n*ones(size(T,1),1)];
    if (size(T,1) > maxTriMembrane)
        maxTriMembrane = size(T,1);
    end


    % Construct myelin no. n
    [fiber_rm{n+1}, Xm{n+1}, Ym{n+1}, Zm{n+1}, trianglesm] = fiberTriangles(points{n+1},M,N,r(n+1),closed);

    [T,V] = makeTriStd(trianglesm);
    %myelinTri = [myelinTri ; [size(totalTriangles,1):size(totalTriangles,1)+size(T,1)-1]' ];
    myelinTri{n+1} = [size(totalTriangles,1):size(totalTriangles,1)+size(T,1)-1]';
    totalTriangles = [totalTriangles ; T - 1 + size(totalVertices,1)];
    totalVertices = [totalVertices ; V];
    %triMyelin = [triMyelin ; n*ones(size(T,1),1)];
    if (size(T,1) > maxTriMembrane)
        maxTriMembrane = size(T,1);
    end
end
 

startBox = 0.06;
volumeFractionAxons = sum(0.5*(f.*r).^2*sin(2*pi/N)*N)/(startBox*2*ss)^2;
volumeFractionMyelin = sum(0.5*r.^2*sin(2*pi/N)*N)/(startBox*2*ss)^2 - volumeFractionAxons;


disp(['Fraction of spins in axons should be about ', num2str(volumeFractionAxons*100), '%']);
disp(['Fraction of spins in myelin should be about ', num2str(volumeFractionMyelin*100), '%']);


% fid = fopen('sim.triangles_17','w');
if (write2File)
    fid = fopen(fiberMeshFile,'w');
    
    %fprintf(fid,'Number_of_vertices\n');
    %fprintf(fid,'%0.0i\n',size(totalVertices,1));
    %fprintf(fid,'Number_of_triangles\n');
    %fprintf(fid,'%0.0i\n',size(totalTriangles,1));
    %fprintf(fid,'Number_of_membrane_types\n');
    %fprintf(fid,'%0.0i\n',2);
    %fprintf(fid,'Number_of_fibers\n');
    %fprintf(fid,'%0.0i\n',numFibers);
    %fprintf(fid,'Max_triangles_on_membrane\n');
    %fprintf(fid,'%0.0i\n',maxTriMembrane);
    %fprintf(fid,'\n');
    fprintf(fid,'V\n');
    fprintf(fid,'%9.4g %9.4g %9.4g\n', totalVertices');
    fprintf(fid,'T\n');
    fprintf(fid,'%5.1i %5.1i %5.1i\n', totalTriangles');
    for (n=0:numFibers-1)
        fprintf(fid,['A',num2str(n),'\n']);
        fprintf(fid,'%5.1i\n',axonTri{n+1}');
        fprintf(fid,['M',num2str(n),'\n']);
        fprintf(fid,'%5.1i\n',myelinTri{n+1}');
    end

    fclose(fid);
end



