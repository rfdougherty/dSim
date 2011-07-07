addpath('analysis');
addpath('fiberGeneration');

%ffName = 'genu_110';
%ffName = 'midBody_060';
%ffName = 'splenium_085';
ffName = 'antBody_090';
fibersFile = fullfile('..','dSim','fibers', [ffName '.fibers']);
gRatios = [0.50:0.05:0.70];
minG = 0.5;
    
for(g=gRatios)
    fiberMeshFile = fullfile('fibers', sprintf('%s_g%01d.fmf', ffName, round(g*100)));
    
    fibers = dSimLoadFibers(fibersFile);
    
    spaceScale = 60;
    fov = spaceScale*2;
    
    % Crop away fibers outside the desired FOV:
    fovFibers = (abs(fibers(:,1))+fibers(:,4))<spaceScale & (abs(fibers(:,2))+fibers(:,4))<spaceScale;
    % Remove very small fibers
    fovFibers = fovFibers & fibers(:,4)>=0.25;
    x = fibers(fovFibers,1);
    y = fibers(fovFibers,2);
    r = fibers(fovFibers,4);
    
    numFibers = numel(x)
    
    fvf = sum(pi*r.^2)./fov^2
    
    % We want to keep the *inner* radius constant with varying g, and keep the
    % outerRadius <= r:
    innerRadius = r .* minG;
    outerRadius = innerRadius ./ g;
    
    % %R = rotationMatrix(-pi/4,pi/2,atan(sqrt(2)));        % Matrix for rotating the fibers - see function doc
    R = [1,0,0;0,1,0;0,0,1];
    points = cell(1,numFibers);
    for(ii=1:numFibers)
        points{ii} = (R*[x(ii) y(ii) -spaceScale; x(ii) y(ii) 0; x(ii) y(ii) +spaceScale]')';
    end
    
    M = 2;                  % Number of points that define spline
    N = 12;                  % Number of points that define cross section
    closed = true;          % Whether we close the fibers at the ends
    
    totalTriangles = [];
    totalVertices = [];
    axonTri = [];
    myelinTri = [];
    maxTriMembrane = 0;
    
    for(ii=1:numFibers)
        % Construct the axon
        [fiber_rc{ii}, Xc{ii}, Yc{ii}, Zc{ii}, trianglesc] = dSimGetFiberTriangles(points{ii},M,N,innerRadius(ii),closed);
        
        [T,V] = dSimTriToEdgeVertex(trianglesc);
        %axonTri = [axonTri ; [size(totalTriangles,1):size(totalTriangles,1)+size(T,1)-1]' ];
        axonTri{ii} = [size(totalTriangles,1):size(totalTriangles,1)+size(T,1)-1]';
        totalTriangles = [totalTriangles ; T - 1 + size(totalVertices,1)];
        totalVertices = [totalVertices ; V];
        %triAxons = [triAxons ; n*ones(size(T,1),1)];
        if (size(T,1) > maxTriMembrane)
            maxTriMembrane = size(T,1);
        end
        
        % Construct the myelin sheath
        [fiber_rm{ii}, Xm{ii}, Ym{ii}, Zm{ii}, trianglesm] = dSimGetFiberTriangles(points{ii},M,N,outerRadius(ii),closed);
        
        [T,V] = dSimTriToEdgeVertex(trianglesm);
        %myelinTri = [myelinTri ; [size(totalTriangles,1):size(totalTriangles,1)+size(T,1)-1]' ];
        myelinTri{ii} = [size(totalTriangles,1):size(totalTriangles,1)+size(T,1)-1]';
        totalTriangles = [totalTriangles ; T - 1 + size(totalVertices,1)];
        totalVertices = [totalVertices ; V];
        %triMyelin = [triMyelin ; n*ones(size(T,1),1)];
        if (size(T,1) > maxTriMembrane)
            maxTriMembrane = size(T,1);
        end
    end
    
    
    startBox = 0.06;
    volumeFractionAxons = sum(0.5*(innerRadius).^2*sin(2*pi/N)*N)/(startBox*2*spaceScale)^2;
    volumeFractionMyelin = sum(0.5*outerRadius.^2*sin(2*pi/N)*N)/(startBox*2*spaceScale)^2 - volumeFractionAxons;
    
    disp(['Fraction of spins in axons should be about ', num2str(volumeFractionAxons*100), '%']);
    disp(['Fraction of spins in myelin should be about ', num2str(volumeFractionMyelin*100), '%']);
    
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
    for(ii=1:numFibers)
        fprintf(fid,['A',num2str(ii-1),'\n']);
        fprintf(fid,'%5.1i\n',axonTri{ii}');
        fprintf(fid,['M',num2str(ii-1),'\n']);
        fprintf(fid,'%5.1i\n',myelinTri{ii}');
    end
    fclose(fid);
    
end

