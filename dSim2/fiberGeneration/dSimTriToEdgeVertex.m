%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Takes a triangle matrix of format
%
%       rawtri = [x1 y1 z1 x2 y2 z2 x3 y3 z3]
%                [x2 y2 z2 x3 y3 z3 x4 y4 z4]
%                ...
%
% where the each line represents the three points of a triangle
% and returns a vertex matrix
%
%       vertices =  [x1 y1 z1]
%                   [x2 y2 z2]
%                   ...
%
% and a triangle matrix
%
%       triangles = [ 1 2 3 ]
%                   [ 2 3 4 ]
%                   ...
% where each line of "triangles" represents a triangle and each element
% represents a vertex (a line in "vertices"). "rawtri" does not have to be ordered
% like shown above. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [triangles, vertices] = dSimTriToEdgeVertex(rawTri)

    vertices = [rawTri(:,1:3);rawTri(:,4:6);rawTri(:,7:9)];
    
    vertices = unique(vertices,'rows');
    
    triangles = zeros(size(rawTri,1),3) - 1;
    
    for (i = 1:size(rawTri,1))
        for (j = 1:3)
            point = rawTri(i,3*j-2:3*j);
            for (k = 1:size(vertices,1))
               % point
               % vertices(k,:)
                if (vertices(k,:) == point)
                    triangles(i,j) = k;
                end
            end
        end
    end
    