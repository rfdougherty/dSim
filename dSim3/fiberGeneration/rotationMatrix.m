% Produces rotation matrix for a rotation of angle alpha
% about a vector u that makes an angle ksi to the z axis and
% whose projection onto the xy-plane makes an angle theta 
% to the x axis. All angles are in radians.
%
% For example, to create a matrix R such that R*v rotates
% v 45 degrees about the 
%           x-axis: R = rotationMatrix(0,pi/2,pi/4)
%           y-axis: R = rotationMatrix(pi/2,pi/2,pi/4)
%           z-axis: R = rotationMatrix(0,0,pi/4)
%
% v should be a 3x1 column vector (or a 3xn matrix with each
% column representing one vector). A positive rotation around 
% u is defined as clockwise around u when u is directed at
% the observer (i.e. left handed rotation).

function [Ru] = rotationMatrix(theta,ksi,alpha)

Rz_theta = [ cos(theta) , sin(theta) , 0 ;
            -sin(theta) , cos(theta) , 0 ;
             0          , 0          , 1 ];
         
Ry_ksi = [ cos(ksi)   , 0          , -sin(ksi)  ;
           0          , 1          ,  0         ;
           sin(ksi)   , 0          ,  cos(ksi) ];
       
Rz_alpha = [ cos(alpha) , sin(alpha) , 0 ;
            -sin(alpha) , cos(alpha) , 0 ;
             0          , 0          , 1 ];
         
Ry_mksi = [ cos(-ksi)   , 0          , -sin(-ksi)  ;
            0           , 1          ,  0          ;
            sin(-ksi)   , 0          ,  cos(-ksi) ];
        
Rz_mtheta = [ cos(-theta) , sin(-theta) , 0 ;
             -sin(-theta) , cos(-theta) , 0 ;
              0           , 0           , 1 ];
          
Ru = Rz_mtheta*Ry_mksi*Rz_alpha*Ry_ksi*Rz_theta;