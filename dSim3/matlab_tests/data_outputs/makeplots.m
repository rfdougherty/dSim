clear all
close all
clc

% analysispath = '../../../analysis';
% addpath(analysispath)
testfile

% [D,md,fa,l1,l2,l3] = dSimFitTensor(mrSignal, delta, Delta, dwGrads)


[D,md,fa,l1,l2,l3] = dSimFitTensor(mrSig{1}, delta{1}, Delta{1}, dwGrads{1});
max_eig = max([l1,l2,l3]);
rd = (l2+l3)/2;
figure(1)
ellipsoid(0,0,0,l1/max_eig,l2/max_eig,l3/max_eig);
xlabel('z')
ylabel('x')
zlabel('y')
axis equal


% ar = [50 60 70 80 90];
% rd = [];
% 
% figure(1)
% for(k = 1:5)
%     [D,md,fa,l1,l2,l3] = dSimFitTensor(mrSig{k}, delta{k}, Delta{k}, dwGrads{k});
%     rd = [rd ; (l2+l3)/2];
%     max_eig = max([l1,l2,l3]);
%     plot_title = {strcat('Axon radius: ',num2str(50+10*(k-1)), '%, New reflection type');strcat('l_1 = ',num2str(l1), ', l_2 = ',num2str(l2),',l_3 = ',num2str(l3))}
%     subplot(3,2,k)
%     ellipsoid(0,0,0,l1/max_eig,l2/max_eig,l3/max_eig);
%     axis([-1 1 -1 1 -1 1])
%     xlabel('z')
%     ylabel('x')
%     zlabel('y')
%     title(plot_title)
%     view(-37.5,30)
% end


% figure(2)
% plot(ar,rd,'o-')
% title('Radial diffusivity as function of inner/outer axon ratio')
% xlabel('Inner/Outer axon ratio (%)');
% ylabel('Radial diffusivity');
% 
% rmpath(analysispath)