clear all
close all
clc

% 5000 particles
no_of_fibers = [1,3,5,9,17,21,29,37,49,61,83,107];
run_time_big_tree = [90,144,190,223,300,347,430,480,560,654,780,893];
run_time_membranefiber_trees = [90,130,150,190,270,290,350,400,470,530,680,800];

figure(1)
hold on
plot(no_of_fibers,run_time_big_tree,'ro-');
plot(no_of_fibers,run_time_membranefiber_trees,'bo-');
hold off
axis([1 no_of_fibers(end)+5 0 run_time_big_tree(end)+20]);
xlabel('No. of fibers');
ylabel('Running time (secs)');
legend('Big R-Tree','Many small R-Trees');


% 3 fibers
no_of_particles = [5,10,20,30];
run_time_big_tree = [144,234,396,497];
run_time_membranefiber_trees = [130,200,320,450];

figure(2)
hold on
plot(no_of_particles,run_time_big_tree,'ro-');
plot(no_of_particles,run_time_membranefiber_trees,'bo-');
hold off
axis([1 no_of_particles(end)+5 0 run_time_big_tree(end)+20]);
xlabel('No. of particles [k]');
ylabel('Running time (secs)');
legend('Big R-Tree','Many small R-Trees');


% Size of R-Tree Array (kB)
no_of_fibers = [21,107];
size_of_rta = [444,2264];