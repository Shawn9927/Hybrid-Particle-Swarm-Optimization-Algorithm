%% I. clear environment
clc;
clear;
close all;
d2r = pi/180;

%% II. model building
% 3 pursuers position, velocity vector and
% Effective detection distance and maximum detection distance
p1 = [0 0; 0 0.01; 30 40];
p2 = [0 0.1; 0 0.01; 30 50];
p3 = [0.1 0; -0.01 0.01; 25 30];
p4 = [0.1 0.1; 0.01 0.1; 30 40];
p5 = [-0.2 0; 0.01 0.1; 30 40];
p6 = [0 0.05; 0 0.01; 20 40];
p7 = [0 0.1; -0.1 0.01; 30 50];
p8 = [-0.1 0; 0.08 0.01; 25 30];
p9 = [0.1 0.1; 0.01 0.16; 20 40];
p10 = [0.2 0; -0.6 0.1; 30 40];
P(1, :, :) = p1;
P(2, :, :) = p2;
P(3, :, :) = p3;
P(4, :, :) = p4;
P(5, :, :) = p5;
P(6, :, :) = p6;
P(7, :, :) = p7;
P(8, :, :) = p8;
P(9, :, :) = p9;
P(10, :, :) = p10;
% 3 evaders position, velocity vector and points of importance
e1 = [15 15; -1 -1; 0.5 0];
e2 = [-20 0.1; 1 0; 0.7 0];
e3 = [0.1 14; 0 -1; 0.5 0];
e4 = [5 10; -0.5 -2; 0.8 0];
e5 = [-15 10; 1 -1; 0.3 0];
e6 = [20 2; -1 -1.4; 0.9 0];
e7 = [-10 14; 0.7 -1; 0.5 0];
e8 = [-19 10; 1.5 -2; 0.5 0];
E(1, :, :) = e1;
E(2, :, :) = e2;
E(3, :, :) = e3;
E(4, :, :) = e4;
E(5, :, :) = e5;
E(6, :, :) = e6;
E(7, :, :) = e7;
E(8, :, :) = e8;
%% III. Main function

tic;
fbest = vnspso(P, E);
toc;

% aaa = [0 0 1 0; 0 1 0 0; 0 0 0 1; 0 1 0 0; 1 0 0 0];
% fitnessf(aaa, P, E)

%% IV. Fitness functions
function Fr1 = threatrangeFit(e)
% threat distance fitness function
obj = [0 0]; %attacking object's position
temp = e(1, :);
R = obj - temp; %range vector
V = e(2, :);  %velocity vector of attacking missile
Rt = 10*sign(dot(R,V))*norm(R)*norm(cross([R 0],[V 0]))/norm(R)/norm(V); % threat distance
temp2 = (1-sign(Rt))*norm(R);
if Rt == 0
    if R(1)*V(1) > 0
        temp2 = 0;
    else
        temp2 = 2*norm(R);
    end
end

Fr1 = exp(-(temp2*10+Rt)*0.1); %threat distance fitness function
end

function Fr2 = velE(e)
% flying velocity fitness function
V = e(2, :);
Fr2 = 1-exp(-5*norm(V));
end

function Fr3 = impotanceE(e)
% threatening level fitness function
Fr3 = e(3, 1);
end

function Frj = threatBonus(e)
% threatening fitness of evader
% p, e refer to pursuer and evader respectively
% tau refers to Weighted coefficients,
tau = [13 13 13];
Frj = tau(1)*threatrangeFit(e)+tau(2)*velE(e)+tau(3)*impotanceE(e);
end

function Sij = probabilityIJ(p, e)
% predicted interception probability of persuer i to evader j
lmd1 = 1;
lmd2 = 1;
omg1 = 0.5;
omg2 = 0.5;

Rap = p(3, 1); % effective detective distance
Trp = p(3, 2); % maximum detective distance
temp = p - e;
R = temp(1, :);  % range vector
dij = norm(R);
temp2 = e - p;
temp3 = p - e;
rij = temp2(1, :);
rji = temp3(1, :);
vi = p(2, :); % vector of pursuer
vj = e(2, :); % vector of evader
eij = acos(dot(rij, vi)/norm(vi)/norm(rij)); % pre-angle of velocity
eji = acos(dot(rji, vj)/norm(vj)/norm(rji)); % pre-angle of velocity
% coefficient of range threatening
if dij <= Rap
    Sdij = 1;
elseif dij < Trp
    Sdij = 1-(dij-Rap)/(Trp-Rap);
else
    Sdij = 0;
end
% coefficient of angle threatening
Seij = exp(-lmd1*(eij+eji)^lmd2);
% coefficient of velocity threatening
if norm(vj) < 0.2
    Svpi = 1;
elseif norm(vj) <= 3
    Svpi = -0.4*norm(vj)+1.2;
else
    Svpi = 0;
end

Sij = omg1*Sdij*Seij+omg2*Svpi; % interception probability
end

function F = fitnessf(Dori, P, E)
% fitness function
% input: D, distribute matrix
% output: fitness value
[m, n] = size(Dori); % m refer to row, n refer to column
Fsr = 0;
for j = 1:n
    TTsij = 1;
    for i = 1:m
        Sij = probabilityIJ(squeeze(P(i, :, :)), squeeze(E(j, :, :))) * Dori(i, j); % core inovation
        TTsij = TTsij * (1 - Sij);
    end
    Ssj = 1 - TTsij;
    if Ssj == 0
        Ssj = -3;
    end
    Fsr = Fsr + threatBonus(squeeze(E(j, :, :)))*Ssj;
end
F = Fsr;
end

function D = decode(Dori)
% decoding function: get standard distribute matrix by decoding original distribute matrix
% original distribute matrix
% standard distribute matrix
D = Dori;
[row, col] = size(D);
for i=1:row
    maxr = max(D(i, :));
    for j=1:col
        if D(i, j) < maxr
            D(i, j) = 0;
        else
            D(i, j) = 1;
        end
    end
end
end

%% V. VNS functions
function K = neighborhoodchange(Kori, nth, P, E)
% neighborhoodchange function
% input: Kori refers to initial particle, nth refers to n-th neighborhood structrue
% output: better particle
[row, col] = size(Kori);
N = 3; % max iterations
% first neighborhood structrue: exchange
if nth == 1
    for i = 1:N
        Kn = Kori;
        fori = fitnessf(Kori, P, E);
        rd1 = randi([1 row]); % obtain random integer between 1 and row
        rd2 = randi([1 row]);
        rd3 = randi([1 col]);
        rd4 = randi([1 col]);
        while rd2 == rd1
            rd1 = randi([1 row]);
            rd2 = randi([1 row]);
        end
        while rd4 == rd3
            rd3 = randi([1 col]);
            rd4 = randi([1 col]);
        end
        % exchange 2 rows
        temp = Kn(rd1, :);
        Kn(rd1, :) = Kn(rd2, :);
        Kn(rd2, :) = temp;
        % exchange 2 columns
        temp = Kn(:, rd3);
        Kn(:, rd3) = Kn(:, rd4);
        Kn(:, rd4) = temp;
        fnow = fitnessf(Kn, P, E);
        % whether to update the particle
        if fnow > fori
            K = Kn;
            Kori = Kn;
        else
            K = Kori;
        end
    end
end
% seceond neighborhood structrue: reverse
if nth == 2
    for i = 1:N
        Kn = Kori;
        fori = fitnessf(Kori, P, E);
        rd1 = randi([1 row]); % obtain random integer between 1 and row;
        temp = zeros(1, col);
        for c = 1:col
            temp(c) = Kn(rd1, 1+col-c);
        end
        Kn(rd1, :) = temp;
        fnow = fitnessf(Kn, P, E);
        if fnow > fori
            K = Kn;
            Kori = Kn;
        else
            K = Kori;
        end
    end
end
% third neighborhood structrue: insert
if nth == 3
    for i = 1:N
        Kn = Kori;
        fori = fitnessf(Kori, P, E);
        rd1 = randi([1 row]); % obtain random integer between 1 and row;
        rd2 = randi([1 col]); % obtain random integer between 1 and column;
        temp = zeros(1, col);
        temp(rd2) = 1;
        Kn(rd1, :) = temp; % insert a new vector in it
        fnow = fitnessf(Kn, P, E);
        if fnow > fori
            K = Kn;
            Kori = Kn;
        else
            K = Kori;
        end
    end
end

end

function [D, isupdate] = vns(Dori, P, E)
% vns algorithm
% input: distribute function
% output: upgraded distribute function, whether to update the matrix
Nmax = 3;
Klmax = 3;
isupdate = false;
n = 1;
while n <= Nmax
    k = 1;
    while k <= Klmax
        fori = fitnessf(Dori, P, E);
        Dn = neighborhoodchange(Dori, k, P, E); % proccede k-th neighborhood structrue
        fnow = fitnessf(Dn, P, E);
        if fnow > fori
            Du = Dn;
            Dori = Dn;
            k = 0;
            isupdate = true;
        else
            Du = Dori;
        end
        k = k+1;
    end
    n = n+1;
    
end
D = Du;
end

%% VI.VNS-PSO algorithm
function Result = vnspso(P, E)
% mixed PSO algorithm
% input: information of pursuer and evader
% ouput: distribute matrix

% obtain number of P and E
[numP,temp,temp] = size(P);
[numE,temp,temp] = size(E);
% coefficients
c1 = 1.1931;
c2 = 1.1931;
omg = 0.87;
maxgen = 100;   % number of iterations
sizepop = 50;   % Population size
Vmax = 1;  % limits of velocity
Vmin = -1;
popmax = 1;   % limits of position
popmin = 0;

% original population
for i = 1:sizepop
    % obtain a particle swarm randomly
    popi = popmin + (popmax - popmin)*rand(numP, numE);
    for k = 1:numP
        popi(k,:) = popi(k,:)/sum(popi(k,:));
    end
    pop(i,:,:) = popi;   % original postion
    V(i,:,:) = Vmin + (Vmax - Vmin)*rand(numP, numE);   % original velocity
    % caculate fitness value
    fitness(i) = fitnessf(squeeze(pop(i,:,:)), P, E);
end

% Individual and global extremums of the initial population
[bestfitness bestindex] = max(fitness);
zbest = pop(bestindex,:,:);   % global extremums
gbest = pop;    % Individual extremums
fitnessgbest = fitness;   % Individual extremums fitness value
fitnesszbest = bestfitness;   % global extremums fitness value



% iterations
for i = 1:maxgen
    for j = 1:sizepop
        % velocity update
        V(j,:,:) = omg*V(j,:,:) + c1*rand*(gbest(j,:,:) - pop(j,:,:)) + c2*rand*(zbest - pop(j,:,:));
        
        % position update
        pop(j,:,:) = pop(j,:,:) + V(j,:,:);
        pop(j,find(pop(j,:,:)<0)) = 0;
        popj = squeeze(pop(j,:,:));
        for k = 1:numP
            popj(k,:) = popj(k,:)/sum(popj(k,:));
        end
        pop(j,:,:) = popj;
        
        % caculate fitness value
        fitness(j) = fitnessf(squeeze(pop(j,:,:)), P, E);
        
        Fhistory(i, j) = fitness(j);
        
        
    end
    for j = 1:sizepop
        % individual extremum update
        if fitness(j) > fitnessgbest(j)
            gbest(j,:,:) = pop(j,:,:);
            fitnessgbest(j) = fitness(j);
        end
        
        % global extremum update
        if fitness(j) > fitnesszbest
            zbest = pop(j,:,:);
            fitnesszbest = fitness(j);
        end
    end
    % vns strategy
    for j = 1:sizepop
        popj = squeeze(pop(j,:,:));
        if i > 3
            if (Fhistory(i, j) - Fhistory(i-1, j)) < 1e-4
                if (Fhistory(i, j) - Fhistory(i-2, j)) < 1e-4
                    if mod(i, 15) == 0
                        if j == 1
                            disp('vns going');
                        end
                        omg = 0.4;
                        % vns strategy
                        [popjn, isupdate] = vns(popj, P, E);
                        pop(j,:,:) = popjn;
                        tmptfitness = fitnessf(popjn, P, E);
                        % individual extremum update
                        if isupdate
                            gbest(j,:,:) = popjn;
                            fitnessgbest(j) = tmptfitness;
                        end
                        % global extremum update
                        if tmptfitness > fitnesszbest
                            zbest = pop(j,:,:);
                            fitnesszbest = tmptfitness;
                        end
                    end
                end
            end
        end
        
    end
    
    yy(i) = fitnesszbest; % fitness value each iteration
    disp(i);
    disp(squeeze(pop(1, :, :)));
end

%save('bestfitness.mat','yy');
% figure;
% plot(yy);
% figure;
% plot(Fhistory(:,1));
% xlabel('iterations');
% ylabel('fitness of one single particle');
% disp(Fhistory(:,1)');
Result = yy(maxgen);
disp('ultimate distribute matrix:');
disp(squeeze(zbest));
disp('ultimate fitness value:');
disp(Result);

end
