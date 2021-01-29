%% HW 1

% looking for unknown signal
% data obtained over 24 hours in half hour increments -> 48 realizations
% source of signal is moving
% need to determine location and path by using the acoustic signature and
% identifying the acoustic admissions.

%% loading data

clear; close all; clc
datast = load('subdata.mat');
data = datast.subdata; % space by time matrix

%% Setup

L = 10; % spatial (10 and -10 are the max and min x-values)
n = 64; % Fourier modes (# of sinusiods to be added together (# of freqncies) to represent the signal)

x2 = linspace(-L,L,n+1); x = x2(1:n); y=x; z=x;
k = (2*pi/(2*L))*[0:(n/2-1) -n/2:-1]; % wave numers (cos(nx)) scaled
ks = fftshift(k); % freqencies (wave numbers) shifted for graphing

[X,Y,Z] = meshgrid(x,y,z);
[Ksx,Ksy,Ksz] = meshgrid(ks,ks,ks);
[Kx,Ky,Kz] = meshgrid(k,k,k);


%% Reshaping into a 3D matric evolving in time
N = 49; % number of realizations
clear Un M

for j=1:N
    Un(:,:,:,j)=reshape(data(:,j),n,n,n);
end
% Each realization of Un is a noisy comlex singal located somewhere in x,y,z space

%% Plot of Un
% M = squeeze(max(max(max(max(abs(Un))))));
%
% for j=1:N
%     close all, isosurface(X,Y,Z,abs(Un(:,:,:,j))/M,0.7)
%     axis([-20 20 -20 20 -20 20]), grid on, drawnow
%     pause(1)
% end
%% Determining frequency signature through spectrum averaging

Unt = zeros(size(Un));  % FT of Un

Untaveit = zeros(size(Un(:,:,:,1)));
for i = 1:N
    Unt(:,:,:,i) = fftn(Un(:,:,:,i));  % FT of Un for each realization
    Untaveit = Untaveit + Unt(:,:,:,i);
end
Untave = Untaveit/N;

% need to find corresping kx ky kz to create filter.
%% Finding corner frequency
MaxVal = max(abs(Untave(:)));
th = 0.85;
nvals = length(find((abs(Untave) >= th*MaxVal)));
[I1,I2,I3] = ind2sub(size(Untave), find((abs(Untave) >= th*MaxVal)));  % I1 I2 I3 are vectors of the locations

for i = 1:nvals
    kxvals(i) = k(I2(i));
    kyvals(i) = k(I1(i));
    kzvals(i) = k(I3(i));
end
KxMax = mean(kxvals);
KyMax = mean(kyvals);
KzMax = mean(kzvals);


%% Plot of Unt
% M = squeeze(max(max(max(max(abs(Unt))))));
%
% for j = 1:N% not sure why some do not show up at all using isosurface
%     M = squeeze(max(max(max(abs(Unt(:,:,:,j))))));
%     close all, isosurface(Kx,Ky,Kz,abs(fftshift(Unt(:,:,:,j)))/M,0.7)
%     axis([-20 20 -20 20 -20 20]), grid on, drawnow
%     pause(1)
% end
%% Plot of Untave

close all, isosurface(Ksx,Ksy,Ksz,abs(fftshift(Untave))/MaxVal,0.7)
axis([-20 20 -20 20 -20 20]), grid on, drawnow
pause(1)



%% Filter

% making 3d Gaussian function
sig = 0.2; % uniform sigma in x =, y and z
mu_x = KxMax; mu_y = KyMax; mu_z = KzMax;

% Gaus3DFilt = @(x,y,z) (1/(2*pi*sig)^(3/2))*exp(-( ...
% ((x-mu_x).^2 + (y-mu_y).^2 + (z-mu_z).^2)/(2*sig^2)));

Gaus3DFilt = @(x,y,z) exp(-sig*((x-mu_x).^2 + (y-mu_y).^2 + (z-mu_z).^2));

KMaxFilt = Gaus3DFilt(Kx,Ky,Kz);
% Unt_norm = Unt/Untmax; Not sure if Unt should be normalized when applying
% the filter

Unts = fftshift(Unt);

% setting up
Ut = zeros(size(Unts));
U = Ut;

FiltMax = max(KMaxFilt(:));
KMaxFiltNorm = KMaxFilt/FiltMax;

for i = 1:N
    Ut(:,:,:,i) = Unt(:,:,:,i).*KMaxFiltNorm;
    U(:,:,:,i) = ifftn(Ut(:,:,:,i));
end

Utave = fftshift(Untave).*KMaxFiltNorm;

%% Plot of Utave (filtered average)
%
%     close all, isosurface(Kx,Ky,Kz,abs(Utave),0.7)
%     axis([-20 20 -20 20 -20 20]), grid on, drawnow
%     pause(1)
%% Plotting Guassian Filter

xslice = KxMax;                               % define the cross sections to view
yslice = KyMax;
zslice = KzMax;

slice(Ksx, Ksy, Ksz, fftshift(KMaxFilt), xslice, yslice, zslice)
colorbar
% display the slices
% ylim([-3 3])
view(-34,24)

%% Plotting Uts
% clear M
%
% M = max(abs(Uts(:)));
%
% for j=1:N
%     close all, isosurface(Kx,Ky,Kz,abs(Uts(:,:,:,j)),0.7)
%     axis([-L L -L L -L L]), grid on, drawnow
%     pause(1)
% end

%% plotting U

% % should see a plob moving around
% for j=1:N
%     close all, isosurface(X,Y,Z,abs(U(:,:,:,j)));
%     axis([-L L -L L -L L]), grid on, drawnow
%     pause(1)
% end


%% Trajectory Coordinates

% To plot a point for each realizaion I will look at only the peak value

for i = 1:N
    Ui = U(:,:,:,i);
    MaxUVal = max(abs(Ui(:)));
    
    [J1,J2,J3] = ind2sub(size(Ui), find(abs(Ui) == MaxUVal));
    Xvec(i) = x(J2);
    Yvec(i) = y(J1);
    Zvec(i) = z(J3);
    
    
end

%% Trajectory Coordinates Un
clear Xvec Yvec Zvec
for i = 1:N
    Ui = Un(:,:,:,i);
    MaxUVal = max(abs(Ui(:)));
    
    DiffUmat = abs(abs(U(:,:,:,i)) - MaxUVal);
    SmallUVal = min(min(min(DiffUmat)));
    Xvec(i) = X(abs(DiffUmat) == SmallUVal);
    Yvec(i) = Y(abs(DiffUmat) == SmallUVal);
    Zvec(i) = Z(abs(DiffUmat) == SmallUVal);
    
    [J1,J2,J3] = ind2sub(size(Ui), find(abs(Ui) == MaxUVal));
    Xvec(i) = x(J2);
    Yvec(i) = y(J1);
    Zvec(i) = z(J3);
    
    
end

%% Plotting

plot3(Xvec,Yvec,Zvec, 'LineWidth', 2)
axis([-L L -L L -L L])

hold on
plot3(Xvec(end),Yvec(end),Zvec(end),'*','LineWidth',3)


%% x and y coordinates for aircraft

xfinal = Xvec(end);
yfinal = Yvec(end);

%% Trajectory Figure for report
clear Xvec Yvec Zvec
figure(1)
subplot(1,2,1) % plotting of Un
for i = 1:N
    Ui = Un(:,:,:,i);
    MaxUVal = max(abs(Ui(:)));
    [P1,P2,P3] = ind2sub(size(Ui), find(abs(Ui) == MaxUVal));
    Xvec0(i) = x(P2);
    Yvec0(i) = y(P1);
    Zvec0(i) = z(P3);
end
plot3(Xvec0,Yvec0,Zvec0, 'LineWidth', 2)
hold on
plot3(Xvec0(end),Yvec0(end),Zvec0(end),'*','LineWidth',3)
axis([-L L -L L -L L])
xlabel('x')
ylabel('y')
zlabel('z')
title('(a) Trajectory from unfiltered data')


subplot(1,2,2) % plotting of U
for i = 1:N
    Ui = U(:,:,:,i);
    MaxUVal = max(abs(Ui(:)));
    [J1,J2,J3] = ind2sub(size(Ui), find(abs(Ui) == MaxUVal));
    Xvec(i) = x(J2);
    Yvec(i) = y(J1);
    Zvec(i) = z(J3);
end
plot3(Xvec,Yvec,Zvec, 'LineWidth', 2)
hold on
plot3(Xvec(end),Yvec(end),Zvec(end),'*','LineWidth',3)
axis([-L L -L L -L L])
xlabel('x')
ylabel('y')
zlabel('z')
title('(b) Trajectory from filtered data')

%% Filtering Figure from report
close all,
figure(2)

subplot(1,2,1)

isosurface(Ksx,Ksy,Ksz,abs(fftshift(Untave))/MaxVal,0.7)
axis([-L L -L L -L L]), grid on, drawnow
xlabel('kx')
ylabel('ky')
zlabel('kz')
title('(a) Average of noisy frequency data')


subplot(1,2,2)
xslice = KxMax;
yslice = KyMax;
zslice = KzMax;
slice(Ksx, Ksy, Ksz, fftshift(KMaxFilt), xslice, yslice, zslice)
axis([-L L -L L -L L])
colorbar
xlabel('kx')
ylabel('ky')
zlabel('kz')
title('(b) Guassian filter with mean k_{0}')