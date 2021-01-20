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
k = (2*pi/(2*L))*[0:(n/2 - 1) -n/2:-1]; ks = fftshift(k); % freqencies (wave numbers) scaled

[X,Y,Z] = meshgrid(x,y,z);
[Kx,Ky,Kz] = meshgrid(ks,ks,ks);

%% Initial Plotting
% for j=1:49
%     Un(:,:,:)=reshape(data(:,j),n,n,n); % must there always be the same number of spacial points per dim as n?
%     M = max(max(max(abs(Un))));
%     close all, isosurface(X,Y,Z,abs(Un)/M,0.7)
%     axis([-20 20 -20 20 -20 20]), grid on, drawnow
%     pause(1)
% end


%% Reshaping into a 3D matric evolving in time
N = 48; % number of realizations
clear Un M

for j=1:N
    Un(:,:,:,j)=reshape(data(:,j),n,n,n); 
end

% M = max(max(max(abs(Un))));
% Each realization of Un is a noisy comlex singal located somewhere in x,y,z space

%% Plot of Un
M = squeeze(max(max(max(max(abs(Un))))));

for j=1:N
    close all, isosurface(X,Y,Z,abs(Un(:,:,:,j))/M,0.7)
    axis([-20 20 -20 20 -20 20]), grid on, drawnow
    pause(1)
end
%% Determining frequency signature through spectrom averaging

Unt = fftn(Un);  % FT of Un

Untave = mean(fftshift(Unt),4);

% need to find corresping kx ky kz to create filter.

% might need to do some shifting of K's???

MaxVal = max(max(max(abs(Untave))));

Diffmat = abs(abs(Untave)- MaxVal);

SmallVal = min(min(min(Diffmat))); % = 0

% location of peak of avarage signal in the shifted frequency domain
KxMax = Kx(abs(Diffmat) == SmallVal);
KyMax = Ky(abs(Diffmat) == SmallVal);
KzMax = Kz(abs(Diffmat) == SmallVal);


%% Plot of Unt
M = squeeze(max(max(max(max(abs(Unt))))));

for j = 1:N% not sure why some do not show up at all using isosurface
    close all, isosurface(X,Y,Z,abs(fftshift(abs(Unt(:,:,:,j))))/M,0.7)
    axis([-20 20 -20 20 -20 20]), grid on, drawnow
    pause(1)
end
%% Plot of Untave

    close all, isosurface(X,Y,Z,abs(Untave)/MaxVal,0.7)
    axis([-20 20 -20 20 -20 20]), grid on, drawnow
    pause(1)

%% Filter

% making 3d Gaussian function
sig = 1; % uniform sigma in x =, y and z
mu_x = KxMax; mu_y = KyMax; mu_z = KzMax;

Gaus3DFilt = @(x,y,z) (1/(2*pi*sig)^(3/2))*exp(-( ...
((x-mu_x).^2 + (y-mu_y).^2 + (z-mu_z).^2)/(2*sig^2))); 

KMaxFilt = Gaus3DFilt(Kx,Ky,Kz);
% Unt_norm = Unt/Untmax; Not sure if Unt should be normalized when applying
% the filter

Unts = fftshift(Unt);

for i = 1:N
Uts(:,:,:,i) = Unts(:,:,:,i).*KMaxFilt;
end

U = ifft(ifftshift(Uts));
 
%% Plotting Uts
for j=1:N
    close all, isosurface(X,Y,Z,abs(abs(Uts(:,:,:,j)))/M,0.7)
    axis([-20 20 -20 20 -20 20]), grid on, drawnow
    pause(1)
end

%% Trajectory Coordinates

% To plot a point for each realizaion I will look at only the peak value
% I will test this for the noisy data as well...

% I x value for each of the 48 realizations

for i = 1:N
MaxUVal = max(max(max(abs(U(:,:,:,i)))));
DiffUmat = abs(abs(U(:,:,:,i)) - MaxUVal);
SmallUVal = min(min(min(DiffUmat)));


Xvec(i) = X(abs(DiffUmat) == SmallUVal);
Yvec(i) = Y(abs(DiffUmat) == SmallUVal);
Zvec(i) = Z(abs(DiffUmat) == SmallUVal);
end

%% Plotting

plot3(Xvec,Yvec,Zvec) % no bueno - currently looks better with Un

% something might be off with needed to shift the filter?
% try isosurface to look at signal after filter to compare to before
% read around page 314 in book




%% x and y coordinates for aircraft


