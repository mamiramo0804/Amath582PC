%% HW 5
% AMATH 582 WI 2021
% Marissa Miramontes
% 03/17/21

clear
clc
close all
%% load the low-res movies
mov1info = VideoReader('ski_drop_low.mp4');
mov2info = VideoReader('monte_carlo_low.mp4');

%%
mov1 = read(mov1info);
mov2 = read(mov2info);

%% convert to gray mat
n1 = mov1info.NumFrames;
T1 = mov1info.Duration;
t1 = linspace(1,T1,n1);

n2 = mov2info.NumFrames;
T2 = mov2info.Duration;
t2 = linspace(1,T2,n2);

for i = 1:n1
    MovMat1(:,:,i) = double(rgb2gray(mov1(:,:,:,i)));
end

for i = 1:n2
    MovMat2(:,:,i) = double(rgb2gray(mov2(:,:,:,i)));
end


%% Data mat X

[row1, col1, fr1] = size(MovMat1);
X1 = zeros(row1*col1,n1);
for i = 1:n1
    frame = MovMat1(:,:,i);
   X1(:,i) =  frame(:);
end

[row2, col2, fr2] = size(MovMat2);
X2 = zeros(row2*col2,n2);
for i = 1:n2
    frame = MovMat2(:,:,i);
   X2(:,i) =  frame(:);
end

%% Contrust data matrices X_M-1 and X_M
% each column is a frame, rows are pixels
% choosing which movie for DMD
t = t2;
X = X2;
X_1 = X(:,1:end-1);
X_2 = X(:,2:end);

%% SVD of X_M-1
[U,S,V] = svd(X_1,'econ');

%% look a subgular values to determine rank r (number of cols of U to use)

N = 50; 
% first three modes account for 75% of the total energy, less than 1% for
% modes past 5
SingVec = diag(S(1:N,1:N));
SinPerc = SingVec/sum(diag(S))*100;
plot(1:N,SinPerc,'o','LineWidth',1)
% semilogy(diag(S)/sum(diag(S)),'o','LineWidth',1)
xlabel('Singular Values')
ylabel('Percent Energy')
sum(SinPerc)


%% Computing Rank-r Matrix A
r = 50; %50; % 89.3% total energy for ski % 50 -> 76.2% for car movie

Ur = U(:,1:r);
Sr = S(1:r,1:r);
Vr = V(:,1:r);

A = Ur'*X_2*Vr/Sr;

%% find eig-vec and vals of rank-r matrix A
[W,D] = eig(A);

%% to get back to high dim space

Phi = X_2*Vr/Sr*W; % DMD modes
lam = diag(D); % = exp(omega*dt)
dtdiff = diff(t);
dt = dtdiff(1);
omega = log(lam)/dt;

%% reconstruction
x1 = X(:,1); % x(t=0) = first frame
b = Phi\x1; % b can be optimized

time_dyn = zeros(r,length(t));
for i = 1:length(t)
   time_dyn(:,i) = (b.*exp(omega*t(i)));
end
X_dmd = Phi*time_dyn;
%% plotting recontructed movie
% look at real

% mov1 (ski)
% for i = 1:n1
% imagesc(real(reshape(X_dmd(:,i),row1,col1)))
% title(num2str(i))
% pause(0.1)
% 
% end

% The movie appears to be a still image of the mountain

% mov1 (ski)
for i = 1:n2
imagesc(real(reshape(X_dmd(:,i),row1,col1)))
title(num2str(i))
pause(0.1)

end

% car movie: you can see a littel swoosh in the begining bot otherwise it
% looks like the background the whole time
%% Plotting Omega vec
plot(abs(omega),'o','LineWidth',1)
%% backgorund will be the mode with eigen value closest to 0?
% For only looking at first 50,  Let's choose the 40th omega values (=0.004)
[M,I] = min(abs(omega));
minOmega = I;
Xdmd_BG = b(minOmega)*Phi(:,minOmega)*exp(omega(minOmega)*t(minOmega)); % background

%% the remaining modes make up the forground?
bf = b; 
bf(minOmega) = [];
Phif = Phi;
Phif(:,minOmega) = [];
omegaf = omega;
omegaf(minOmega) = [];
time_dynf = zeros(r-1,length(t));
for i = 1:length(t)
   time_dynf(:,i) = (bf.*exp(omegaf*t(i)));
end
Xdmd_FG = Phif*time_dynf;
%
%% Movie of Ski FG
for i = 1:n1
imagesc(real(reshape(Xdmd_FG(:,i),row1,col1)))
title(num2str(i))
pause(0.1)

end
%% Back ground Image
imagesc(real(reshape(Xdmd_BG,row1,col1)))

%% Subtracting FG from BG
Xlowrank = Xdmd_BG; 
Xsparse = X_dmd - abs(Xlowrank); % X sparse should corrisponde to Xdmd_FG
Rmask = double(Xsparse < 0);
R = Xsparse.*Rmask;
Xdmd_BG_R = abs(Xlowrank) + R ;
Xdmd_FG_R = Xsparse - R;

%% Plotting these new Xdmds
close all
% for i = 1:n1
% imagesc(real(reshape(Xdmd_FG_R(:,i),row1,col1)))
% title(num2str(i))
% colormap('gray')
% 
% pause(0.1)
% end

for i = 1:n2
imagesc(real(reshape(Xdmd_FG_R(:,i),row1,col1)))
title(num2str(i))
colormap('gray')

pause(0.1)
end

imagesc(real(reshape(Xdmd_FG_R(:,235),row1,col1)))
title(num2str(i))
colormap('gray')

%% background image

close all
for i = 1:n2
imagesc(real(reshape(Xdmd_BG_R(:,i),row1,col1)))
title(num2str(i))
colormap('gray')

pause(0.1)
end