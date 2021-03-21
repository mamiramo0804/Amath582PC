%% AMATH 582 Final Project
clear
clc
close all
%% load data

dataLoad = load('Sensor2020data.mat');
dataset = dataLoad.SenDataStr;

%% Measured variables 
% for now I'm just going to play around with one data set:
% Pulstatile pump on, Lavare cycle on, at 2100 rpm, HVAD
% t - time (s)
% Q - LVAD flow rate (mLPM)
% P1 - Pressure before LVAD (mmHg)
% P2 - Pressure after LVAD (mmHg)
% pump - Voltage of puslatile piston pump (V)
% trig - Voltage of camera trigger (V)

%% Seperating out data sets
% These data sets look intact (the signal doesn't drop out to 0) and don't need cropping
% P = pulsatile pump on/off
% L = Larare Cycle on/off (occurs every 60s for 3 seconds): baseline-200 rpm for 2
% sec, baseline+200 rpm for 1 second

% 2100 HVAD
data2100(1) = dataset(51); % P L
data2100(2) = dataset(52); % P noL
data2100(3) = dataset(53); % noP L
data2100(4) = dataset(54); % noP noL

% 2440 HVAD
data2440(1) = dataset(55); % P L
data2440(2) = dataset(56); % P noL
data2440(3) = dataset(57); % noP L
data2440(4) = dataset(58); % noP noL

% 2800 HVAD
data2800(1) = dataset(59); % P L
data2800(2) = dataset(60); % P noL
data2800(3) = dataset(61); % noP L
data2800(4) = dataset(62); % noP noL
%% Plotting them all to see what the signals look like
% Some of them the singal drops out to 0s or needs cropping.

% close all
% figure(1)
% saveOpt = 0;
% for i = 1:length(dataset)
%     data = dataset(i);
%     t = data.t;
%     Q = data.Q;
%     dP = data.P2-data.P1;   
%     
%     subplot(1,2,1)
%     plot(t,Q)
%     title('Flow Rate')
%     subplot(1,2,2)
%     plot(t,dP)
%     title('dP')
%     filename = ['SensorRawDataFigs_' int2str(i) '.png'];
%     if saveOpt
%     saveas(gca,filename);
%     end
% end

%% Choosing just one signal to work with
clear Un
N = 1; % chose which conditions
Un(1,:) = data2100(N).Q;  
Un(2,:) = data2100(N).P1;  
Un(3,:) = data2100(N).P2;  
t = data2100(N).t;   
% Un = data2100(N).Q(1:end-1);  
% t = data2100(N).t(1:end-1);    
%%
ymax = 5;
ymin = 1;

plot(t,Un)
ylim([ymin ymax])
%% Fequency analysis
n = length(t);
L  = t(end); % sec
k = (1/L)*[0:n/2-1 -n/2:-1];%(2*pi/L)*[0:n/2-1 -n/2:-1];
ks = fftshift(k);
for i = 1:3
Unt(i,:) = fft(Un(i,:));
end
%% frequncy filtering
% Does it make sense that I look at multiple singnals to find a mean or key
% freqencies. I should look at multiple signals with and without lavare
% and look for differencies.

% Frequency filtering about different max k vals?
% Try filtered about k largest?
sig = 10; %10; % smaller -> wider
num = 7; % identified by looking at the number of peaks

Unt = Unt(1,:); % Q 

[Utmax,I] = maxk(Unt,num);

% building 1d Gaussian function
if num == 1
    mu_x = k(I);
    Gaus1DFilt = @(x) exp(-sig*((x-mu_x).^2));
    
    KMaxFilt = Gaus1DFilt(k);
elseif num == 3
    mu_x1 = k(I(1));
    Gaus1DFilt1 = @(x) exp(-sig*((x-mu_x1).^2));
    mu_x2 = k(I(2));
    Gaus1DFilt2 = @(x) exp(-sig*((x-mu_x2).^2));
    mu_x3 = k(I(3));
    Gaus1DFilt3 = @(x) exp(-sig*((x-mu_x3).^2));
   
    KMaxFilt = Gaus1DFilt1(k)+Gaus1DFilt2(k)+Gaus1DFilt3(k);
elseif num == 5
    mu_x1 = k(I(1));
    Gaus1DFilt1 = @(x) exp(-sig*((x-mu_x1).^2));
    mu_x2 = k(I(2));
    Gaus1DFilt2 = @(x) exp(-sig*((x-mu_x2).^2));
    mu_x3 = k(I(3));
    Gaus1DFilt3 = @(x) exp(-sig*((x-mu_x3).^2));
    mu_x4 = k(I(4));
    Gaus1DFilt4 = @(x) exp(-sig*((x-mu_x4).^2));
    mu_x5 = k(I(5));
    Gaus1DFilt5 = @(x) exp(-sig*((x-mu_x5).^2));
    
    
    KMaxFilt0 = Gaus1DFilt1(k)+Gaus1DFilt2(k)+Gaus1DFilt3(k)+Gaus1DFilt4(k)+Gaus1DFilt5(k);
    KMaxFilt23 = Gaus1DFilt1(k)+Gaus1DFilt2(k)+Gaus1DFilt3(k);
    KMaxFilt45 = Gaus1DFilt1(k)+Gaus1DFilt4(k)+Gaus1DFilt5(k);
    
    
    
   elseif num == 7
    mu_x1 = k(I(1));
    Gaus1DFilt1 = @(x) exp(-sig*((x-mu_x1).^2));
    mu_x2 = k(I(2));
    Gaus1DFilt2 = @(x) exp(-sig*((x-mu_x2).^2));
    mu_x3 = k(I(3));
    Gaus1DFilt3 = @(x) exp(-sig*((x-mu_x3).^2));
    mu_x4 = k(I(4));
    Gaus1DFilt4 = @(x) exp(-sig*((x-mu_x4).^2));
    mu_x5 = k(I(5));
    Gaus1DFilt5 = @(x) exp(-sig*((x-mu_x5).^2));
    mu_x6 = 2.9;
    Gaus1DFilt6 = @(x) exp(-sig*((x-mu_x6).^2));
    mu_x7 = -2.9;
    Gaus1DFilt7 = @(x) exp(-sig*((x-mu_x7).^2));
    
    KMaxFilt0 = Gaus1DFilt1(k)+Gaus1DFilt2(k)+Gaus1DFilt3(k)+Gaus1DFilt4(k)+Gaus1DFilt5(k);
    KMaxFilt23 = Gaus1DFilt1(k)+Gaus1DFilt2(k)+Gaus1DFilt3(k);
    KMaxFilt45 = Gaus1DFilt1(k)+Gaus1DFilt4(k)+Gaus1DFilt5(k);
    KMaxFilt67 = Gaus1DFilt1(k)+Gaus1DFilt6(k)+Gaus1DFilt7(k);
    elseif num == 9
    mu_x1 = k(I(1));
    Gaus1DFilt1 = @(x) exp(-sig*((x-mu_x1).^2));
    mu_x2 = k(I(2));
    Gaus1DFilt2 = @(x) exp(-sig*((x-mu_x2).^2));
    mu_x3 = k(I(3));
    Gaus1DFilt3 = @(x) exp(-sig*((x-mu_x3).^2));
    mu_x4 = k(I(4));
    Gaus1DFilt4 = @(x) exp(-sig*((x-mu_x4).^2));
    mu_x5 = k(I(5));
    Gaus1DFilt5 = @(x) exp(-sig*((x-mu_x5).^2));
    mu_x6 = k(I(6));
    Gaus1DFilt6 = @(x) exp(-sig*((x-mu_x6).^2));
    mu_x7 = k(I(7));
    Gaus1DFilt7 = @(x) exp(-sig*((x-mu_x7).^2));
    mu_x8 = k(I(8));
    Gaus1DFilt8 = @(x) exp(-sig*((x-mu_x8).^2));
    mu_x9 = k(I(9));
    Gaus1DFilt9 = @(x) exp(-sig*((x-mu_x9).^2));
    
    KMaxFilt0 = Gaus1DFilt1(k)+Gaus1DFilt2(k)+Gaus1DFilt3(k)+Gaus1DFilt4(k)+Gaus1DFilt5(k)+Gaus1DFilt6(k)+Gaus1DFilt7(k)+Gaus1DFilt8(k)+Gaus1DFilt9(k);
    KMaxFilt23 = Gaus1DFilt1(k)+Gaus1DFilt2(k)+Gaus1DFilt3(k);
    KMaxFilt45 = Gaus1DFilt1(k)+Gaus1DFilt4(k)+Gaus1DFilt5(k);
    KMaxFilt67 = Gaus1DFilt1(k)+Gaus1DFilt6(k)+Gaus1DFilt7(k);
    KMaxFilt89 = Gaus1DFilt1(k)+Gaus1DFilt8(k)+Gaus1DFilt9(k);
end

% apply filter
FiltMax0 = max(KMaxFilt0(:));
FiltMax23 = max(KMaxFilt23(:));
FiltMax45 = max(KMaxFilt45(:));
FiltMax67 = max(KMaxFilt67(:));
FiltMax89 = max(KMaxFilt89(:));


KMaxFiltNorm0 = KMaxFilt0/FiltMax0;
KMaxFiltNorm23 = KMaxFilt23/FiltMax23;
KMaxFiltNorm45 = KMaxFilt45/FiltMax45;
KMaxFiltNorm67 = KMaxFilt67/FiltMax67;

KMaxFiltNorm89 = KMaxFilt45/FiltMax89;

Ut0 = Unt.*KMaxFiltNorm0;
Ut23 = Unt.*KMaxFiltNorm23;
Ut45 = Unt.*KMaxFiltNorm45;
Ut67 = Unt.*KMaxFiltNorm67;

Ut89 = Unt.*KMaxFiltNorm89;

Ufilt0 = ifftn(Ut0);
Ufilt23 = ifftn(Ut23);
Ufilt45 = ifftn(Ut45);
Ufilt67 = ifftn(Ut67);

Ufilt89 = ifftn(Ut89);

%% Orignal filtering
% sig = 10; %10; % smaller -> wider num = 3; % identified by looking at the
% number of peaks
%
% for i = 1:3 [Utmax,I] = maxk(Unt(i,:),num);
%
% % building 1d Gaussian function if num == 1
%     mu_x = k(I); Gaus1DFilt = @(x) exp(-sig*((x-mu_x).^2));
%
%     KMaxFilt = Gaus1DFilt(k);
% elseif num == 3
%     mu_x1 = k(I(1)); Gaus1DFilt1 = @(x) exp(-sig*((x-mu_x1).^2)); mu_x2 =
%     k(I(2)); Gaus1DFilt2 = @(x) exp(-sig*((x-mu_x2).^2)); mu_x3 =
%     k(I(3)); Gaus1DFilt3 = @(x) exp(-sig*((x-mu_x3).^2));
%
%     KMaxFilt = Gaus1DFilt1(k)+Gaus1DFilt2(k)+Gaus1DFilt3(k);
% elseif num == 5
%     mu_x1 = k(I(1)); Gaus1DFilt1 = @(x) exp(-sig*((x-mu_x1).^2)); mu_x2 =
%     k(I(2)); Gaus1DFilt2 = @(x) exp(-sig*((x-mu_x2).^2)); mu_x3 =
%     k(I(3)); Gaus1DFilt3 = @(x) exp(-sig*((x-mu_x3).^2)); mu_x4 =
%     k(I(4)); Gaus1DFilt4 = @(x) exp(-sig*((x-mu_x4).^2)); mu_x5 =
%     k(I(5)); Gaus1DFilt5 = @(x) exp(-sig*((x-mu_x5).^2));
%
%
%         KMaxFilt =
%         Gaus1DFilt1(k)+Gaus1DFilt2(k)+Gaus1DFilt3(k)+Gaus1DFilt4(k)+Gaus1DFilt5(k);
%
%
% end
% 
% % apply filter FiltMax = max(KMaxFilt(:)); KMaxFiltNorm =
% KMaxFilt/FiltMax; Ut = Unt(i,:).*KMaxFiltNorm;
% 
% Ufilt(i,:) = ifftn(Ut);
% 
% end
%% plotting filter
close all
figure(1)
Filt = KMaxFiltNorm0;
plot(ks,fftshift(abs(Unt))/max(abs(Unt(:))),ks,fftshift(abs(Filt))/max(abs(Filt(:))),'LineWidth',(1));
xlim([-3 3])
ylabel('FFT Weights')
xlabel('k (Hz)');

figure(2)
Filt = KMaxFiltNorm45;
plot(ks,fftshift(abs(Unt))/max(abs(Unt(:))),ks,fftshift(abs(Filt))/max(abs(Filt(:))),'LineWidth',(1));
xlim([-3 3])
ylabel('FFT Weights')
xlabel('k (Hz)');

figure(3)
Filt = KMaxFiltNorm67;
plot(ks,fftshift(abs(Unt))/max(abs(Unt(:))),ks,fftshift(abs(Filt))/max(abs(Filt(:))),'LineWidth',(1));
xlim([-3 3])
ylabel('FFT Weights')
xlabel('k (Hz)');
% hold on plot(ks,fftshift(KMaxFiltNorm),'LineWidth',(1)); xlim([-20 20])
% hold off xlabel('k') ylabel('FFT')
%% plotting filtered signal
% plot(t,Un(1,:),t,Un(2,:),t,Un(3,:))
% hold on
% plot(t,abs(Ufilt(1,:)),abs(Ufilt(2,:)),abs(Ufilt(3,:)),'LineWidth',(2))
% ylim([ymin ymax])
% hold off
% legend('Un spec','Filter');


% I would like a plot of all the fitered signals but this looks wierd
% Ufilt =  Ufilt45;
Ufilt =  Ufilt67;

close all
figure(1)
plot(t,Un(1,:)); %,'LineWidth',(1))
hold on
plot(t,abs(Ufilt0),'LineWidth',(2))
hold off
xlabel('time(s)')
ylabel('Flow Rate (L/min)')

figure(2)
plot(t,Un(1,:)); %,'LineWidth',(1))
hold on
plot(t,abs(Ufilt45),'LineWidth',(2))
hold off
xlabel('time(s)')
ylabel('Flow Rate (L/min)')

figure(3)
plot(t,Un(1,:)); %,'LineWidth',(1))
hold on
plot(t,abs(Ufilt67),'LineWidth',(2))
hold off
xlabel('time(s)')
ylabel('Flow Rate (L/min)')
% close all
% figure(1)
% plot(t,xn1,'LineWidth',(1))
% hold on
% yyaxis('right')
% plot(t,xn2,t,xn3,'k','LineWidth',(1))
% hold off
% legend('Flow Rate','Pressure 1', 'Pressure 2')

% SHOULD APPLY TO GABOR
%% spectrogram
% to try to find when the signal frequency might be changing.
f = 100;
% td =t,f);
% xnd1 = downsample(xn1,f);
yg = downsample(data2100(3).Q,f);
tg =  downsample(data2100(3).t,f);
a = 5; % for gabor filter - straking cirtucal past 10 and steaking horiz less than 1
filtOpt = 0;
numks = 5;
tstep = 0.01; % sec
Filt = Ufilt23; % Freq Filt to be slid in time
[ygfin, kgabor, tslide] = gaborfiltv2(yg,tg,a,tstep,filtOpt,Filt);

%% plot spectrogram
close all
pcolor(tslide,fftshift(kgabor),log(abs(ygfin).'+1)), shading interp
% pcolor(tslide,fftshift(kgabor),log(abs(ygfin).'+1)), shading interp

% pcolor(tslide,ks*0.1,y_GNRg.'), shading interp

colormap('hot')
% set(gca,'Ylim', [-1 1])
colorbar
% caxis([3.5 max(max(log(abs(ygfin).'+1))) ])

xlabel('Time (sec)');
ylabel('Frequency (Hz)');
%% SVD
% Try inclusong multiple variables in the data matrix and see what happens

% create data matrix for just one varibale
% must down sample for using svd (not econ)
% cond = N; 
cond = 1;
% 1 = P L
% 2 = P noL
% 3 = noP L
% 4 = noP noL

xn1 = data2100(cond).Q; 
xn2 = data2100(cond).P1; 
xn3 = data2100(cond).P2; 
t = data2100(cond).t;

% xn1 = data2440(cond).Q; 
% xn2 = data2440(cond).P1; 
% xn3 = data2440(cond).P2; 
% t = data2440(cond).t;

% xn1 = data2800(cond).Q; 
% xn2 = data2800(cond).P1; 
% xn3 = data2800(cond).P2; 
% t = data2800(cond).t;

% down sample rate -> numel = N/f
% f = 100;
% td = downsample(t,f);
% xnd1 = downsample(xn1,f);
% xnd2 = downsample(xn2,f);
% xnd3 = downsample(xn3,f);
% stadardizing of all varables is done since looking at multiple variables
% of different untis (L/min vs mmHg)

% centering
% x1c = xnd1-mean(xnd1);
% x2c = xnd1-mean(xnd2);
% x3c = xnd1-mean(xnd3);

x1c = xn1-mean(xn1);
x2c = xn2-mean(xn2);
x3c = xn3-mean(xn3);

% nromalizing by standard deviation
x1 = x1c/std(x1c);
x2 = x2c/std(x2c);
x3 = x3c/std(x3c);


X = [x1; x2; x3];
% [u, s, v] = svd(X);
% [u2, s2, v2] = svd(X');
[u,s,v]=svd(X,'econ');
%% plotting oringal data
close all
figure(1)
plot(t,xn1,'LineWidth',(1))
hold on
yyaxis('right')
plot(t,xn2,t,xn3,'k','LineWidth',(1))
hold off
legend('Flow Rate','Pressure 1', 'Pressure 2')
%% plot u modes with data
close all
% figure(2)
% plot(1:3,u(:,1),'o',1:3,u(:,2),'o',1:3,u(:,3),'o','LineWidth',(3))
% legend('mode 1','mode 2','mode 3')

plot3(s(1,1)*[0,u(1,1)],s(1,1)*[0,u(2,1)],s(1,1)*[0,u(3,1)],...
    s(2,2)*[0,u(1,2)],s(2,2)*[0,u(2,2)],s(2,2)*[0,u(3,2)],...
    s(3,3)*[0,u(1,3)],s(3,3)*[0,u(2,3)],s(3,3)*[0,u(3,3)],...
    'LineWidth', (2)); 
% plot3(s(1,1)*u(1,:),s(2,2)*u(2,:),s(3,3)*u(3,:),'LineWidth', (2)); 
% plot3(u(1,:),u(2,:),u(3,:),'LineWidth', (2)); 
hold on
plot3(x1,x2,x3,'o')
hold off
legend('mode 1', 'mode 2', 'mode 3')
xlabel('Q')
ylabel('P1')
zlabel('P2')
xlim([-5,5]);
ylim([-5,5]);
zlim([-5,5]);

%% plot v modes (how each snapshot projects onto u vecs)
figure(1)
subplot(1,3,1)
plot(t,v(:,1))
title('mode 1')
subplot(1,3,2)
plot(t,v(:,2))
title('mode 2')
subplot(1,3,3)
plot(t,v(:,3))
title('mode 3')

%% plotting singular values
svec = diag(s);
% Singular Value as a Percentage of Total Energy
plot(1:3,svec/sum(svec)*100,'o','LineWidth',(2))
ylabel('Singular Value Percentage')
xlabel('Mode')

%% plotting reconstruction with second mode...looks like a mean change
% plot(t,u(1,1)*svec(1)*(v(1,1))'+u(2,3)*svec(2)*(v(:,3))'+u(1,3)*svec(3)*(v(:,3))')
plot(t,u(1,1)*svec(1)*(v(1,1))'+u(1,3)*svec(3)*(v(:,3))')

%% PCA
% svd of the covererniance of X

[up, sp, vp] = svd(cov(X'));
%% plotting for PCA
close all
plot3(sp(1,1)*[0,up(1,1)],sp(1,1)*[0,up(2,1)],sp(1,1)*[0,up(3,1)],...
    sp(2,2)*[0,up(1,2)],sp(2,2)*[0,up(2,2)],sp(2,2)*[0,up(3,2)],...
    sp(3,3)*[0,up(1,3)],sp(3,3)*[0,up(2,3)],sp(3,3)*[0,up(3,3)],...
    'LineWidth', (2)); 
hold on
plot3(x1,x2,x3,'o')

hold off
legend('mode 1', 'mode 2', 'mode 3')
xlabel('Q')
ylabel('P1')
zlabel('P2')
xlim([-5,5]);
ylim([-5,5]);
zlim([-5,5]);
%% Robust PCA
% book uses inexact_alm_rpca
% acts as an advances filter to clean the data matrix
% M = L + S + N (M = data, L = low rank matrix, S = sparce matrix due to
% measurement error, N = dense noise)

%% Maching learning (pulse detection)?
% Unfortunatly I do not have data that indicates the actual timing of the
% pulse of Lavare to use for training. I DO KNOW the timing of the singals
% (every 60 seconds).

%% Compressed Sensing
% I can see how well I can recreate the singnal with compressed sensing?
% I could try applying this to the corrupt data sets I have. 

% Opt 1: apply compressive sensing to un altered bad signal and see what
% happens

% Opt 2: Get rid of 0s (when the sidnal dissapears) and apply compr. sens.

% compare to good singal of same parametes?
