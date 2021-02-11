%% AMATH 582 HW 2
% Due 2/10/21
% Marissa Miramontes
clear 
clc
close all
%% Set up and plotting signal from hw assignment

[y_GNR, Fs_GNR] = audioread('GNR.m4a'); % Fs is the rample rate (samples/sec)
tr_GNR = length(y_GNR)/Fs_GNR; % record time in seconds

[y_FLD, Fs_FLD] = audioread('Floyd.m4a');
tr_FLD = length(y_FLD)/Fs_FLD; % record time in seconds

%% listening to GNR
%  p = audioplayer(y_GNR, Fs_GNR); play(p);
        
%% listening to FLD
% p = audioplayer(y_FLD, Fs_FLD); play(p);

%% plotting orignal audio files
% figure(1)
% plot((1:length(y_GNR))/Fs_GNR,y_GNR);
% xlabel('Time [sec]'); ylabel('Amplitude');
% title('GNR')
% 
% figure(2)
% plot((1:length(y_FLD))/Fs_FLD,y_FLD);
% xlabel('Time [sec]'); ylabel('Amplitude');
% title('FLD')
%% Note Frequncies vector

noteCell = table2cell(readtable('notes.xlsx')); % col 1 = freq, col 2 = notes


%% GNR with gabor filter g from notes g = exp(-(t-time)^2
stepGNR = 0.1; 
aGNR = 100; 
[y_GNRg, kGNR, tslideGNR] = gaborfilt(y_GNR,Fs_GNR,aGNR,stepGNR,0);

%% GNR plotting spectrogram (for guitar)
close all
pcolor(tslideGNR,fftshift(kGNR),log(abs(y_GNRg.')+1)), shading interp
% pcolor(tslide,ks*0.1,y_GNRg.'), shading interp

colormap('hot')
set(gca,'Ylim', [200 600])
% set(gca,'Ylim', [0 5000])
xlabel('Time (sec)');
ylabel('Frequency (Hz)');

for i = 1:length(noteCell)
    hold on
    YL = yline(noteCell{i,1},'--','Linewidth',1,'color','white');
    YL.Label = noteCell{i,2};
    YL.LabelVerticalAlignment = 'middle';
    
end
title('Guns N Roses: "Sweet Child O Mine"')

%% FLD clip gabor filtering with g = exp(-(t-tou)^2

% Q: why note use 1/(2piL) here?? because we want units of Hz and not
% radians

% the 60 second sample is too long for matlab
% first few seconds of clip:
N = length(y_FLD);
sec = 13;
y_FLD_sl = y_FLD(1:sec*Fs_FLD);

stepFLD = 0.1; 
aFLD = 75; 
[y_FLDg, kFLD, tslideFLD] = gaborfilt(y_FLD_sl,Fs_FLD,aFLD,stepFLD,0);
%% play smaller clip

% p = audioplayer(y_FLD_sl, Fs_FLD); play(p);
%% FLD plotting spectrogram (for bass)
close all
pcolor(tslideFLD,fftshift(kFLD),log(abs(y_FLDg.')+1)), shading interp
% pcolor(tslide,ks*0.1,y_GNRg.'), shading interp

colormap('hot')
% set(gca,'Ylim', [200 600])
set(gca,'Ylim', [70 320]) 
xlabel('Time (sec)');
ylabel('Frequency (Hz)');

for i = 1:length(noteCell)
    hold on
    YL = yline(noteCell{i,1},'--','Linewidth',1,'color','white');
    YL.Label = noteCell{i,2};
    YL.LabelVerticalAlignment = 'middle';
    
end
title('Pink Floyd: "Comfortably Numb"')
%% plotting the FT vs k GNR

plot(fftshift(kGNR),abs(fftshift(fft(y_GNR))/max(abs(fft(y_GNR)))));

%% plotting the FT vs k FLD

plot(fftshift(kFLD),abs(fftshift(fft(y_FLD_sl))/max(abs(fft(y_FLD_sl)))));

%% Filtering in freq space for every tou after applying gabor filter g
sig = 0.5; % smaller sigma wider guassian
numks = 100; %2000; % number of peak k vals to filter about
kcutGNR = 1050;
kcutFLD = 1050;%320;
[y_GNRgf, kGNR, tslideGNR] = gaborfilt(y_GNR,Fs_GNR,aGNR,stepGNR,1,sig,numks,kcutGNR);
[y_FLDgf, kFLD, tslideFLD] = gaborfilt(y_FLD_sl,Fs_FLD,aFLD,stepFLD,1,sig,numks,kcutFLD);
disp('done')
%% GNR filtered PLOTTING
close all
pcolor(tslideGNR,fftshift(kGNR),fftshift(log(abs(y_GNRgf.')+1))), shading interp
% pcolor(tslide,ks*0.1,y_GNRg.'), shading interp

colormap('hot')
set(gca,'Ylim', [200 600])
% set(gca,'Ylim', [0 200])
xlabel('Time (sec)');
ylabel('Frequency (Hz)');

for i = 1:length(noteCell)
    hold on
    YL = yline(noteCell{i,1},'--','Linewidth',1,'color','white');
    YL.Label = noteCell{i,2};
    YL.LabelVerticalAlignment = 'middle';
    
end
title('Guns N Roses: "Sweet Child O Mine"')

%% FLD filtered PLOTTNIG

close all
pcolor(tslideFLD,fftshift(kFLD),fftshift(log(abs(y_FLDgf.')+1))), shading interp
% pcolor(tslide,ks*0.1,y_GNRg.'), shading interp

colormap('hot')
set(gca,'Ylim', [70 650])
% set(gca,'Ylim', [70 320]) 
xlabel('Time (sec)');
ylabel('Frequency (Hz)');

for i = 1:length(noteCell)
    hold on
    YL = yline(noteCell{i,1},'--','Linewidth',1,'color','white');
    YL.Label = noteCell{i,2};
    YL.LabelVerticalAlignment = 'middle';
    
end
title('Pink Floyd: "Comfortably Numb"')

%% now apply cut of about 150 Hz to only capture base
kcutFLD = 150;
[y_FLDgf, kFLD, tslideFLD] = gaborfilt(y_FLD_sl,Fs_FLD,aFLD,stepFLD,1,sig,numks,kcutFLD);

%% FLD filtered max 150 Hz PLOTTNIG

close all
pcolor(tslideFLD,fftshift(kFLD),fftshift(log(abs(y_FLDgf.')+1))), shading interp
% pcolor(tslide,ks*0.1,y_GNRg.'), shading interp

colormap('hot')
set(gca,'Ylim', [70 150])
% set(gca,'Ylim', [70 320]) 
xlabel('Time (sec)');
ylabel('Frequency (Hz)');

for i = 1:length(noteCell)
    hold on
    YL = yline(noteCell{i,1},'--','Linewidth',1,'color','white');
    YL.Label = noteCell{i,2};
    YL.LabelVerticalAlignment = 'middle';
    
end
title('Pink Floyd: "Comfortably Numb"')

%% Combining FLD base with GNR guiatar
y_FLDgfpad = zeros(size(y_GNRgf));
[r, c] = size(y_FLDgf);
y_FLDgfpad(1:r,1:c) = y_FLDgf ;
y_combo = abs(y_FLDgfpad)+ abs(y_GNRgf);


%% COMBO filtered PLOTTING
close all
pcolor(tslideGNR,fftshift(kGNR),fftshift(log(y_combo.'+1))), shading interp
% pcolor(tslide,ks*0.1,y_GNRg.'), shading interp

colormap('hot')
set(gca,'Ylim', [70 600])
% set(gca,'Ylim', [0 200])
xlabel('Time (sec)');
ylabel('Frequency (Hz)');

% for i = 1:length(noteCell)
%     hold on
%     YL = yline(noteCell{i,1},'--','Linewidth',1,'color','white');
%     YL.Label = noteCell{i,2};
%     YL.LabelVerticalAlignment = 'middle';
%     
% end
title('"Sweet Child O Mine" + "Comfortably Numb')
%% cut off freq does not help clean spectrogram
% %% filtering in Freq space GNR
% maskcutGNR = k<1050 & k>-1050;% max freq for guitar
% y_GNRtcut = fft(y_GNR).*double(maskcutGNR);
% % cut off freq
% plot(ks,abs(fftshift(y_GNRt))/max(abs(y_GNRt)))
% hold on 
% plot(ks,abs(fftshift(y_GNRtcut))/max(abs(y_GNRtcut)))
% hold off
% xlim([-10000 10000]);
% 
% %% filtering in Freq space FLD
% maskcutFLD = k<320 & k>-320;% max freq for guitar
% y_FLD_sltcut = fft(y_FLD_slt).*double(maskcutFLD);
% % cut off freq
% plot(ks,abs(fftshift(y_FLD_slt))/max(abs(y_FLD_slt)))
% hold on
% plot(ks,abs(fftshift(y_FLD_sltcut))/max(abs(y_FLD_sltcut)))
% hold off
% xlim([-10000 10000]);
% % plot(ks, fftshift(maskcutFLD));
% % hold off
% 
% %% Now apply Gabor Tansform to Cutoff Freq filtered
% 
% y_GNRcut = ifft(y_GNRtcut);
% 
% step = 0.1; 
% a = 100; %a = 10000; %10000000; % width of g and detmines resoltion in time
% tslide = 0:step:L;
% y_GNRg = ones(length(tslide),n);
% 
% for i = 1:length(tslide)
% g = exp(-a*(t_GNR-tslide(i)).^2);
% y_GNRgcut(i,:) = abs(fftshift(fft(y_GNRcut.*g)));
% end
% %% Plot GNR cut spectrogram
% close all
% pcolor(tslide,ks,log(abs(y_GNRgcut.')+1)), shading interp
% % pcolor(tslide,ks*0.1,y_GNRg.'), shading interp
% 
% colormap('hot')
% set(gca,'Ylim', [200 600])
% % set(gca,'Ylim', [0 5000])
% xlabel('Time (sec)');
% ylabel('Frequency (Hz)');
% 
% for i = 1:length(noteCell)
%     hold on
%     YL = yline(noteCell{i,1},'--','Linewidth',1,'color','white');
%     YL.Label = noteCell{i,2};
%     YL.LabelVerticalAlignment = 'middle';
%     
% end
% title('Guns N Roses: "Sweet Child O Mine"')