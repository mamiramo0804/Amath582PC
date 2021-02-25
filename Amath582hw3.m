%% AMATH 582 HW 3
% Marissa Miramontes
% 2-24-21
clc
clear
close all
%% loading ALL data
dirst = dir('cam*.mat');
n = length(dirst);
for i = 1:n
    datast(i).name = dirst(i).name;
    datast(i).data = load(datast(i).name);
end

%% test 1

cam1_0 = datast(1).data.vidFrames1_1;
cam2_0 = datast(5).data.vidFrames2_1;
cam3_0 = datast(9).data.vidFrames3_1;


[z1 x1 rbg t1] = size(cam1_0);
[z2 x2 rbg t2] = size(cam2_0);
[z3 x3 rbg t3] = size(cam3_0);


%% applying RBG theshhold to idenift point on bucket
% imshow(cam1(:,:,:,1))
% rgb_red = cam1(225, 329, :, 1); 
% % [205 138 152]; % rgb of red on bucket ([x y] = [329 225] for t = 1;
% % need to identify these pixels and then compute some centroid
% 
% % Want to keep same ratio of RGB (pinkish color) - otherwise you can end
% % up with grayish colors with in the max and min R G B values
% % CHANGING RANGE OF HUES
% RGdiff = rgb_red(:,:,1) - rgb_red(:,:,2);
% RBdiff = rgb_red(:,:,1) - rgb_red(:,:,3);
% 
% RGRBrange = 25;
% 
% RGdiffmax = RGdiff + RGRBrange;
% RGdiffmin = RGdiff - RGRBrange;
% 
% RBdiffmax = RBdiff + RGRBrange;
% RBdiffmin = RBdiff - RGRBrange;%1.5*RGRBrange;
% % rgb_max = 1.1*rgb_red;
% % rgb_min = 0.90*rgb_red;
% 
% % CHANGIG RANGE OF BRIGHTNESS (giving a larger range of light to dark
% % pinks)
% rbg_range = 100;
% rgb_max = rgb_red+rbg_range;
% rgb_min = rgb_red-1.5*rbg_range;
% 
% % rgbmax = reshape(rgb_max,[1 1 3]);
% rbgmaxMat = repmat(rgb_max,[z1 x1 1]);
% 
% % rgbmin = reshape(rgb_min,[1 1 3]);
% rbgminMat = repmat(rgb_min,[z1 x1 1]);
% % want a 3D mat for each t that is logicals -> logcal 2D mat for each t
% % (all(X,DIM) command)
% for i = 1:t1
%     RGdiffmat = cam1(:,:,1,i) - cam1(:,:,2,i);
%     RBdiffmat = cam1(:,:,1,i) - cam1(:,:,3,i);
% mask = cam1(:,:,:,i) < rbgmaxMat ...
%     & cam1(:,:,:,i) > rbgminMat ...
%     & RGdiffmat < RGdiffmax ...
%     & RGdiffmat > RGdiffmin ...
%     & RBdiffmat < RBdiffmax ...
%     & RBdiffmat > RBdiffmin;
% cam1mask(:,:,i) = all(mask,3);
% 
% 
% end

%% plotting to compare
% close all
% figure(1);
% for i = 1:t1
% subplot(1,2,1)
% imagesc(cam1mask(:,:,i))
% % title(['i = ' num2str(i)]);
% subplot(1,2,2)
% imshow(cam1(:,:,:,i))
% drawnow
% pause(0.5)
% end
% figure(3); imagesc(cam1(:,:,:,1))

%% downsizing pixels in frames for memory
z = z1/4; x = x1/4;
cam1 = imresize(cam1_0,[z x]);
cam2 = imresize(cam2_0,[z x]);
cam3 = imresize(cam3_0,[z x]);


%% RGB to gray scale
% the size cam1 is x (480) by z (640) by RGB (3) by time (226)
C1raw = zeros([z,x,t1]);
for i = 1:t1
    C1raw(:,:,i) = double(rgb2gray(cam1(:,:,:,i)));
end
C2raw = zeros([z,x,t2]);
for i = 1:t2
    C2raw(:,:,i) = double(rgb2gray(cam2(:,:,:,i)));
end
% C3 is rotated 90 degress 
C3_0 = zeros([z,x,t3]);
C3_80 = C3_0;
C3_90 = C3_0;
for i = 1:t3
    C3_0(:,:,i) = double(rgb2gray(cam3(:,:,:,i)));
    C3_80(:,:,i) = imrotate(double(rgb2gray(cam3(:,:,:,i))),-80,'bilinear','crop');
    C3_90(:,:,i) = imrotate(double(rgb2gray(cam3(:,:,:,i))),-90,'bilinear','crop');
end
% chosing angle to use: 0, -90, or -80 (will compare) 
% assuming -80 is ideal for alihngin the direction of motion with the other views:
C3raw = C3_0;

% %% Plotting frames C1 
% close all
% figure(1)
% for i = 1:t1
% subplot(1,2,1)
% pcolor(flipud(C1raw(:,:,i))), shading interp, colormap(gray),set(gca,'Xtick',[],'Ytick',[]);
% colorbar
% drawnow
% 
% subplot(1,2,2)
% pcolor(flipud(double(C1raw(:,:,i)==max(max(C1raw(:,:,i)))))), shading interp, colormap(gray),set(gca,'Xtick',[],'Ytick',[]);
% drawnow
% pause(0.3)
% end

%% try subtracting out the mean image

C1meanIm = mean(C1raw,3);
C2meanIm = mean(C2raw,3);
C3meanIm = mean(C3raw,3);

C1mod = C1raw-C1meanIm;
C2mod = C2raw-C2meanIm;
C3mod = C3raw-C3meanIm;


%% Plotting frames C1 without mean
close all
figure(1)
for i = 1:t1
subplot(1,2,1)
pcolor(flipud(C1mod(:,:,i))), shading interp, colormap(gray),set(gca,'Xtick',[],'Ytick',[]);
drawnow

subplot(1,2,2)
pcolor(flipud(double(C1mod(:,:,i)==max(max(C1mod(:,:,i)))))), shading interp, colormap(gray),set(gca,'Xtick',[],'Ytick',[]);
drawnow
pause(0.3)
end

%% filtering the image to smooth out edges from subtracting mean

C1filt = imgaussfilt3(C1mod,3); % 'FilterDomain','frequency');
C2filt = imgaussfilt3(C2mod,3);
C3filt = imgaussfilt3(C3mod,3);
%% normalize each image and center about 0
% for each frame:
% shift by lowest valuve( make all positive)
% normalize
for i = 1:t1
    minval = min(min(C1filt(:,:,i)));
    shift = C1filt(:,:,i)-minval;
C1sc(:,:,i) = shift/max(max(abs(shift))); % scaled
end
C1 = C1sc; % final version

for i = 1:t2
    minval = min(min(C2filt(:,:,i)));
    shift = C2filt(:,:,i)-minval;
C2sc(:,:,i) = shift/max(max(abs(shift))); % scaled
end
C2 = C2sc; % final version

for i = 1:t3
    minval = min(min(C3filt(:,:,i)));
    shift = C3filt(:,:,i)-minval;
C3sc(:,:,i) = shift/max(max(abs(shift))); % scaled
end
C3 = C3sc; % final version

%% define x and z
% define x and z values (0 to 1) for each pixel
zvec = 1:z;
xvec = 1:x;
[Z,X] = meshgrid(zvec,xvec);
%% computing median location of brightest spot
th = 0.90;
for i = 1:t1
    
    c1 = C1(:,:,i);
    MaxVal = max(c1(:));
    
    nvals = length(find((c1 >= th*MaxVal)));
    [I1,I2] = ind2sub(size(c1), find((c1 >= th*MaxVal)));  % I1 I2 are vectors of the locations
    
    for j = 1:nvals
        zvals(j) = zvec(I1(j));
        xvals(j) = xvec(I2(j));
        
    end
    zMax1(i) = median(zvals);
    xMax1(i) = median(xvals);
    % finding loaction of median z and x value for verification
%     maskmat1(:,:,i) = double(abs(diff(Z-zMax1(i))) == min(min(abs(diff(Z-zMax1(i)))))).*double(abs(diff(X-xMax1(i))) == min(min(abs(diff(X-xMax1(i))))));
    
end

for i = 1:t2
    
    c2 = C2(:,:,i);
    MaxVal = max(c2(:));
    
    nvals = length(find((c2 >= th*MaxVal)));
    [I1,I2] = ind2sub(size(c2), find((c2 >= th*MaxVal)));  % I1 I2 are vectors of the locations
    
    for j = 1:nvals
        zvals(j) = zvec(I1(j));
        xvals(j) = xvec(I2(j));
        
    end
    zMax2(i) = median(zvals);
    xMax2(i) = median(xvals);
    % finding loaction of median z and x value for verification
%     maskmat1(:,:,i) = double(abs(diff(Z-zMax1(i))) == min(min(abs(diff(Z-zMax1(i)))))).*double(abs(diff(X-xMax1(i))) == min(min(abs(diff(X-xMax1(i))))));
    
end

for i = 1:t3
    
    c3 = C3(:,:,i);
    MaxVal = max(c3(:));
    
    nvals = length(find((c3 >= th*MaxVal)));
    [I1,I2] = ind2sub(size(c3), find((c3 >= th*MaxVal)));  % I1 I2 are vectors of the locations
    
    for j = 1:nvals
        zvals(j) = zvec(I1(j));
        xvals(j) = xvec(I2(j));
        
    end
    zMax3(i) = median(zvals);
    xMax3(i) = median(xvals);
    % finding loaction of median z and x value for verification
%     maskmat1(:,:,i) = double(abs(diff(Z-zMax1(i))) == min(min(abs(diff(Z-zMax1(i)))))).*double(abs(diff(X-xMax1(i))) == min(min(abs(diff(X-xMax1(i))))));
    
end

%% plotting z and x
close all
figure(1)
subplot(2,1,1)
plot(1:t1,zMax1,1:t2,zMax2,1:t3,zMax3)
title('z locations (pixels)')
xlabel('frames')
legend('Camera 1', 'Camera 2', 'Camera 3');
subplot(2,1,2)
plot(1:t1,xMax1,1:t2,xMax2,1:t3,xMax3)
title('x locations (pixels)')
xlabel('frames')
legend('Camera 1', 'Camera 2', 'Camera 3');
%% PCA
X123 = [zMax1;xMax1;zMax2(1:t1);xMax2(1:t1);zMax3(1:t1);xMax3(1:t1)];
A = (X123')*(X123);
[V,D] = eigs(A, 20, 'lm');

%%
figure(1)
plot((1:20),diag(D))
figure(2)
v1 = V(:,1)';
plot(1:t1,v1)
title('v1');

%%


%% plotting filtered imag
close all

for i = 1:t1
%     subplot(1,2,1)
pcolor(flipud(C1(:,:,i))), shading interp, colormap(gray),set(gca,'Xtick',[],'Ytick',[]);
colorbar
drawnow

% subplot(1,2,2)
% pcolor(flipud(maskmat1(:,:,i))), shading interp, colormap(gray),set(gca,'Xtick',[],'Ytick',[]); % doesnt work at the moment
% drawnow
% pause(0.3)
end
%% 
% 
% %% locating the max of the image
% % thresh = 50;
% % for i = 1:t1
% % C1filtmat = C1filt(:,:,i);
% % maxVal = max(C1filtmat(:));
% % thesval = maxVal - thresh;
% % mask(:,:,i) = C1filtmat.*double(C1filtmat>thesval);
% % end
% % 
% %% plotting max vals
% % 
% % close all
% % for i = 1:t1
% % pcolor(flipud(mask(:,:,i))), shading interp, colormap(gray),set(gca,'Xtick',[],'Ytick',[]);
% % colorbar
% % drawnow
% % pause(0.3)
% % end
% %% thinking
% % [U,S,V] = svd(C);
% % do i see how well harmonic motion maps onto this dataset?
% % maybe I extract x and z from the S matrix
% 
% 
% %% Plotting frames C2 
% % close all
% % for i = 1:t2
% % pcolor(flipud(mask(:,:,i))), shading interp, colormap(gray),set(gca,'Xtick',[],'Ytick',[]);
% % drawnow
% % pause(0.3)
% % end
% 
% 
% %% Plotting frames C3 - should rotate and resize
% % close all
% % for i = 1:t3
% % pcolor(flipud(C3(:,:,i))), shading interp, colormap(gray),set(gca,'Xtick',[],'Ytick',[]);
% % drawnow
% % pause(0.3)
% % end
% %% plotting side by side RAW 
% % note that these were *not* all filmed at the same time
% % close all
% % figure(1)
% % for i = 1:t1
% % subplot(1,3,1)
% % pcolor(flipud(C1raw(:,:,i))), shading interp, colormap(gray),set(gca,'Xtick',[],'Ytick',[]); drawnow
% % title('C1');
% % subplot(1,3,2)
% % pcolor(flipud(C2raw(:,:,i))), shading interp, colormap(gray),set(gca,'Xtick',[],'Ytick',[]); drawnow
% % title('C2');
% % subplot(1,3,3)
% % pcolor(flipud(C3raw(:,:,i))), shading interp, colormap(gray),set(gca,'Xtick',[],'Ytick',[]); drawnow
% % title('C3');
% % pause(1)
% % end
% 
% %% plotting side by side FILTERED
% % note that these were *not* all filmed at the same time
% close all
% figure(1)
% for i = 1:t1
% subplot(1,3,1)
% pcolor(flipud(C1(:,:,i))), shading interp, colormap(gray),set(gca,'Xtick',[],'Ytick',[]); drawnow
% title('C1');
% subplot(1,3,2)
% pcolor(flipud(C2(:,:,i))), shading interp, colormap(gray),set(gca,'Xtick',[],'Ytick',[]); drawnow
% title('C2');
% subplot(1,3,3)
% pcolor(flipud(C3(:,:,i))), shading interp, colormap(gray),set(gca,'Xtick',[],'Ytick',[]); drawnow
% title('C3');
% pause(1)
% end
% %% Are C1, C2 and C3 statistically related?
% % assumed averageed average in notes for a and b
% 
% % mean of non-zero values (black in the image)
% mean1 = mean(C1(C1~=0)); % 105
% mean2 = mean(C2(C2~=0)); % 131
% mean3 = mean(C3(C3~=0)); % 96
% 
% % calculating covariance
% % cov(X)
% %     For matrices, where 
% %     each row is an observation, and each column a variable, cov(X) is the 
% %     covariance matrix.  DIAG(cov(X)) is a vector of variances for each 
% %     column, and SQRT(DIAG(cov(X))) is a vector of standard deviations. 
% %     cov(X,Y), where X and Y are matrices with the same number of elements,
% %     is equivalent to cov([X(:) Y(:)]). 
% %     The mean is removed from each column before calculating the result.
% 
% %% X matricies for each camera (rows are frames, col are pixels)
% % X1 = zeros([t1,numel(C1raw(:,:,1))]);
% % for i = 1:t1
% %    C1vec = C1raw(:,:,i);
% %    X1(i,:) =  C1vec(:)';
% % end
% % 
% % X2 = zeros([t2,numel(C2raw(:,:,1))]);
% % for i = 1:t2
% %    C2vec = C2raw(:,:,i);
% %    X2(i,:) =  C2vec(:)';
% % end
% % 
% % X3 = zeros([t3,numel(C3raw(:,:,1))]);
% % for i = 1:t3
% %    C3vec = C3raw(:,:,i);
% %    X3(i,:) =  C3vec(:)';
% % end
% % 
% % X1cov = cov(X1); X1var = daig(X1cov);
% % X2cov = cov(X2); X2var = daig(X2cov);
% % X3cov = cov(X3); X3var = daig(X3cov);
% % 
% % % t2 > t3 > t1 -> all same dim of t2 (filling with 0s)
% % X1pad = padarray(X1,0,t2-t1,'post');
% % X3pad = padarray(X3,0,t2-t3,'post');
% % 
% % % cov with each other using these X mats
% % X123cov1 = cov([X1pad(:) X2(:) X3pad(:)]);
% %% Do I just include all the frames into one long X matrix?
% % Cmat0 = cat(3,C1pad,C2raw);
% % Cmat = cat(3,Cmat0,C3raw);
% % for i = 1:t2
% % %     
% % %     Cvec = 
% % %    X123v2(i,:) = Cvec 
% % end
% 
% %% X matrix for all cameras (rows are frames, col are pixels)
% % need to pad with zeros first...
% 
% % t2 > t3 > t1 -> all same dim of t2 (filling with 0s)
% C1pad = cat(3,C1,zeros([z,x,t2-t1]));
% C3pad = cat(3,C3,zeros([z,x,t2-t3]));
% 
% % X123 = zeros([3*t2,numel(C2(:,:,1))]);
% X1 = zeros([t2,numel(C2(:,:,1))]);
% X2 = X1;
% X3 = X1;
% 
% for i = 1:t2
%     c1 = C1pad(:,:,i);
%     X1(i,:) = c1(:)';
%     
%     c2 = C2(:,:,i);
%     X2(i,:) = c2(:)';
%     
%     c3 = C3pad(:,:,i);
%     X3(i,:) = c3(:)';
% end
% X123 = [X1;X2;X3]; % t2*3 by number of pixels in a frame
% 
% %% computing eigenvectors and diagonal
% 
% A = (X123')*(X123);
% [V,D] = eigs(A, 20, 'lm');
% 
% %% 
% figure(1)
% plot((1:20),diag(D))
% figure(2)
% v1 = reshape(V(:,1),x,z);
% pcolor(flipud(v1)), shading interp, colormap(gray),set(gca,'Xtick',[],'Ytick',[]); drawnow
% title('v1');
% 
