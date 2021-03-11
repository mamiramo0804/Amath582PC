%% Assignment 4
% Marissa Miramontes
% 03-10-21
% text book for course was used for referece to codes
% Driven Modeling and Scientific Computation by J.N. Kutz
clear
clc
close all

%% loading training and test data
%  [images_tr, labels_tr] = mnist_parse('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz');
%  [images_te, labels_te] = mnist_parse('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz'); % test data

%% loading training and test data
gunzip('*.gz');
[images_tr0, labels_tr0] = mnist_parse('train-images-idx3-ubyte', 'train-labels-idx1-ubyte');
[images_te0, labels_te0] = mnist_parse('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte'); % test data

%% double values
[r, c, ntr] = size(images_tr0);
[r, c, nte] = size(images_te0);

images_tr = double(images_tr0);
labels_tr = labels_tr0;
images_te = double(images_te0);
labels_te = labels_te0;

%% svd 
% create data matrix X and remove mean
Xtr = zeros([ntr r*c]); Xtrm = Xtr;
Xte = zeros([nte r*c]); Xtem = Xte;
for i = 1:ntr
Xtr(i,:) = reshape(images_tr(:,:,i),[1,r*c]);
Xtrm(i,:) = Xtr(i,:) - mean(Xtr(i,:));
end
for i = 1:nte
Xte(i,:) = reshape(images_te(:,:,i),[1,r*c]);
Xtem(i,:) = Xte(i,:) - mean(Xte(i,:));
end

%% SVD
% recall: columns of U are the princple modes and diag(S) are the wieghts
% U(:,1)*S(1,1)*V^*(:,1) = mode 1 preojection (r*c size mat -> look at first
% col) (max value -> 150)

[Utr,Str,Vtr] = svd(Xtrm','econ');
[Ute,Ste,Vte] = svd(Xtem','econ'); 
% scaling of U(:,i) results in pixel values 10^4 vs 0.09
Strvec = diag(Str);
%% plotting SVD results TRAINING DATA ONLY
close all
Strvec_per = Strvec/sum(Strvec)*100;
N = c*r; % basically reaches 0 at 713
Nrank = 12; % Modes greater than the 12th each make up less than 1% of the sum of the weights
figure(1)
plot(1:N,Strvec_per(1:N),'o')
hold on
plot(Nrank,Strvec_per(Nrank),'r*','MarkerSize',12,'LineWidth',2);
ylabel('Mode weight as a percentage of the sum of the weights')
xlabel('Mode number')
hold off
% semilogy(1:N,Strvec(1:N),'-o');
grid on
figure(2)
for i = 1:Nrank
    subplot(3,4,i)
imagesc(reshape(Utr(:,i),r,c))
end
%% 3D plot of V TRAINING DATA ONLY
% each axis is a chosen mode to project the images onto and see the
% corresponding value
% each point should be labeled with a color representing the number

% V2 = Utr(2,2)*Vtr(:,2);
% V3 = Utr(3,3)*Vtr(:,3);
% V5 = Utr(5,5)*Vtr(:,5);
V2 = Vtr(:,2);
V3 = Vtr(:,3);
V5 = Vtr(:,5);

colorMat = [(1/255)*[205,92,92]; ... % red-pink 0
            (1/255)*[100,149,237]; ... % light blue 
            (1/255)*[32,178,170]; ... % turquiose 2
            (1/255)*[255,127,80]; ... % orange 3
            (1/255)*[255,179,179]; ... % light pink 4
            (1/255)*[153, 255, 102]; ... % lime grean 5
            (1/255)*[128, 0, 255]; ... % purple 6 
            (1/255)*[255, 204, 0]; ... % yellow 7
            (1/255)*[0, 51, 204]; ... % royal blue 8
            (1/255)*[0, 102, 34];]; ... % dark green 9
            
% plot3(V2,V3,V5,'o')
close all
% sorting into num vecs 1-9
for i = 0:9
    mask = labels_tr == i;
    V2i = V2(mask);
    V3i = V3(mask); 
    V5i = V5(mask);
    plot3(V2i,V3i,V5i,'*','color',colorMat(i+1,:),'LineWidth',0.75)
    hold on
end
hold off
legend('0','1', '2', '3', '4', '5', '6','7', '8', '9')
xlabel('Mode 2')
ylabel('Mode 3')
zlabel('Mode 5')
%% LDA (linear classifier) prep (addpated from book)
Nmodes = 12;
SVt = Str*Vtr'; % strength of the prejections onto each princ. dir. (U) -> each row is a mode
U =  Utr(:,1:Nmodes);
modeproj = SVt(1:Nmodes,:); % proj of all images for first n modes

% separate out numbers
for i = 0:9
    mask = labels_tr == i;
    SVti = [];
    for j = 1:Nmodes
%     ProjCell{i+1,j} = SVt(mask); % rows are digets, colums are modes 1 to N
%     % note: length of each vector within the cell is the number of associated images for each digit
    SVtimodej = SVt(j,mask); % rows -> modes, col -> images (1 by lenght(mask) vector)
    SVti = [SVti; SVtimodej];
    end  
    SVtcell{i+1} = SVti; % SVt matrix for each digit (num of images varies)
end


%% within class variances (adapted from book)
% for 2 digits
for i = 1:2 %  0 and 1
    meanSVt{i} = mean(SVtcell{i},2);
    [nmodes, nimages] = size(SVtcell{i});
    Sw = 0;
    for k = 1:nimages 
        Sw = Sw + (SVtcell{i}(:,k)-meanSVt{i})*(SVtcell{i}(:,k)-meanSVt{i})';
    end
    
end
Sb = (meanSVt{1}-meanSVt{2})*(meanSVt{1}-meanSVt{2})'; % between classes

%% LDA (adapted from book)
[V2,D] = eigs(Sb,Sw);
[lam,ind] = max(abs(diag(D)));
w = V2(:,ind); w = w/norm(w,2);
SVt0 = SVtcell{1};
SVt1 = SVtcell{2};
v0 = w'*SVt0; v1 = w'*SVt1; % estimated data? 
result = [v0, v1];

if mean(v0)>mean(v1)
   w = -w;
   v0 = -v0;
   v1 = -v1;
end
% thresholding

sort0 = sort(v0);
sort1 = sort(v1);
t1 = length(sort0);
t2 = 1;
while sort0(t1)>sort1(t2)
    t1 = t1-1;
    t2 = t2+1;
end
threshold = (sort0(t1)+sort1(t2))/2;

%% Test Data *includes digets 0 to 9
Test = Xtem'; % mean is alread subtracted out (col -> image, row -> pixel)
TestMat = U'*Test; % tested agsint a 0 and 1 classifier;
pval = w'*TestMat;

%% Checking accuracy
ResVec = (pval>threshold); % > thresh (1), < thresh (0)
testlabels = labels_te;
% Number of Mistakes with just 0s and 1s
errNum = sum(abs(double(ResVec(testlabels == 0 | testlabels == 1)) - testlabels(testlabels == 0 | testlabels == 1)'));
% Rate of success with just 0s and 1s
succRate = 1-errNum/length(pval);
%% Plotting LDA results
close all
figure(1)

subplot(1,2,1)
histogram(sort0,25); 
hold on
xline(threshold,'r','LineWidth', 1)
hold off 
title('Digit 0')

subplot(1,2,2)
histogram(sort1,25);
hold on
xline(threshold,'r','LineWidth', 1)
hold off 
title('Digit 1')
%% implementing SVM and decision tree classifiers (code provided by hw)
 
 % classification tree on fisheriris data
load fisheriris;
tree=fitctree(meas,species,'MaxNumSplits',3,'CrossVal','on');
view(tree.Trained{1},'Mode','graph');
classError = kfoldLoss(tree);
% SVM classifier with training data, labels and test set
Mdl = fitcsvm(xtrain,label);
test labels = predict(Mdl,test);