function [ygfin, k, tslide] = gaborfilt(y,f,a,tstep,filtOpt,sig,numks,kcut)
% The gabor filter: g = exp(-a*(t-Tou).^2)
% a = width of g and determines resoltion in time
% tstep = dt in dummy var tou
% numks = number of peak k values to filter about
n = length(y);
t = (1:n)/f;
y = y';
L = t(end);

k = (1/L)*[0:n/2-1 -n/2:-1];

tslide = 0:tstep:L;
yg = ones(length(tslide),n);
Gauss1D = @(x,mu) exp(-sig*((x-mu).^2));

if filtOpt % filter at each tou step about peak k
    maskcut = k<kcut & k>-kcut;

    for i = 1:length(tslide)
        g = exp(-a*(t-tslide(i)).^2);
        ygn = y.*g;
        ygnft = fft(ygn);
        ygnftcut = ygnft.*double(maskcut);
        [Mvals,Ivals] = maxk(ygnftcut,numks);
        kvals = k(Ivals);
        %         kvalsneg = -kvals;
        %         kvec = [kvals kvalsneg];
        
        GFsum = 0;
        for j = 1:length(kvals)
            GF = Gauss1D(k,kvals(j));
            GFsum = GFsum + GF;
        end
        
        % Gaus1DFilt = @(x,mu) exp(-sig*((x-mu).^2));
        % KMaxFilt = Gaus1DFilt(k)/max(abs(Gaus1DFilt(k)));
        
        
        ygft = ygnftcut.*GFsum/(max(ygnft)*max(GFsum));
        yg(i,:) = ygft;
    end
    
else
    for i = 1:length(tslide)
        g = exp(-a*(t-tslide(i)).^2);
        yg(i,:) = abs(fftshift(fft(y.*g)));
    end
end
ygfin = yg;
end