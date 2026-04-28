function rez = merge_posthoc3(rez)
%fracse = 0.1;
mu = rez.mu;
fracse = rez.ops.fracse;

ops = rez.ops;
LAM = ops.lam(3) * (20./mu).^2;
Nfilt = rez.ops.Nfilt;

Cmerge = Inf *ones(Nfilt);
tfi = rez.iNeigh;
tf = rez.cProj;
clusterIDs = rez.st3(:,2);
%
nSpikes = size(rez.st3,1);

fmax = zeros(nSpikes,1, 'single');

% find all pairs of similar clusters
pairs = {};
for testID = 1:Nfilt
    spikesTest = clusterIDs==testID;
%     tfnew = bsxfun(@plus, tf(spikesTest, :), LAM(tfi(:, testID))'.*mu(tfi(:, testID))');
%     tf(spikesTest, :) = bsxfun(@rdivide, tfnew, sqrt(1+LAM(tfi(:, testID)))');
    
    pp = tfi(:, testID);
    pp(pp==testID) = [];
    pairs{testID} = pp;
    [~, isame] = min( abs(tfi(:, testID)-testID));
    fmax(spikesTest, 1) = tf(spikesTest, isame);
end


%
inewclust = 0;
clear iMegaC
picked = zeros(Nfilt, 1);
% tic
while 1
    % find cluster with the most spikes
    [maxseed, iseed] = max(rez.nbins(1:Nfilt) .* (1-picked), [], 1);
    
%     [maxseed, iseed] = max(mu(1:Nfilt) .* (1-picked), [], 1);

    % if cluster has less than 500 spikes in it, don't continue running
    if maxseed<500
        break;
    end
    
    picked(iseed) = 1;
    
    run_list = [iseed];
    pair_list = pairs{iseed};
    strun = find(clusterIDs==iseed);
    
    % run through putative pair matches and combine if it makes sense
    while ~isempty(pair_list)
        
        % pick the putative match with the most spikes
        [mmax, ipair] = max(rez.nbins(pair_list));
        
        % if it has less than 100 spikes, you can't get a good read from
        % these methods, so stop trying
        if mmax<100
            break;
        end
        
        ipair = pair_list(ipair);
        
        % check backwards: is the cluster in run_list also a putative match
        % to the cluster we're about to check 
        imm = ismember(tfi(:, ipair), run_list);
        if sum(imm)
            %
            new_spikes = find(clusterIDs==ipair);
            f1new = max(tf(new_spikes, imm), [], 2);
            
            f2new = fmax(new_spikes);
            
            f1old = fmax(strun);
            f2old = NaN * ones(numel(f1old), 1, 'single');
            i0 = 0;
            for j = 1:length(run_list)
                ifeat = find(tfi(:, run_list(j))==ipair);
                if ~isempty(ifeat)
                    f2old(i0 + (1:rez.nbins(run_list(j))),1) = ...
                        tf(clusterIDs==run_list(j), ifeat);
                    i0 = i0 + rez.nbins(run_list(j));
                end
            end
            
            f1old(isnan(f2old))=[];
            f2old(isnan(f2old))=[];
            mo = merging_score(f1old - f2old, f1new-f2new, ops.fracse);
           
            
            
            %%
            if mo<3 % merge?
                
                % check if the cross-correlation matches the
                % autocorrelation
                binSize = 2;
                num = max(max(strun), max(new_spikes));
                binfun = @(x) (x==0) + ceil(x/binSize);
                s1 = binfun(strun);
                s2 = binfun(new_spikes);
                sp1 = full(sparse(s1, ones(numel(s1),1), ones(numel(s1),1), num, 1));
                sp2 = full(sparse(s2, ones(numel(s2),1), ones(numel(s2),1), num, 1));
                
                nlags = binfun(100);
%                 bs = basisFactory.basisFactory('history', binfun, nlags)
%                 X1 = bs.convolve(sp1(1:end-1));
%                 X2 = bs.convolve(sp2(1:end-1));
%                 w1 = (X1'*X1)\(X1'*sp1(2:end));
%                 w2 = (X2'*X2)\(X2'*sp2(2:end));
%                 plot(bs.B*w1); hold on; plot(bs.B*w2)
                
                xc1 = xcorr(sp1, nlags, 'unbiased');
                xc2 = xcorr(sp2, nlags, 'unbiased');
                xc3 = xcorr(sp1, sp2, nlags, 'unbiased');
                xc1(nlags+1) = 0;
                xc2(nlags+1) = 0;
                xc1 = smooth(xc1, 5);
                xc2 = smooth(xc2, 5);
                xc3 = smooth(xc3, 5);
%                 xc3(nlags+1) = 0;
                lags = -nlags:nlags;
                nfun = @(x) x/norm(x);
                xc1 = nfun(xc1);
                xc2 = nfun(xc2);
                xc3 = nfun(xc3);
                
%                 figure(1); clf
%                 plot(lags,xc1, lags, xc2, lags, xc3)
%                 xproj = [xc1 xc2 xc3]'*[xc1 xc2 xc3];
                xproj = corr([xc1 xc2 xc3]);
%                 title(mean(xproj([2,3,6])))
%                 drawnow
                
                
                if mean(xproj([2,3,6])) > .5
                    
                    fprintf('Merging %d\n', ipair)
                    
                    strun = cat(1, strun, new_spikes);
                    run_list(end+1) = ipair;
                    picked(ipair)   = 1;
                    if mmax>300
                        pair_list = unique(cat(1, pair_list, pairs{ipair}));
                        pair_list(ismember(pair_list, run_list)) = [];
                    end
                end
            end
        end
        pair_list(pair_list==ipair) = [];
    end
    
    inewclust = inewclust + 1;
    
    iMegaC{inewclust} = run_list;
%     [sum(picked) run_list]
end

% toc
%

iMega = zeros(Nfilt, 1);
for i = 1:length(iMegaC)
   iMega(iMegaC{i}) = iMegaC{i}(1); 
end
rez.iMega = iMega;
rez.iMegaC = iMegaC;


rez.st3(:,5) = iMega(rez.st3(:,2));