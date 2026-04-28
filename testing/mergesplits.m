function [d2d, iY, drez] = distance_betwxt(dWU)
[nt0, Nchan, Nfilt] = size(dWU);

dWU = reshape(dWU, nt0*Nchan, Nfilt);
d2d = dWU' * dWU;

mu = sum(dWU.^2,1).^.5;
mu = mu';

muall2 = repmat(mu.^2, 1, Nfilt);
d2d = 1 - 2 * d2d./(1e-30 + muall2+ muall2');

d2d  = 1- triu(1 - d2d, 1);

[dMin, iY] = min(d2d, [], 1);

drez = dMin;


end


function steps = merging_score(fold, fnew, fracse)


troughToPeakRatio = 3;

l1 = min(fnew);
l2 = max(fold);

se = (std(fold) + std(fnew))/2;
se25 = fracse * se;
b2 = [0:se25:-l1];
b1 = [0:se25:l2];

hs1 = my_conv(histc(fold, b1), 1);
hs2 = my_conv(histc(-fnew, b2), 1);

mmax = min(max(hs1), max(hs2));

m1 = ceil(mean(fold)/se25);
m2 = -ceil(mean(fnew)/se25);

steps = sum(hs1(1:m1)<mmax/troughToPeakRatio) + ...
    sum(hs2(1:m2)<mmax/troughToPeakRatio);


function WtW = pairwise_dists(WU, WUinit)

WU = reshape(WU, [], size(WU,3));
WUinit = reshape(WUinit, [], size(WUinit,3));
WtW = WU' * WUinit;


mu = sum(WU.^2,1);
mu = mu(:);

muinit = sum(WUinit.^2,1);
muinit = muinit(:);

mu     = repmat(mu, 1, size(WUinit,2));
muinit = repmat(muinit', size(WU,2), 1);

WtW = 1 - 2*WtW ./ (muinit + mu);


function [dWUtot, dbins, nswitch, nspikes, iYout] = ...
    replace_clusters(dWUtot,dbins, Nbatch, mergeT, splitT,  WUinit, nspikes)

uu = Nbatch * dbins;
nhist = 1:1:100;
% nSpikes = sum(uu,1);
nSpikes = sum(nspikes,2)';

[score, iY1, mu1, mu2, u1, u2]   = split_clust(uu, nhist);
[~, iY, drez]                    = distance_betwxt(dWUtot);

[dsort, isort] = sort(drez, 'ascend');
iYout = iY(isort);

nmerged = sum(dsort<mergeT);
nsplit = sum(score>splitT);

mu = sum(sum(dWUtot.^2,1),2).^.5;
mu = mu(:);
freeInd = find(nSpikes<200 | mu'<10 | isnan(mu'));

for k = 1:nmerged
    % merge the two clusters
    iMerged = iY(isort(k));
    wt = [nSpikes(iMerged); nSpikes(isort(k))];
    wt = wt/sum(wt);
%     mu(iMerged) = [mu(iMerged) mu(isort(k))] * wt;
    
    dWUtot(:,:,iMerged)  = dWUtot(:,:,iMerged) * wt(1) + dWUtot(:,:,isort(k)) * wt(2);
    dWUtot(:,:,isort(k)) = 1e-10;
    
    nspikes(iMerged, :) = nspikes(iMerged, :) + nspikes(isort(k), :);
    nspikes(isort(k), :) = 0;
end


for k = 1:min(nmerged+numel(freeInd), nsplit)
    if k<=numel(freeInd)
        inew= freeInd(k);
    else
        inew = isort(k - numel(freeInd));
    end
    
    mu0 = mu(iY1(k));
    
    % split the bimodal cluster, overwrite merged cluster
    mu(inew)     = mu1(k);
    mu(iY1(k))   = mu2(k);
    
    dbins(:, inew)     = u1(:, k) /Nbatch;
    dbins(:, iY1(k))   = u2(:, k) /Nbatch;

    nspikes(inew, :)     = nspikes(iY1(k), :)/2;
    nspikes(iY1(k), :)   = nspikes(iY1(k), :)/2;
    dWUtot(:,:,inew)     = mu1(k)/mu0 * dWUtot(:,:,iY1(k)); %/npm(iY1(k));
    dWUtot(:,:,iY1(k))   = mu2(k)/mu0 * dWUtot(:,:,iY1(k)); %/npm(iY1(k));
end

d2d                 = pairwise_dists(dWUtot, WUinit);
dmatch              = min(d2d, [], 1);

[~, inovel] = sort(dmatch, 'descend');
% inovel = find(dmatch(1:1000)>.4);
% inovel = inovel(randperm(numel(inovel)));

i0 = 0;

for k = 1+min(nmerged+numel(freeInd), nsplit):nmerged+numel(freeInd)
    % add new clusters
    i0 = i0 + 1;
    if i0>numel(inovel)
        break;
    end
    if k<=numel(freeInd)
        inew= freeInd(k);
    else
        inew = isort(k - numel(freeInd));
    end
     
    dbins(:, inew)     = 1;
    
    nspikes(inew, :) = 1/8;
    
    
    dWUtot(:,:,inew)     = WUinit(:,:,inovel(i0)); %ratio * mu1(k)/mu0 * dWUtot(:,:,iY1(k));
    
end

nswitch = [min(nmerged, nsplit) i0]; %min(nmerged+numel(freeInd), nsplit);



function [score, iY, mu1, mu2, u1, u2] = split_clust(uu, nhist)

nhist = nhist(:);

nspikes = sum(uu, 1);

uc = zeros(size(uu));
for i = 1:size(uu,2)
    uc(:,i) = my_conv(uu(:,i)',  max(.5, min(4, 2000/nspikes(i))))'; %.5
%       uc(:,i) = my_conv2(uu(:,i),  max(.25, min(4, 2000/nspikes(i))), 1);
end
%
uc = uc ./repmat(sum(uc,1),size(uc,1), 1);
ucum = cumsum(uc, 1);
%
dd = diff(uc, 1);

iY = zeros(1000,1);
mu1 = zeros(1000,1);
mu2 = zeros(1000,1);
var1 = zeros(1000,1);
var2 = zeros(1000,1);
u1 = zeros(size(uu,1), 1000);
u2 = zeros(size(uu,1), 1000);

maxM = max(uc, [], 1);

inew = 0;

Nfilt = size(uu,2);
mu0 = sum(repmat(nhist(1:100, 1), 1, Nfilt) .* uc, 1);
var0 = sum((repmat(nhist(1:100), 1, Nfilt) - repmat(mu0, 100, 1)).^2 .* uc, 1);

for i = 1:Nfilt
    ix = find(dd(1:end-1, i)<0 & dd(2:end, i)>0);
    
    ix = ix(ucum(ix, i)>.1 & ucum(ix, i)<.8 & uc(ix,i)<.8 * maxM(i)); %.9 not .95
    if nspikes(i) > 500 && numel(ix)>0
        ix = ix(1);
        
        inew = inew + 1;
        
        normuc    = sum(uc(1:ix, i));
        mu1(inew) = sum(nhist(1:ix)     .* uc(1:ix, i))    /normuc;
        mu2(inew) = sum(nhist(1+ix:100) .* uc(1+ix:100, i))/(1-normuc);
        
        var1(inew) = sum((nhist(1:ix)-mu1(inew)).^2     .* uc(1:ix, i))    /normuc;
        var2(inew) = sum((nhist(1+ix:100)-mu2(inew)).^2 .* uc(1+ix:100, i))/(1-normuc);
        
        u1(1:ix,inew) = uu(1:ix, i);
        u2(1+ix:100,inew) = uu(1+ix:100, i);
        
        iY(inew) = i;
    end
    
end

mu1 = mu1(1:inew);
mu2 = mu2(1:inew);
var1 = var1(1:inew);
var2 = var2(1:inew);
u1 = u1(:,1:inew);
u2 = u2(:,1:inew);

n1 = sum(u1,1)';
n2 = sum(u2,1)';
iY = iY(1:inew);

score = 1 - (n1.*var1 + n2.*var2)./((n1+n2).*var0(iY)');
% score = ((n1+n2).*var0(iY)' - (n1.*var1 + n2.*var2))./var0(iY)';
[~, isort] = sort(score, 'descend');

iY = iY(isort);
mu1 = mu1(isort);
mu2 = mu2(isort);
u1 = u1(:,isort);
u2 = u2(:,isort);
score = score(isort);