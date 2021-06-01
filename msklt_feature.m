function feat = msklt_feature(imdist)
%------------------------------------------------
% Feature Computation
% imdist should be double format
%-------------------------------------------------
imdist = double(imdist);
%% MSCN
window = fspecial('gaussian',7,7/6);
window = window/sum(sum(window));
%% O
O1 = 0.3*imdist(:,:,1) + 0.04*imdist(:,:,2) - 0.35*imdist(:,:,3);
O2 = 0.34*imdist(:,:,1) - 0.6*imdist(:,:,2) + 0.17*imdist(:,:,3);
O3 = 0.06*imdist(:,:,1) + 0.63*imdist(:,:,2) + 0.27*imdist(:,:,3);
%%
mu = filter2(window, O1, 'same');
sigma = sqrt(abs(filter2(window, O1.*O1, 'same') - mu.*mu));
O1 = (O1-mu)./(sigma+1);

mu = filter2(window, O2, 'same');
sigma = sqrt(abs(filter2(window, O2.*O2, 'same') - mu.*mu));
O2 = (O2-mu)./(sigma+1);

mu = filter2(window, O3, 'same');
sigma = sqrt(abs(filter2(window, O3.*O3, 'same') - mu.*mu));
O3 = (O3-mu)./(sigma+1);

%%
feat = [];
blk_sz_st = [0 2 4];
scalenum = 3;
for itr_scale = 1 : scalenum
    blk_sz = blk_sz_st(itr_scale);
    %% load klt kernels
    if blk_sz>1
        load(['KLT/Kernels/KLT_kernel_MSCN_x',num2str(blk_sz)]);
        %%
        [hgt, wdt, ~] = size(imdist);
        hgt = floor(hgt/blk_sz)*blk_sz;
        wdt = floor(wdt/blk_sz)*blk_sz;
        X1 = im2col(O1(1:hgt,1:wdt),[blk_sz,blk_sz],'distinct')';
        X2 = im2col(O2(1:hgt,1:wdt),[blk_sz,blk_sz],'distinct')';
        X3 = im2col(O3(1:hgt,1:wdt),[blk_sz,blk_sz],'distinct')';
    else
        X1 = O1;
        X2 = O2;
        X3 = O3;
    end    
    %% Chanllel O1
    if blk_sz>1
        coef  = X1*kernel{1};
    else
        coef = X1(:);
    end
    featO1 = [];
    for i = 1:size(coef,2)
        [alpha, overallstd] = estimateggdparam(coef(:,i));
        featO1 = [featO1, alpha, overallstd];
    end
    %% Chanllel O2
    if blk_sz>1
        coef  = X2*kernel{2};
    else
        coef = X2(:);
    end
    featO2 = [];
    for i = 1:size(coef,2)
        [alpha, overallstd] = estimateggdparam(coef(:,i));
        featO2 = [featO2, alpha, overallstd]; 
    end
    %% Chanllel O3
    if blk_sz>1
        coef  = X3*kernel{3};
    else
        coef = X3(:);
    end
    featO3 = [];
    for i = 1:size(coef,2)
        [alpha, overallstd] = estimateggdparam(coef(:,i));
        featO3 = [featO3, alpha, overallstd]; 
    end
    %%
    %% feature fusion
    feat = [feat featO1 featO2 featO3];
    blk_sz = blk_sz * 2;
end

end
