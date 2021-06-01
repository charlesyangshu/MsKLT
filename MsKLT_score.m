function qualityscore = MsKLT_score(imdist)
%% feature extraction
feat = msklt_feature(imdist);

%% Quality Score Computation
load('model_Color');
qualityscore = predict(model,feat);

end