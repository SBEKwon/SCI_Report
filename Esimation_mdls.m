clear all
close all
clc
%%
load('W_right_feature.mat')

features = right_features(:,1:1083);
W_pain = right_features(:,1084);
W_stiff = right_features(:,1085);
W_pf = right_features(:,1086);
W_total = right_features(:,1087);

W_class = zeros(length(W_total),1);
for i = 1:length(W_total)
    if W_total(i) <37
        W_class(i) = 0;
    elseif W_total(i) > 36 && W_total(i) <61
        W_class(i) = 1;
    elseif W_total(i) > 60
        W_class(i) = 2;
    end
end


%%
[r,c] = size(features);
p_vals = [];
for i = 1:c
    p_vals(i) = anova1(features(:,i),W_class,'off');
end
indc = [];
for i = 1:length(p_vals)
    if p_vals(i) < 0.0001
        indc = [indc; i];
    end
end

ano_features = features(:,indc);

ind0 = []; ind1 = []; ind2 = [];
for i = 1:length(W_class)
    if W_class(i) == 0
        ind0 = [ind0; i];
    elseif W_class(i) == 1
        ind1 = [ind1; i];
    elseif W_class(i) == 2
        ind2 = [ind2; i];
    end
end


feature0 = ano_features(ind0,:);
feature1 = ano_features(ind1,:);
feature2 = ano_features(ind2,:);
class0 = W_class(ind0);
class1 = W_class(ind1);
class2 = W_class(ind2);

[r1,c1] = size(ano_features);
ttest_1 = []; ttest_2 = []; ttest_3 = [];
for i = 1:c1
    input1 = [feature0(:,i);feature1(:,i)];
    input2 = [class0;class1];
    input3 = [feature0(:,i);feature2(:,i)];
    input4 = [class0;class2];
    input5 = [feature1(:,i);feature2(:,i)];
    input6 = [class1;class2];
    [h,p1] = ttest(input1,input2);
    ttest_1 = [ttest_1;p1];
    [h,p2] = ttest(input3,input4);
    ttest_2 = [ttest_2;p2];
    [h,p3] = ttest(input5,input6);
    ttest_3 = [ttest_3;p3];
end

indc1 = []; indc2 = []; indc3 = [];
for i = 1:length(ttest_1)
    if ttest_1(i) > 0.00003
        indc1 = [indc1;i];
    end
end

for i = 1:length(ttest_2)
    if ttest_2(i) > 0.00003
        indc2 = [indc2;i];
    end
end

for i = 1:length(ttest_3)
    if ttest_3(i) > 0.00003
        indc3 = [indc3;i];
    end
end

indc = [indc1;indc2;indc3];
indc = unique(indc);

f_features = ano_features;
f_features(:,indc) = [];

%%

und_ind1 = [];
for i = 1:length(W_total)
    if W_total(i) < 60 && W_total(i) > 30
        und_ind1 = [und_ind1, i];
    end
end
r1 = randi([1 length(und_ind1)],1,180);
f_features(und_ind1(r1),:) = [];
W_total(und_ind1(r1),:) = [];


und_ind2 = [];
for i = 1:length(W_total)
    if W_total(i) < 31
        und_ind2 = [und_ind2, i];
    end
end
r2 = randi([1 length(und_ind2)],1,60);
f_features(und_ind2(r2),:) = [];
W_total(und_ind2(r2),:) = [];

W_score = W_total;

%% Linear model
constant = ones(size(W_score));
tw_independent_variable = [f_features, constant];
[tw_r,~,~,~,stat1] = regress(W_score,tw_independent_variable);
tw_est= sum(repmat(tw_r',length(tw_independent_variable),1).*tw_independent_variable,2);
tw_est = round(tw_est);

W_score(isnan(tw_est)) = [];
tw_est(isnan(tw_est)) = [];
tw_est_corr=corr(W_score, tw_est);

figure();
scatter(tw_est,W_score);axis([0,100,0,100]);
xlabel('Predicted WOMAC Score');ylabel('Actual WOMAC Score');title('WOMAC Estimation Model')
%
data2 =tw_est; data1 = W_score;
data_mean = mean([data1,data2],2);  % Mean of values from each instrument
data_diff = data1 - data2;              % Difference between data from each instrument
md = mean(data_diff);               % Mean of difference between instruments
sd = std(data_diff);                % Std dev of difference between instruments
figure,plot(data_mean,data_diff,'o','MarkerSize',3,'LineWidth',2)   % Bland Altman plot
hold on,plot(data_mean,md*ones(1,length(data_mean)),'-k')             % Mean difference line
plot(data_mean,2*sd*ones(1,length(data_mean)),'-k')                   % Mean plus 2*SD line
plot(data_mean,-2*sd*ones(1,length(data_mean)),'-k')                  % Mean minus 2*SD line
grid on
title('Bland Altman plot','FontSize',9)
xlabel('Mean of two measures','FontSize',8)
ylabel('Difference between two measures','FontSize',8)


isNotMissing = ~isnan(tw_est) & ~isnan(W_score);
validationRMSE = sqrt(nansum(( tw_est - W_score ).^2) / numel(W_score(isNotMissing) ));

%% ML estimation
cvp = cvpartition(size(W_total, 1), 'Holdout', 0.3);
trainingPredictors = f_features(cvp.training, :);
trainingResponse = W_total(cvp.training, :);
template = templateTree(...
    'MinLeafSize', 8);
regressionEnsemble = fitrensemble(...
    trainingPredictors, ...
    trainingResponse, ...
    'Method', 'Bag', ...
    'NumLearningCycles', 30, ...
    'Learners', template);
ensemblePredictFcn = @(x) predict(regressionEnsemble, x);
validationPredictFcn = @(x) ensemblePredictFcn(x);
validationPredictors = f_features(cvp.test, :);
validationResponse = W_total(cvp.test, :);
validationPredictions = validationPredictFcn(validationPredictors);

isNotMissing = ~isnan(validationPredictions) & ~isnan(validationResponse);
validationRMSE = sqrt(nansum(( validationPredictions - validationResponse ).^2) / numel(validationResponse(isNotMissing) ));

figure();
scatter(validationPredictions,validationResponse);axis([0,100,0,100]);
xlabel('Predicted WOMAC Score');ylabel('Actual WOMAC Score');title('Hold out')

data2 =validationPredictions; data1 = validationResponse;
data_mean = mean([data1,data2],2);  % Mean of values from each instrument
data_diff = data1 - data2;              % Difference between data from each instrument
md = mean(data_diff);               % Mean of difference between instruments
sd = std(data_diff);                % Std dev of difference between instruments
figure,plot(data_mean,data_diff,'ok','MarkerSize',5,'LineWidth',2)   % Bland Altman plot
hold on,plot(data_mean,md*ones(1,length(data_mean)),'-k')             % Mean difference line
plot(data_mean,2*sd*ones(1,length(data_mean)),'-k')                   % Mean plus 2*SD line
plot(data_mean,-2*sd*ones(1,length(data_mean)),'-k')                  % Mean minus 2*SD line
grid on
title('Bland Altman plot','FontSize',9)
xlabel('Mean of two measures','FontSize',8)
ylabel('Difference between two measures','FontSize',8)
