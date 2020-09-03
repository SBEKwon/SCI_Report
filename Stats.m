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

%% W_total
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

indct = [indc1;indc2;indc3];
indct = unique(indct);

f_features = ano_features;
f_features(:,indct) = [];
indc(indct) = [];

%% pain
Wp_class = zeros(length(W_total),1);
for i = 1:length(W_total)
    if W_pain(i) <8
        Wp_class(i) = 0;
    elseif W_pain(i) > 7 && W_pain(i) <13
        Wp_class(i) = 1;
    elseif W_pain(i) > 14
        Wp_class(i) = 2;
    end
end
indc_pain = [];

for i = 1:length(p_vals)
    if p_vals(i) < 0.001
        indc_pain = [indc_pain; i];
    end
end

pain_features = f_features(:,indc_pain);


%% stiff
Ws_class = zeros(length(W_total),1);
for i = 1:length(W_total)
    if W_stiff(i) <4
        Ws_class(i) = 0;
    elseif W_stiff(i) > 4 && W_stiff(i) <6
        Ws_class(i) = 1;
    elseif W_stiff(i) > 5
        Ws_class(i) = 2;
    end
end

indc_stiff = [];
for i = 1:length(p_vals)
    if p_vals(i) < 0.001
        indc_stiff = [indc_stiff; i];
    end
end

stiff_features = f_features(:,indc_stiff);


%% Function

Wpf_class = zeros(length(W_total),1);
for i = 1:length(W_total)
    if W_pf(i) <27
        Wpf_class(i) = 0;
    elseif W_pf(i) > 26 && W_pf(i) <43
        Wpf_class(i) = 1;
    elseif W_pf(i) > 42
        Wpf_class(i) = 2;
    end
end

indc_pf = [];
for i = 1:length(p_vals)
    if p_vals(i) < 0.001
        indc_pf = [indc_pf; i];
    end
end

pf_features = f_features(:,indc_pf);



