% ----------------------------------------------------------------------- %
%    File_name: Calib.m
%    Programmer: Seungjae Yoo                             
%                                           
%    Last Modified: 2020_03_30                           
%                                                            
 % ----------------------------------------------------------------------- %
 function [P,V_train,X_train] = Calib(answer,ref)
 
data_label = string(answer(1,1));
m = double(string(answer(2,1)));
low_f = double(string(answer(3,1)));
high_f = double(string(answer(4,1)));
referencing = double(string(answer(5,1)));
 
FILENAME = strcat('C:\Users\유승재\Desktop\BCIIV_2a_mat\A0',data_label,'T_2_mat');
load(FILENAME);

s = 0.1*double(s);
cnt = s';
cnt(~isfinite(cnt)) = min(min(cnt)); % Encode Nan as the negative maximum value
%% EOG Remove
Y = cnt(1:22,:)';
U = cnt(23:25,:)';
C_nn = U'*U;
C_ny = U'*Y;
b = pinv(C_nn)*C_ny;

S = Y - U*b;

cnt = S';
%% Preprocessing
if referencing ~= 0
    %%% Calculate differential voltage
    for i = 1 : size(cnt,1)
        cnt(i,:) = cnt(i,:) - cnt(ref,:);
    end
    
    % common average
    if referencing == 1
        cnt_y = cnt; % Exclude electrode (AF3, AF4, O1, O2, PO1, PO2)
        Means = (1/size(cnt,1))*sum(cnt);
        for i = 1 : size(cnt_y,1)
            cnt_y(i,:) = cnt_y(i,:) - Means; % CAR
        end
        cnt_y = cnt_y([2:9 11:18],:);
        % LAP
    elseif referencing == 2
        cnt_n = myLAP(cnt,nfo); % Laplacian
        cnt_y = cnt_n([2:9 11:18],:); % Exclude electrode (AF3, AF4, O1, O2, PO1, PO2)
    end
else
    %%% Calculate differential voltage
    for i = 1 : size(cnt,1)
        cnt(i,:) = cnt(i,:) - cnt(ref,:);
    end
    
    cnt_y = cnt([2:9 11:18],:); % Exclude electrode (AF3, AF4, O1, O2, PO1, PO2)
end



%%
%BPF Design
 bpFilt = designfilt('bandpassiir','SampleRate',250,'PassbandFrequency1',low_f, ...
        'PassbandFrequency2',high_f,'StopbandFrequency1',low_f-2,'StopbandFrequency2',high_f+2, ...
        'StopbandAttenuation1',40,'StopbandAttenuation2',40, 'PassbandRipple',1,'DesignMethod','cheby2');
% Apply BPF
for i = 1:size(cnt_y,1)
    cnt_c(i,:) = filtfilt(bpFilt, cnt_y(i,:));
%     cnt_c(i,:) = filter(bpFilt, cnt_c(i,:));
end

%% Calculate covariance for all Classes
Class_1 = [];
Class_2 = [];
Class_3 = [];
Class_4 = [];

i=1;
while i <=length(h.EVENT.TYP)
    if h.EVENT.TYP(i)== 1023
        i = i + 2;
    elseif h.EVENT.TYP(i)== 769
        Class_1 = [Class_1 h.EVENT.POS(i)];
        i = i + 1;
    elseif h.EVENT.TYP(i)== 770
        Class_2 = [Class_2 h.EVENT.POS(i)];
        i = i + 1;
    elseif h.EVENT.TYP(i)== 771
        Class_3 = [Class_3 h.EVENT.POS(i)];
        i = i + 1;
    elseif h.EVENT.TYP(i)== 772
        Class_4 = [Class_4 h.EVENT.POS(i)];
        i = i + 1;
    else
        i = i + 1;
    end    
end

[N_1 Cov_1] = getCovariance(Class_1,cnt_c);
[N_2 Cov_2] = getCovariance(Class_2,cnt_c);
[N_3 Cov_3] = getCovariance(Class_3,cnt_c);
[N_4 Cov_4] = getCovariance(Class_4,cnt_c);
%% 
for i = 1:4
    if i == 1
        C_a = Cov_1;
        C_b = (N_2*Cov_2 + N_3*Cov_3 + N_4*Cov_4)/(N_2 + N_3 + N_4);
    elseif i == 2
        C_a = Cov_2;
        C_b = (N_1*Cov_1 + N_3*Cov_3 + N_4*Cov_4)/(N_1 + N_3 + N_4);
    elseif i ==3
        C_a = Cov_3;
        C_b = (N_2*Cov_2 + N_1*Cov_1 + N_4*Cov_4)/(N_2 + N_1 + N_4);
    else
        C_a = Cov_4;
        C_b = (N_2*Cov_2 + N_3*Cov_3 + N_1*Cov_1)/(N_2 + N_3 + N_1);
    end
    
    C_c = C_a + C_b;
    
    % EVD for composite covariance
    [V, D] = eig(C_c);
    
%     if ~isempty(find(diag(D)<0)), error("Nagative eigen value"); end
    
    % sort eigen vector with descend manner
    [d, ind] = sort(abs(diag(D)),'descend');
    D_new = diag(d);
    V_new = V(:,ind);
    
    % whitening transformation
    whiten_tf = V_new*D_new^(-0.5);
    W = whiten_tf';
    
    % Apply whitening to each averaged covariances
    Sa = W*C_a*W';
    Sb = W*C_b*W';
    
    
    % EVD for transformed covariance
    [U, phsi] = eig(Sb,Sa+Sb);
    if ~isempty(find(diag(phsi)<0)), error("Nagative eigen value"); end
  
    [d, ind] = sort(abs(diag(phsi)),'descend');
    phsi_new = diag(d);
    U_new = U(:,ind);
    
    P{i} = (U_new'*W)';
end
%% 

[train_1,train_R1,M_1,M_R1, Q_1, Q_R1] = getFeature_vector(Class_1,[Class_2 Class_3 Class_4],P{1},cnt_c,m);
[train_2,train_R2,M_2,M_R2, Q_2, Q_R2] = getFeature_vector(Class_2,[Class_1 Class_3 Class_4],P{2},cnt_c,m);
[train_3,train_R3,M_3,M_R3, Q_3, Q_R3] = getFeature_vector(Class_3,[Class_2 Class_1 Class_4],P{3},cnt_c,m);
[train_4,train_R4,M_4,M_R4, Q_4, Q_R4] = getFeature_vector(Class_4,[Class_2 Class_3 Class_1],P{4},cnt_c,m);

X_train{1,1} = train_1; X_train{1,2} = train_R1; M_train{1,1} = M_1; M_train{1,2} = M_R1; Q_train{1,1} = Q_1; Q_train{1,2} = Q_R1;
X_train{2,1} = train_2; X_train{2,2} = train_R2; M_train{2,1} = M_2; M_train{2,2} = M_R2; Q_train{2,1} = Q_2; Q_train{2,2} = Q_R2;
X_train{3,1} = train_3; X_train{3,2} = train_R3; M_train{3,1} = M_3; M_train{3,2} = M_R3; Q_train{3,1} = Q_3; Q_train{3,2} = Q_R3;
X_train{4,1} = train_4; X_train{4,2} = train_R4; M_train{4,1} = M_4; M_train{4,2} = M_R4; Q_train{4,1} = Q_4; Q_train{4,2} = Q_R4;
%%
for k = 1:4
    
    X1 = X_train{k,1};
    X2 = X_train{k,2};
    
    M_1 = M_train{k,1};
    M_2 = M_train{k,2};
    C_1 = Q_train{k,1};
    C_2 = Q_train{k,2};
    
    Sb = (M_1-M_2)*(M_1-M_2)';
    Sw = C_1+C_2;
    tmp = pinv(Sb)*Sw;
    
    [V, D] = eig(tmp);
    [d, ind] = sort(abs(diag(D)),'descend');
    D_new = diag(d);
    V_new = V(:,ind);
    
    V_train{k,1} = V_new(:,1); 
  
end
%%%%%%%%%% Kernel density plot %%%%%%%%%%%%%%%%%%
% for k = 1:4
%     
%     data1 = V_train{k,1}'*X_train{k,1};
%     data2 = V_train{k,1}'*X_train{k,2};
%     
%     figure
%     subplot(2,1,1)
%     histogram(data1); hold on;
%     histogram(data2); hold on;
%     legend
%     
%     syms x y;
%     h1 = ((4/(3*length(data1)))^0.2)*std(data1);
%     h2 = ((4/(3*length(data2)))^0.2)*std(data2);
%     phi_1 = (1/sqrt(2*pi))*exp(-y^2/(2*h1^2));
%     phi_2 = (1/sqrt(2*pi))*exp(-y^2/(2*h2^2));
%     
%     p_1 = 0;
%     for i= 1:length(data1)
%         p_1 = p_1 + subs(phi_1,x-data1(i));
%     end
%     p_1 = p_1/length(data1);
%     
%     p_2 = 0;
%     for i= 1:length(data2)
%         p_2 = p_2 + subs(phi_2,x-data2(i));
%     end
%     p_2 = p_2/length(data2);
%     
%     
%     subplot(2,1,2)
%     ezplot(p_1); hold on;
%     ezplot(p_2); hold on;
%     legend
% end
%%%%%%%%%% feature vector scatter plot %%%%%%%%%%%%%%%%%%
% for k = 1:4
%     
%     data1 = X_train{k,1};
%     data2 = X_train{k,2};
%     
%     figure  
%     
%     for i= 1:length(data1)
%         fp = data1(:,i);
%         scatter3(fp(1),fp(2),fp(4),'r'); hold on;
%     end
%     for i= 1:length(data2)
%         fp = data2(:,i);
%         scatter3(fp(1),fp(2),fp(4),'b'); hold on;
%     end
%     
%    
% end


 end
