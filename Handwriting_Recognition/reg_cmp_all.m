clear;clc;
load mnist_all.mat

%% setting
MINI_DATA = 1;      % 是否简化数据
M_TRAIN = 1000;     % 简化后每组训练集选M_TRAIN个进行试验
M_TEST = 200;      % 简化后每组测试集选M_TEST个进行试验
NET_MODE = 'rbf';   % 确定网络模型


%% get train and test data

train_in = [];
train_out = [];
test_in = [];
test_out = [];
for i = 0 : 9
    temp = zeros(1, 10);
    temp_train = eval(['train', num2str(i)]);
    temp_test = eval(['test', num2str(i)]);
    if MINI_DATA  % 简化数据
        n = randperm(length(temp_train));
        m = randperm(length(temp_test));
        n = n(1:M_TRAIN);
        m = m(1:M_TEST);
        temp(i+1) = 1;
        train_in = [train_in; temp_train(n, :)];
        train_out = [train_out; repmat(temp, M_TRAIN, 1)];
        test_in = [test_in; temp_test(m, : )];
        test_out = [test_out; repmat(temp, M_TEST, 1)];
    else
        temp(i+1) = 1;
        train_in = [train_in; temp_train];
        train_out = [train_out; repmat(temp, length(temp_train), 1)];
        test_in = [test_in; temp_test];
        test_out = [test_out; repmat(temp, length(temp_test), 1)];
    end
end
train_in = im2double(train_in);
test_in = im2double(test_in);

%% train
train_start = clock;
if strcmp(NET_MODE,'rbf')
    net = newrbe(train_in', train_out', 7);
elseif strcmp(NET_MODE,'pnn')
    tc_train = ind2vec(train_out');
    net = newrbe(train_in', tc_train, 9);
elseif strcmp(NET_MODE,'grnn')
    net = newgrnn(train_in', train_out', 1);
end
train_end = clock;
train_time = etime(train_end, train_start);
%% show accuracy

% train accuracy
% outputs = round(sim(net, train_in'));
% right = 0;
% for i = 1 : length(train_out)
%     if (outputs(:,i) == train_out(i,:)')
%         right = right + 1;
%     end
% end
% accuracy_train = right / length(train_out)

% test accuracy
test_start = clock;
outputs = round(sim(net, test_in'));
test_end = clock;
test_time = etime(test_end,test_start);
% right = 0;
% for i = 1 : length(test_out)
%     if (outputs(:,i) == test_out(i,:)')
%         right = right + 1;
%     end
% end
% accuracy_test = right / length(test_out)

train_time
test_time