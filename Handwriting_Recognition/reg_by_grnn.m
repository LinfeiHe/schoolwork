clear;clc;
load mnist_all.mat

%% setting

MINI_DATA = 1;      % 是否简化数据
M_TRAIN = 1000;     % 简化后每组训练集选M_TRAIN个进行试验
M_TEST = 200;      % 简化后每组测试集选M_TEST个进行试验

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

%%% get best sigma
accuracy_train = zeros(1, 20);
accuracy_test = zeros(1, 20);
for k = 1:20
    %% train

    net = newgrnn(train_in', train_out', k * 0.1);

    %% accuracy

    % train accuracy
    outputs = round(sim(net, train_in'));
    right = 0;
    for i = 1 : length(train_out)
        if (outputs(:,i) == train_out(i,:)')
            right = right + 1;
        end
    end
    accuracy_train(k) = right / length(train_out);

    % test accuracy
    outputs = round(sim(net, test_in'));
    right = 0;
    for i = 1 : length(test_out)
        if (outputs(:,i) == test_out(i,:)')
            right = right + 1;
        end
    end
    accuracy_test(k) = right / length(test_out);
end

%% draw

[best_test_accuracy, best_sigma] = max(accuracy_test)
figure(1)
plot(0.1:0.1:2, accuracy_train, 'bo', 0.1:0.1:2, accuracy_test, 'r-*', best_sigma/10, best_test_accuracy, 'kx');
text(best_sigma/10-0.2, best_test_accuracy+0.02, ['best accuracy( ', num2str(best_sigma * 0.1), ' , ', num2str(best_test_accuracy),' )'])
title('train and test accuracy with different sigma by GRNN');
xlabel('不同sigma值');
ylabel('预测准确率');
legend('训练集准确率','测试集准确率', '预测峰值点');



