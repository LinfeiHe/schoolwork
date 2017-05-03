%% clear all
clear
clc

%% load embeded audio
load message.mat
[aud,fs] = audioread('embeded.wav');
[M, N] = size(aud);

%% detect
L = 273;
leng = 4400;
aud_s = cell(L,1);
for i = 1 : L
    aud_s{i,1} = aud( leng * ( i - 1 ) + 1: leng * i, : );
end


msg = zeros(L, 1);
for i = 1 : L
    XHAT=abs(ifft(log((abs(fft(aud_s{i,1}'))).^2)));
    if( XHAT(201) > XHAT(301) )
        msg(i) = 1;
    end
end


%% output message
k = num2str(msg);
s = reshape(msg_bin, 273,1);
accuracy = length(find(k == s))/273
