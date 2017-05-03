%% clear all
clear
clc

%% read audio
[aud, fs] = audioread('original.wav');
[M, N] = size(aud);

%% segmentation
leng = 4400;

aud_s = cell(floor(M / leng),1);
for i = 1 : floor(M / leng)
    aud_s{i,1} = aud( leng * ( i - 1 ) + 1: leng * i, : );
end


%% embeding
lamda = 0.2;
m0 = 200;
m1 = 300;
msg= 'the final work is completed by helinfei';
msg_bin = dec2bin(msg);
[R,Q] = size(msg_bin);

k = 1;
for i = 1:R
    for j = 1:Q
        if(msg_bin(i,j))
            m = m1;
        else
            m = m0;
        end
        aud_s{k,1}( m + 1 : leng, :) = aud_s{k,1}( m + 1 : leng, :) + lamda * aud_s{k,1}( 1 : leng - m, :);
        k = k + 1;
    end
end
%% stitching
aud_o = [];
for i = 1 : floor(M / leng)
    aud_o = cat(1,aud_o, aud_s{i,:});
end
aud_o = [aud_o; aud( floor(M / leng) * leng + 1 : end, :)];

%% write/output
audiowrite('embeded.wav', aud_o, fs);