close,clear

S=[1 1;2 1;3 1;4 1;1 2;3 2;1 3;2 3;3 3;4 2;4 3];%ÿһ��һ��״̬
A=[1;2;3;4];%ÿһ��һ���������ֱ����N,S,E,W
R=-0.02*ones(size(S,1),1);%ȡS���У�ÿһ�д����״̬��reward
R(10)=-1;R(11)=1;
gamma=0.99;

load mp

pi=[3 3 1 1 2 3 3 3 3]';
%pi=ceil(rand(9,1)*size(A,1));%��ʼ����ȡ
pipre=zeros(9,1);
count=0;

while(max(abs(pi-pipre))>0) %PV����
    pipre=pi;
    PA=zeros(size(S,1),size(S,1));
    for index=1:9
        PA(index,:)=mp(size(A,1)*(index-1)+pipre(index),:);
    end
    V=(eye(11)-gamma*PA)\R;
    
    for index=1:9
        [mavval maxpos]=max(mp(size(A,1)*(index-1)+1:size(A,1)*index,:)*V);
        pi(index)=maxpos;
    end
    count=count+1;
end

show