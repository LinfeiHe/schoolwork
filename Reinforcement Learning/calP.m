close,clear,clc
%计算转移概率

P=zeros(4*12,12);
for index=1:size(P,1)%每行考虑
    if mod(index,4)==1%N
        state=(index-1)/4+1;
        if (state+4)==6 || (state+4)>12
            P(index,state)=P(index,state)+0.8;
        else
            P(index,state+4)=P(index,state+4)+0.8;
        end
        
        if (state+1)==6 || (state+1)==5 || (state+1)==9 || (state+1)==13
            P(index,state)=P(index,state)+0.1;
        else
            P(index,state+1)=P(index,state+1)+0.1;
        end
        
        if (state-1)==6 || (state-1)==0 || (state-1)==4 || (state-1)==8
            P(index,state)=P(index,state)+0.1;
        else
            P(index,state-1)=P(index,state-1)+0.1;
        end
    end
    
    if mod(index,4)==2%S
        state=(index-2)/4+1;
        if (state-4)==6 || (state-4)<1
            P(index,state)=P(index,state)+0.8;
        else
            P(index,state-4)=P(index,state-4)+0.8;
        end
        
        if (state+1)==6 || (state+1)==5 || (state+1)==9 || (state+1)==13
            P(index,state)=P(index,state)+0.1;
        else
            P(index,state+1)=P(index,state+1)+0.1;
        end
        
        if (state-1)==6 || (state-1)==0 || (state-1)==4 || (state-1)==8
            P(index,state)=P(index,state)+0.1;
        else
            P(index,state-1)=P(index,state-1)+0.1;
        end
    end
    
    if mod(index,4)==3%E
        state=(index-3)/4+1;
        if (state+1)==6 || (state+1)==5 || (state+1)==9 || (state+1)==13
            P(index,state)=P(index,state)+0.8;
        else
            P(index,state+1)=P(index,state+1)+0.8;
        end
        
        if (state+4)==6 || (state+4)>12
            P(index,state)=P(index,state)+0.1;
        else
            P(index,state+4)=P(index,state+4)+0.1;
        end
        
        if (state-4)==6 || (state-4)<1
            P(index,state)=P(index,state)+0.1;
        else
            P(index,state-4)=P(index,state-4)+0.1;
        end
    end
    
    if mod(index,4)==0%W
        state=index/4;
        if (state-1)==6 || (state-1)==0 || (state-1)==4 || (state-1)==8
            P(index,state)=P(index,state)+0.8;
        else
            P(index,state-1)=P(index,state-1)+0.8;
        end
        
        if (state+4)==6 || (state+4)>12
            P(index,state)=P(index,state)+0.1;
        else
            P(index,state+4)=P(index,state+4)+0.1;
        end
        
        if (state-4)==6 || (state-4)<1
            P(index,state)=P(index,state)+0.1;
        else
            P(index,state-4)=P(index,state-4)+0.1;
        end
    end
end

%然后要删除6 8 12状态对应的行21-24 29-32 45-48和列6，还要调整顺序
Prow1=P(1:20,:);
Prow2=P(25:28,:);
Prow3=P(33:44,:);
Prow=[Prow1;Prow2;Prow3];
Pcol1=Prow(:,1:5);
Pcol2=Prow(:,7);
Pcol3=Prow(:,9:11);
Pcol4=Prow(:,8);
Pcol5=Prow(:,12);
mp=[Pcol1 Pcol2 Pcol3 Pcol4 Pcol5];
save mp.mat mp