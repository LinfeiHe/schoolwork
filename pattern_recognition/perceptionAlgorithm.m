clear;clc;
x0=[0,0,0,1];x1=[1,0,0,1];x2=[1,0,1,1];x3=[1,1,0,1];x4=[0,0,-1,-1];x5=[0,-1,0,-1];x6=[0,-1,-1,-1];x7=[-1,-1,-1,-1];
X=[x0;x1;x2;x3;x4;x5;x6;x7];
w0=[-1,-2,-2,0];
flag = 0;
for k = 1 : 200
    for i = 1 : 8
        d = w0 * X(i,:)';
        if (d <= 0)
            w0 = w0 + X(i,:);
            flag = 1;
        end
    end
    if (~flag)
        k,w0
        break;
    else
        flag = 0;
    end
end


[a,b]=meshgrid(1:0.1:10,1:0.1:10);
z=(3*a-2*b+1)./3;
mesh(a,b,z);




