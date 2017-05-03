clear;clc;
x0=[0,0,0,1];x1=[1,0,0,1];x2=[1,0,1,1];x3=[1,1,0,1];x4=[0,0,-1,-1];x5=[0,-1,0,-1];x6=[0,-1,-1,-1];x7=[-1,-1,-1,-1];
x=[x0;x1;x2;x3;x4;x5;x6;x7];
xinv=inv(x'*x)*x';
b = [1,1,1,1,1,1,1,1]';
for k = 1 : 200
    w = xinv * b;
    e = x*w - b;
    if(e ~= 0)
        w = w + xinv * abs(e);
        b = b + e + abs(e);
    else
        k,w'
        break;
    end
end
[a,b]=meshgrid(1:0.1:10,1:0.1:10);
z=(2*a-2*b+1)./2;
mesh(a,b,z);