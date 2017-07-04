A=['N','S','E','W'];
policy=[A;A;A];
value=zeros(3,4);

value(3,1)=V(1);value(3,2)=V(2);value(3,3)=V(3);value(3,4)=V(4);
value(2,1)=V(5);value(2,3)=V(6);
value(1,1)=V(7);value(1,2)=V(8);value(1,3)=V(9);
value(2,4)=V(10);value(1,4)=V(11);

policy(3,1)=A(pi(1));policy(3,2)=A(pi(2));policy(3,3)=A(pi(3));policy(3,4)=A(pi(4));
policy(2,1)=A(pi(5));policy(2,3)=A(pi(6));
policy(1,1)=A(pi(7));policy(1,2)=A(pi(8));policy(1,3)=A(pi(9));
policy(2,2)='O';policy(1,4)='O';policy(2,4)='O';

value
policy