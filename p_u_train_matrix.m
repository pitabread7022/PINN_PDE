clc;
clear all;
x=linspace(0,1,1000);
x=transpose(x);
t=linspace(0,5,5000);
t=transpose(t);
p=zeros(length(x),length(t));
u=zeros(length(x),length(t));
for i = 1:length(x)
    for j=1:length(t)
        p(i,j)=sin(pi*x(i))*cos(-pi*t(j));
        u(i,j)=-sin(pi*t(j))*cos(pi*x(i));
    end
end
save('Training_Dataset.mat','p', 'u', 'x','t');
subplot(2,1,1);
surf(x,t,transpose(p),'edgecolor', 'none')
xlabel('x');
ylabel('t');
zlabel('p');
title(' Pressure ');

subplot(2,1,2); 
surf(x,t,transpose(u),'edgecolor', 'none')
xlabel('x');
ylabel('t');
zlabel('u');
title(' Velocity ');
savefig('NN_u_p');
