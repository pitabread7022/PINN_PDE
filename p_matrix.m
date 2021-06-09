% clc;
% clear all;
x=linspace(0,1,100);
x=transpose(x);
t=linspace(0,0.005,500);
t=transpose(t);
p=zeros(length(x),length(t));
for i = 1:length(x)
    p(i,1)=sin(pi*x(i));
end
save('acoustic_wave_t005_n500x100.mat','p','x','t');
