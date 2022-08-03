% Computing the nearest polynomial to multiple given polynomials
% via weighted $\ell_{2,q}$-norm minimization and its complex extension
% Jul. 29, 2022
% by Wenyu Hu

clc
clear all
close all
format long
% global B Ve;

L=2;
m = 4*L-1 ;    % function number
n = 4*L ;    % basis number
z = 1*sqrt(-1) ;    % -1, 2 % given zero
%z =2
tol = 1e-15;
q = 1.75;
w =ones(m,1);
W = 1*diag(w);
objective = [];
rho = [];

isConverged=0;
iterMax = 100 ;

%% Ez
% e_j(x), Ve
Ve=zeros(n,1);
for k=1:n
    Ve(k)=z^(k-1);
end
Re_e=real(Ve);
Im_e=imag(Ve);
Ez=[Re_e,Im_e;-Im_e,Re_e];
delta=norm(Re_e,2)^2+norm(Im_e,2)^2;

% Vf0
f0=@(x)1;
Vf0=zeros(m,1);
for i=1:m
    Vf0(i)=f0(z);  % vector f0
end
%% G
if m<=n
    % initialize A, Vf, B
    G=zeros(m,2*n); % 3X3
    G(:,1)=-1;
    for k=1:m
        G(k,k+1)=1;
    end
    G = double(G);
else
    disp(['error!!'])
end
%% F
Vf=zeros(m,1);
A=G(:,1:n);
Vf=Vf0+A*Ve;
F=[real(Vf),imag(Vf)];
%% B
B=zeros(m,m);
B=eye(m)-ones(m,1)*ones(m,1)'/m;
%% O
Ok=eye(m);

%% Y
Y = zeros(m,2*n);
iter = 0;

tic
while ~isConverged
    iter = iter+1;
    
    %% update Sigma
    Sigk = W'*Ok*W;
    
    %% updata Y
    Yk = Y;
    Y = ones(m,1)*ones(m,1)'*Sigk*G;
    Y = Y-ones(m,1)*(ones(m,1)'*Sigk*F)*Ez'/delta;
    Y = Y/(ones(m,1)'*Sigk*ones(m,1))-G;
    
    %% update Ok
    tau = 1e-15;
    Ok=zeros(m);
    WY = W*Y;    
    for k=1:m
        normWY=norm(WY(k,:),2);
        if normWY==0
            temp=tau;
        else
            temp=normWY;
        end
        temp=temp^(2-q);
        Ok(k,k)=1/temp;
    end
    
    %% Decision
    if iter >=1
        Ychg(iter) = norm(Y-Yk, inf);
        Res1 = norm(B*(Y+G),inf);  % BY=-BG
        Res2 = norm(Y*Ez+F,inf);  % YEz=-F
        rhok=max([Ychg(iter),Res1,Res2]);
        objective(iter) = compute_norm(W*Y,2,q);
        rho(iter) = rhok;
        if rhok<=tol
            isConverged = 1;
        end
    end
    if iter >iterMax
        break;
    end
end
TimeX = toc;

%% u
u = (Y+G)'*ones(m,1)/m;

%% compute c*
Vc = u(1:n)+u(n+1:2*n)*sqrt(-1)


% compute f(z)
disp(['### Output: q=' num2str(q)]);

disp(['### Functional value f(z): ']); % 对复数来说，a'代表了向量的共轭转置
f_val = f0(z)+transpose(Vc)*Ve


%     % output X*
%     disp(['### X*: ']);
%     X
% output minimal weighted distance:
disp(['### Minimal distance: '])
MinDist = compute_norm(W*Y,2,q)

%     % output distance \|f*-f_k\|_2
%     disp(['### Distances of each row : ']);
%     dist =[];
%     for k=1:m
%         dist(k) = norm(Y(k,:));
%     end
%     dist
        

figure(1), plot(1:iter,objective,'-rd','linewidth',1.5),xlabel('iteration number'),ylabel('||WY_k||_{2,q}^q')
figure(2), plot(1:iter,rho,'-bs','linewidth',1.5),xlabel('iteration number'),ylabel('\tau_k')
    
    
    
    