function [x, its, dk, ek, fk, time] = func_GAPGA(para, ProxJ, GradF, ObjPhi, xsol)
% Ngai-Son (2022), p. 642 experimental parameters; strong convexity = 0.
% Preserve the original pre-update dk/fk and post-update ek recording.
clock_id = tic;
n = para.n;
mu = para.mu;
gamma = para.gamma;
tol = para.tol;
maxits = para.maxits;
verbose = para.verbose;
x = para.x0;
y = x;
C = 4/gamma;
% Algorithm 1.1: y1=y0, x1=x0; initialize z1 before k=1.
z = ProxJ(y-(1/C)*GradF(y), mu/C);
dk = zeros(maxits, 1);
ek = zeros(maxits, 1);
fk = zeros(maxits, 1);
if verbose
    itsprint(sprintf('        step %08d: residual = %.3e', 1,1), 1);
end
for its = 1:maxits
    if verbose
        fk(its) = ObjPhi(x);
        dk(its) = norm(x-xsol, 'fro');
    end
    x_old = x;
    k = its;
    tau_k = 4/(k+4);
    alpha_k = k;
    beta_k = k/2; %#ok<NASGU>
    % Paper parameter retained for traceability. For alpha_k = k and
    % beta_k = k/2, its contribution is already absorbed into the closed-form
    % tau_k = 4/(k+4), so beta_k is not used separately below.
    x = ProxJ(y-gamma*GradF(y), mu*gamma);
    y_next = tau_k*z + (1-tau_k)*x;
    step = (alpha_k+1)/C;
    z = ProxJ(z-step*GradF(y_next), mu*step);
    y = y_next;
    res = norm(x_old-x, 'fro');
    ek(its) = res;
    if verbose && mod(its, 10) == 0
        itsprint(sprintf('        step %08d: residual = %.3e', its,res), its);
    end
    if (res/prod(n) < tol) || (res > 1e10)
        break;
    end
end
fprintf('\n');
dk = dk(1:its);
ek = ek(1:its);
fk = fk(1:its);
time = toc(clock_id);
end
