function [x, its, dk, ek, fk, time] = func_FISTA(para, ProxJ, GradF, ObjPhi, xsol)
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
t = 1;
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
    x = ProxJ(y-gamma*GradF(y), mu*gamma);
    t_old = t;
    t = (1 + sqrt(1 + 4*t_old^2))/2;
    y = x + ((t_old-1)/t)*(x-x_old);
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
