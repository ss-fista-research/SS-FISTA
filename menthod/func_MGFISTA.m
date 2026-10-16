function [x, its, dk, ek, fk, time] = func_MGFISTA(para, ProxJ, GradF, ObjPhi, xsol)
% Zhang--Liu--Tang (2025), fixed-step MG-FISTA, Algorithm 2.
% Image experiment parameters: a=3, p=0.6, exponent=0.98.
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
x_previous = x;
a = 3;
p = 0.6;
exponent = 0.98;
assert(p >= 0.5 && p <= 1 && exponent >= 0 && exponent <= 1);
if exponent > 0
    assert(a >= max(1,(2*exponent*p)^(1/exponent)));
else
    assert(a >= 1);
end
t = (p*(a-1)/a)^exponent;
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
    t_old = t;
    t = (p*(its+a-1)/a)^exponent;
    y = x + ((t_old-1)/t)*(x-x_previous);
    x = ProxJ(y-gamma*GradF(y), mu*gamma);
    x_previous = x_old;
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
