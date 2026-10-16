function [x, its, dk, ek, fk, time] = func_NAGEL(para, ProxJ, GradF, ObjPhi, xsol)
% Verma--Shukla (2020), Algorithm 1 / Eq. (4), fixed-step NAGA.
% Algorithm 3 is the least-squares specialization named NAGEL.
% Use the same recurrence with the supplied gradient for other convex losses.
% Practical CD inertia alpha_k=(k-1)/(k+a), a=3; beta=3/4.
% Section 5.2 cites CD inertia without specifying a; a=3 is our explicit choice.
% This practical alpha_k tends to one: do not claim Theorem 1 (A4) covers it.
% Both proximal-gradient calls use gamma; no line search or restart is added.
clock_id = tic;
n = para.n;
mu = para.mu;
gamma = para.gamma;
tol = para.tol;
maxits = para.maxits;
verbose = para.verbose;
x = para.x0;
x_previous = x; % Paper initialization x_0 = x_1.
a = 3;
beta = 3/4;
assert(a > 2 && gamma > 0);
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
    alpha = (its-1)/(its+a);
    beta_k = beta;
    assert(beta_k > 0 && beta_k <= 1);
    y = x + alpha*(x-x_previous);
    u = ProxJ(y-gamma*GradF(y), mu*gamma);
    z = (1-beta_k)*y + beta_k*u;
    x = ProxJ(z-gamma*GradF(z), mu*gamma);
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