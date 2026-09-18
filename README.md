# SS-FISTA

MATLAB implementation of SS-FISTA and comparison methods.

## Acknowledgment of adapted experimental code

Parts of the experimental code for sparse logistic regression and
penalized principal component pursuit were adapted from the MATLAB
implementation accompanying:

J. Liang, T. Luo, and C.-B. Schönlieb,
“Improving ‘Fast Iterative Shrinkage-Thresholding Algorithm’:
Faster, Smarter and Greedier.”

Original implementation:
[https://github.com/jliang993/Faster-FISTA](https://github.com/jliang993/Faster-FISTA)

The corresponding experimental drivers and supporting routines in this
repository were subsequently modified to implement the algorithms,
parameter settings, preprocessing procedures, stopping criteria, and
evaluation protocol used in the present work. All numerical results
reported for the present work were generated using the current
implementation. The upstream implementation is acknowledged as a code
source only; the numerical results reported here are not taken from the
upstream repository.

## Requirements

- MATLAB R2022a
- MATLAB Wavelet Toolbox (`wthresh` is used by the sparse logistic and PCP
  scripts, and by the optional lasso configuration of the linear-inverse
  script)

## Included experiments

1. Sparse Logistic Regression (`splice`, `ionosphere`, `australian`, and
   `diabetes`)
2. Feasibility-Constrained Sparse Learning
3. Linear Inverse Problems
4. Penalized Principal Component Pursuit

## Main scripts

- `main_sparse_logistic.m`
- `main_constrained_sparse.m`
- `main_linear_inverse.m`
- `main_pcp.m`

## Algorithms

- FISTA
- FISTA-Mod
- GAPGA
- SS-FISTA

## Public PCP example

The repository includes the pre-cropped Lobby sequence as the directly
runnable penalized principal component pursuit example.  The public PCP data
directory contains only `data/pcp/Lobby.mat`, using the formal Lobby frame
window `1:300` (preview local index 90).  `main_pcp.m` constructs its
numerical reference solution internally with the local
`pcp_historical_reference` routine; that reference-only routine is not one of
the four benchmark algorithms and its CPU time and iterations are excluded
from the four reported method timings.  No external `*_reference.mat` file is
required.

The other PCP datasets used in the paper are not distributed through this
public repository.  Researchers who require these additional data for
reproducing the corresponding experiments may contact the corresponding
author using the contact information provided in the paper.

The four formal benchmark algorithms remain FISTA, FISTA-Mod, GAPGA, and the
current SS-FISTA implementation.  They receive the same numerical reference
and record the reference-error traces dk1, dk2, dk4, and dk5.  The driver
writes the paper-style Error = dk(end) values, generates the reference-error
convergence figure, and produces the original, sparse-foreground, and
low-rank-background Lobby figures.  Actual PCP objective values remain
available in the saved results as supplementary diagnostics.

`main_pcp.m` therefore loads the complete supplied `data.I` matrix directly
(`f = data.I`) and does not perform any additional runtime frame-window crop
or original-frame slicing.  The runtime `double` conversion and rescaling to
`[0,1]` remain part of the original PCP optimization protocol.  `Lobby.mat`
is already pre-cropped to original frames `1:300`, and no additional runtime
frame cropping is performed.

Each main script resolves paths relative to its own location.  Run a script
from any MATLAB working directory after opening this package.
