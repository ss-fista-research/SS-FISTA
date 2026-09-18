# Third-Party Notices

Parts of the experimental code for sparse logistic regression and
penalized principal component pursuit were adapted from:

Jingwei Liang, Tao Luo, and Carola-Bibiane Schönlieb,
Faster-FISTA

Original repository:
[https://github.com/jliang993/Faster-FISTA](https://github.com/jliang993/Faster-FISTA)

Upstream license file:
[https://github.com/jliang993/Faster-FISTA/blob/master/LICENSE](https://github.com/jliang993/Faster-FISTA/blob/master/LICENSE)

The upstream repository is distributed under the MIT License.

Copyright (c) 2018 Jingwei Liang

The following license text is reproduced from the upstream repository's
`LICENSE` file:

MIT License

Copyright (c) 2018 Jingwei Liang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Other third-party components requiring manual license review

The following files are included in this public package and retain their
original source attributions. Their exact redistribution terms could not be
established from an authoritative license text for the specific source
versions used here. No license is inferred from Faster-FISTA or from source
code similarity.

### Gabriel Peyré MATLAB toolbox files

Included files:

- `toolbox/perform_l1ball_projection.m`
- `toolbox/rescale.m`

The files retain their headers identifying Gabriel Peyré. The authoritative
source located for these routines is Gabriel Peyré's MATLAB toolbox
distribution, including the MATLAB Central File Exchange entry “Toolbox
Sparse Optmization”:

- [Gabriel Peyré MATLAB toolboxes](https://github.com/gpeyre/matlab-toolboxes)
- [Toolbox Sparse Optmization, MATLAB Central File Exchange](https://www.mathworks.com/matlabcentral/fileexchange/16204-toolbox-sparse-optmization)

The File Exchange record identifies Gabriel Peyré as the author and records
license changes across toolbox versions, but the exact license text for the
specific 1.5.0.0 source version used here is not exposed in the accessible
authoritative record.

**LICENSE STATUS UNRESOLVED — MANUAL REVIEW REQUIRED**

### SparseLab `SparseVector.m`

Included file:

- `toolbox/SparseVector.m`

The file retains its `SparseLab Version:100` attribution. The authoritative
source identified is the original SparseLab Version 100 distribution and its
official documentation. The SparseLab documentation states that the package
is copyrighted by the original authors and refers copying permissions to
`COPYING.m`; it also describes permission to distribute verbatim copies of the
entire package as a unit. That documentation does not establish a standalone
license for extracting and redistributing this single file in the present
package.

- [About SparseLab documentation](https://www.stodden.net/papers/AboutSparseLab.pdf)
- [SparseLab documentation mirror](https://citeseerx.ist.psu.edu/document?doi=f2d4f0517da5af0f49fd5b8c8b320c1b5566176f&repid=rep1&type=pdf)

**LICENSE STATUS UNRESOLVED — MANUAL REVIEW REQUIRED**

### Laurent Condat `prox_tv1D.m`

Included file:

- `toolbox/prox_tv1D.m`

The file retains its attribution to Laurent Condat and its “Version 2.0,
Aug. 3, 2017” header. The authoritative source identified is Laurent
Condat's official software page, which links the 2017 MATLAB implementation
and the related total-variation software:

- [Laurent Condat software page](https://lcondat.github.io/software.html)
- [Laurent Condat publications and supplementary software](https://lcondat.github.io/publications.html)
- [2017 MATLAB implementation linked by the official software page](https://lcondat.github.io/download/TV_Condat_v2.m)

The official source page does not provide an explicit license text for the
MATLAB file corresponding to the included source. No license is inferred from
third-party wrappers or forks.

**LICENSE STATUS UNRESOLVED — MANUAL REVIEW REQUIRED**
