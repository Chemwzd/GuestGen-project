# Third-party notices

## ED2Mol (MIT License)

Parts of this repository are adapted from
[ED2Mol](https://github.com/pineappleK/ED2Mol):

* `src/edvae/ed2mol/vae.py` — the 3D convolutional encoder/decoder follows the
  ED generator architecture of ED2Mol.
* `src/fragment_growth/ed2mol/` — the fragment-growth engine (model wrappers,
  core placement, graph featurization, scoring and the growth loops) is a
  reorganized adaptation of the ED2Mol codebase used for this project.

ED2Mol is distributed under the MIT License:

```text
MIT License

Copyright (c) 2025 Pineapple

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
```

## External programs and packages

This project calls or depends on third-party software that is **not** included
in this repository and remains under its own license, including xTB, Multiwfn,
Open Babel, PyTorch, RDKit, PyVista/VTK, LMDB, torch-geometric, PLIP and
`supramolecular-sugar`. Refer to each project's license before redistribution.

## Data

The example structures, fragment/core libraries and model weights in this
repository are research data released for academic use together with the
corresponding publication. The full training dataset is distributed separately
from the source repository.
