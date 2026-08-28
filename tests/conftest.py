"""Session-wide test setup.

Caps the OpenMP thread count before anything imports ``torch``.

Without this the suite hangs. ``extraction``'s integration tests are the first
thing in the run to import ``torch``, which starts an OpenMP pool sized to the
machine; ``battery`` and ``simulation`` then run scikit-learn, which has an
OpenMP pool of its own. The two runtimes oversubscribe the cores and a
``simulation`` test that takes six seconds on its own stops finishing -- at 394%
CPU, so it reads as a hang rather than as slowness, and it surfaces in a test
that has nothing to do with the one that caused it.

``pyproject.toml`` already pins one OpenMP runtime from conda-forge for a
related reason. This is the test-side half of the same problem, and it is set
as an environment variable rather than via ``torch.set_num_threads`` because it
has to be in place *before* the import, not after.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
