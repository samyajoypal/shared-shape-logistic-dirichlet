Shared-shape logistic-Dirichlet mixtures
======================================

Numerical implementations and reproducibility materials for studying shape
sharing in logistic-Dirichlet mixtures. Components share a positive shape
vector and have their own locations and positive coordinate scales. The
comparisons include separate-shape logistic-Dirichlet mixtures, a Gaussian
mixture with the same correlation restriction, and other mixture families.

Repository: https://github.com/samyajoypal/shared-shape-logistic-dirichlet

The repository supports two distinct tasks:

* Regenerate the eight main numerical result tables from the supplied,
  hash-checked summaries. This is quick and does not fit models.
* Rerun the statistical studies from data and recorded simulation seeds.
  These runs perform the original optimization and bootstrap calculations.

Installation
------------

Use Python 3.12 on Linux or a compatible Unix system. Some runners use Unix
process and file-lock facilities. From the repository root::

    python3.12 -m venv .venv
    .venv/bin/python -m pip install -r requirements-lock.txt

The environment pins ``fmvmm==3.0.2`` and the numerical library versions.
Use the same Python patch version and dependencies throughout a frozen run.
Fresh runs record their environment and refuse incompatible resumes.

Reproduce the supplied result tables
-----------------------------------

Run::

    .venv/bin/python scripts/reproduce_saved_tables.py --output artifacts/saved_tables

The command verifies the input hashes, writes eight main-table CSV files,
three complete supplementary result CSV files, and a reproduction manifest.
Choose a new output directory to run it again. It never overwrites an
existing output directory or selects new model fits.

The main outputs are:

* ``table_01_core_simulations.csv``: shared-shape simulation settings.
* ``table_02_original_density.csv``: original-application density prediction.
* ``table_03_original_mean_prediction.csv``: original-application prediction loss.
* ``table_04_geographic_transfer.csv``: geographic transfer comparisons.
* ``table_05_original_numerical_diagnostics.csv``: numerical checks.
* ``table_06_original_stability.csv``: conditional-mean resampling variation.
* ``table_07_additional_mean_and_stability.csv``: additional-application loss,
  parameter counts and resampling variation.
* ``table_08_additional_density.csv``: additional-application density contrasts.

These are numerical tables, with more diagnostic columns and precision than
the printed displays. ``saved_summaries/`` retains the broader comparisons,
unfavorable results, available-fit counts, intervals and sensitivity results.
It contains summaries, not all fitted-model or bootstrap checkpoints. Full
reruns produce new checkpoints from the supplied code and configurations.

Data
----

The UCI Abalone and Seeds data are included with attribution and their
CC BY 4.0 terms. Other application data are obtained from their public
sources by an explicit download command::

    .venv/bin/python scripts/fetch_data.py

This prepares olive oil, crabs, Hydrochem, GEMAS and Old Faithful, verifies
the source hashes and records prepared-data hashes. ``--datasets original``, ``panel`` or
``benchmarks`` prepares a smaller group. The scripts stop if upstream bytes
change; they do not silently replace a scientific input. Source URLs,
measurement descriptions and applicable notices are in ``data/`` and
``LICENSES/DATA_LICENSES.rst``. Download access is required only for fetching
these inputs, not for reproducing the supplied summary tables.

Tests and full experiments
-------------------------

After preparing the data, run::

    .venv/bin/python -m pytest -q

The tests cover densities, transformations, gradients, information matrices,
conditional predictions, fitting safeguards, deterministic task allocation,
and retention of failed attempts. The full study is separate from the test
suite: it includes 10,500 primary simulated datasets, an interval study with
4,500 outer samples, and repeated application and bootstrap fits.

``docs/RUNNING.rst`` gives the full command sequence, including companion
comparators, refinement, conditional-mean risk and reporting. Use fresh
output directories under ``results/``. Task ranges are half-open: ``0:10``
runs indices 0 through 9. Independent task indices may run on separate
workers after the design is frozen; aggregation reports every planned task,
including unavailable or failed fits.

Code and interpretation
-----------------------

``scripts/shared_shape_ld.py`` implements the LD density, derivatives,
sampling and fitting. ``ld_information.py`` implements observed information;
``gaussian_shared_correlation.py`` supplies the matched Gaussian control;
``regularized_skew_normal.py`` supplies the bounded skew-normal comparator.
The ``phase2_*`` runners implement the primary studies and ``phase3_*`` the
additional applications. ``configs/`` records the scientific designs.

Shape sharing is a modelling restriction, not a claim of universal
superiority. Conditional-mean estimation error, squared prediction loss
against an observed response, density prediction and bootstrap variation
are different outcomes. The output records preserve these distinctions,
optimization diagnostics, and the exploratory status of additional analyses.

Rights and attribution
----------------------

See ``LICENSES/NOTICE.rst`` for the original-code notice and
``LICENSES/DATA_LICENSES.rst`` for third-party data terms. Dependencies are
installed from their original distributions. The flexible-mixture framework
is described by Pal and Heumann, *International Statistical Review* 94(1),
61-96 (2026), https://doi.org/10.1111/insr.12593; the experiments use the
separately versioned ``fmvmm`` 3.0.2 package.
