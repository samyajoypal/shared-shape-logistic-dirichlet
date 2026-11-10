Running the numerical studies
=============================

Run all commands from the repository root after installing the pinned
dependencies. The examples use separate output directories; do not reuse a
directory from another configuration. Large task ranges below describe the
complete study and can take substantial computing time. Each runner's
``--help`` describes its individual-task interface.

For the two focused numerical displays and all expanded comparison tables,
no new fits are needed::

    .venv/bin/python scripts/reproduce_saved_tables.py --output artifacts/saved_tables

The focused application display uses six-acid olive oil and Abalone. The
complete application designs below also retain the other recorded
comparisons. Their task ranges and scientific settings are unchanged.

Primary simulations and Gaussian companion
-------------------------------------------

The primary design has 10,500 tasks. Freeze each design once, execute all
indices, then aggregate. Finish the primary tasks before their companion
fits, which depend on the same generated datasets::

    .venv/bin/python scripts/phase2_simulations.py --freeze --output results/simulations
    .venv/bin/python scripts/phase2_simulations.py --output results/simulations --task-range 0:10500
    .venv/bin/python scripts/phase2_simulations.py --output results/simulations --aggregate
    .venv/bin/python scripts/phase2_gaussian_control.py --freeze --source-kind simulations --primary-run results/simulations --output results/gaussian_simulations
    .venv/bin/python scripts/phase2_gaussian_control.py --output results/gaussian_simulations --task-range 0:10500
    .venv/bin/python scripts/phase2_gaussian_control.py --output results/gaussian_simulations --aggregate

Conditional-mean risk uses the existing fitted models in the designated
6,000 primary datasets. It evaluates a new target without refitting them.
The binding command records the newly reproduced upstream runs; it does
not describe a fresh rerun as a historical preregistration::

    .venv/bin/python scripts/public_workflows.py bind-risk --primary results/simulations --gaussian results/gaussian_simulations --output results/conditional_risk
    .venv/bin/python scripts/simulation_conditional_mean_risk.py --output results/conditional_risk --task-range 0:6000
    .venv/bin/python scripts/simulation_conditional_mean_risk.py --output results/conditional_risk --aggregate
    .venv/bin/python scripts/report_simulation_conditional_risk.py --frozen-report results/conditional_risk --output results/conditional_risk_summary

Interval study
--------------

The interval runner records each outer sample separately and does not need
a separate freeze command. The full design includes 4,500 outer samples
with bootstrap refitting::

    .venv/bin/python scripts/phase2_inference.py --output results/intervals --task-range 0:4500
    .venv/bin/python scripts/phase2_inference.py --output results/intervals --aggregate

Original applications
---------------------

Prepare the data first. The original-application design contains 903 tasks;
the Gaussian companion has 750 tasks, followed by 18 geographic-transfer
fits. Refined application fits use all of these completed inputs::

    .venv/bin/python scripts/fetch_data.py --datasets original
    .venv/bin/python scripts/phase2_applications.py --freeze --output results/original_applications
    .venv/bin/python scripts/public_workflows.py applications --run results/original_applications --task-range 0:903
    .venv/bin/python scripts/phase2_applications.py --output results/original_applications --aggregate
    .venv/bin/python scripts/phase2_gaussian_control.py --freeze --source-kind applications --primary-run results/original_applications --output results/gaussian_applications
    .venv/bin/python scripts/phase2_gaussian_control.py --output results/gaussian_applications --task-range 0:750
    .venv/bin/python scripts/phase2_gaussian_control.py --output results/gaussian_applications --aggregate
    .venv/bin/python scripts/phase2_gaussian_transport_followup.py --freeze --primary-run results/original_applications --output results/gaussian_transport
    .venv/bin/python scripts/phase2_gaussian_transport_followup.py --output results/gaussian_transport --task-range 0:18
    .venv/bin/python scripts/phase2_gaussian_transport_followup.py --output results/gaussian_transport --aggregate
    .venv/bin/python scripts/phase2_application_refinement.py --freeze --primary-run results/original_applications --gaussian-run results/gaussian_applications --transport-run results/gaussian_transport --output results/refined_applications
    .venv/bin/python scripts/phase2_application_refinement.py --output results/refined_applications --task-range 0:783
    .venv/bin/python scripts/phase2_application_refinement.py --output results/refined_applications --aggregate --phase all

Fit the log-linear reference on the original splits, then evaluate the same
conditional-mean target for the mixture and regression procedures::

    .venv/bin/python scripts/assess_old_conditional_reference.py --primary results/original_applications --output results/original_reference
    .venv/bin/python scripts/assess_conditional_mean_prediction.py --primary results/original_applications --gaussian results/gaussian_applications --transport results/gaussian_transport --refined results/refined_applications --reference results/original_reference --output results/original_mean
    .venv/bin/python scripts/assess_conditional_prediction_stability.py --primary results/original_applications --gaussian results/gaussian_applications --refined results/refined_applications --output results/original_stability
    .venv/bin/python scripts/assess_old_conditional_reference.py --reference-run results/original_reference --mixture-stability results/original_stability --output results/original_stability_with_reference

Additional applications
-----------------------

The additional panel contains Hydrochem, GEMAS and Abalone. Run all 198
prediction tasks before the 597 bootstrap tasks, then aggregate both phases::

    .venv/bin/python scripts/fetch_data.py --datasets panel
    .venv/bin/python scripts/phase3_applications.py --freeze --output results/additional_applications
    .venv/bin/python scripts/phase3_applications.py --output results/additional_applications --task-range 0:198
    .venv/bin/python scripts/phase3_applications.py --output results/additional_applications --task-range 198:795
    .venv/bin/python scripts/phase3_applications.py --output results/additional_applications --aggregate --phase all
    .venv/bin/python scripts/assess_phase3_predictions.py --run results/additional_applications --output results/additional_density --retrieval-complete
    .venv/bin/python scripts/assess_conditional_mean_prediction.py --phase3-run results/additional_applications --output results/additional_mean
    .venv/bin/python scripts/assess_phase3_stability.py --run results/additional_applications --output results/additional_stability --retrieval-complete

Use ``--retrieval-complete`` only when all intended tasks are present in
the selected run directory. It states the completeness of the local input
collection; it does not turn failed fits into successful fits. Reports retain
planned denominators and unavailable outcomes.

Saved results versus fresh runs
------------------------------

``scripts/reproduce_saved_tables.py`` reads the supplied summaries and
reproduces the focused numerical displays and expanded comparisons without
new fitting. The
commands above instead create new fits and reports. Numerical optimization
can vary with platform and library versions; retained attempts, convergence
flags and source/configuration hashes make such differences inspectable.
Use the same scientific settings for a reproduction and a new output
directory for a changed design. Do not replace unfavorable seeds, models,
datasets or failed attempts based on their outcomes.

Complete saved numerical reporting
---------------------------------

For the quick path, without model fitting or downloads, run from the
repository root::

    .venv/bin/python scripts/reproduce_saved_tables.py --output artifacts/saved_tables

Use a new output directory. The 23 CSV outputs comprise the twelve existing
tables, five complete interval tables, and six complete primary-simulation
and conditional-risk-subset tables. Every supplied numeric input is checked
against ``export_manifest.json`` before reporting. The twelve existing CSV
outputs are also covered by byte-preservation regression tests.

The additional reports alone can be generated with::

    .venv/bin/python scripts/complete_saved_reporting.py --output artifacts/complete_reports

This command writes ``complete_intervals/``, ``complete_simulations/`` and a
manifest recording relative input/output hashes. Interval tables are
recalculated from 54,000 saved method-by-target records and 4,500 outer
diagnostics. The simulation tables are checked saved summaries covering
all 65 settings, eight comparators and nine fitted methods. Recomputing
their estimates from fitted parameters instead requires the full simulation
runs described earlier in this document; no such fit is performed by either reporting command.

The numeric interval module can also summarize the supplied extracted
records directly::

    .venv/bin/python scripts/interval_reporting.py --raw saved_summaries/interval_records --output artifacts/interval_report

For a separately generated complete study, ``--frozen DIRECTORY`` selects
the location of ``aggregate.json`` and ``production_configuration.json``;
otherwise those two files are read from ``--raw``. The module verifies all
4,500 task identities, all 108 setting/target/method rows and 81 paired
comparisons against the aggregate. Its record schema and thresholds are
the fixed nine-setting design supplied here. The top-level quick command
additionally verifies that the supplied record bytes match the public
export hashes.

The interval MCSE uses independent outer samples. Bootstrap-draw counts are
descriptive within-sample totals. Conditional coverage and length use
available intervals; each paired difference uses the same available outer
samples. Undefined results remain missing when no interval is available.
Exact binomial Monte Carlo intervals remain nondegenerate when a plug-in
MCSE is zero. No complete bootstrap-fit bundles are needed for these
reporting commands, and none are included.
