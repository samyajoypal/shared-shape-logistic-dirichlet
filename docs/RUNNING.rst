Running the numerical studies
=============================

Run all commands from the repository root after installing the pinned
dependencies. The examples use separate output directories; do not reuse a
directory from another configuration. Large task ranges below describe the
complete study and can take substantial computing time. Each runner's
``--help`` describes its individual-task interface.

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
reproduces the published numerical displays without new fitting. The
commands above instead create new fits and reports. Numerical optimization
can vary with platform and library versions; retained attempts, convergence
flags and source/configuration hashes make such differences inspectable.
Use the same scientific settings for a reproduction and a new output
directory for a changed design. Do not replace unfavorable seeds, models,
datasets or failed attempts based on their outcomes.
