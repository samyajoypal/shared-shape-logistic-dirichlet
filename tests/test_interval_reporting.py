"""Denominators, paired dependence and uncertainty for saved interval reports."""
import math

import numpy as np
import pandas as pd
import pytest

import interval_reporting as report


def interval_frame(available, covered, lengths, method="bootstrap_percentile"):
    return pd.DataFrame({
        "task_index": range(len(available)), "available": available, "covered": covered,
        "length": lengths, "lower": [0. if a else np.nan for a in available],
        "upper": lengths, "truth": 1., "full_draw_availability": False, "method": method,
    })


def test_conditional_and_all_sample_denominators_differ():
    frame=interval_frame([True,True,False,False],[True,False,None,None],[2.,4.,np.nan,np.nan])
    value=report.summarize_group(frame,4)
    assert value["available"]==2 and value["covered"]==1
    assert value["availability"]==.5
    assert value["conditional_coverage"]==.5
    assert value["covered_and_available"]==.25
    assert value["conditional_coverage_mcse"]==pytest.approx(math.sqrt(.25/2))
    assert value["covered_and_available_mcse"]==pytest.approx(math.sqrt(.25*.75/4))
    assert value["mean_length"]==3.
    assert value["mean_length_mcse"]==1.


def test_zero_available_is_na_not_zero_coverage():
    frame=interval_frame([False,False],[None,None],[np.nan,np.nan],"wald")
    value=report.summarize_group(frame,2)
    assert value["availability"]==value["covered_and_available"]==0.
    assert value["conditional_coverage"] is None
    assert value["conditional_coverage_mcse"] is None
    assert value["mean_length"] is None
    assert value["mean_length_mcse"] is None
    assert value["availability_mc95_high"]>0
    assert value["full_draw_available"] is None
    assert value["full_draw_availability"] is None


def test_perfect_small_sample_coverage_retains_exact_uncertainty():
    value=report.proportion(13,13)
    assert value["estimate"]==1 and value["mcse"]==0
    assert 0<value["mc95_low"]<.8
    assert value["mc95_high"]==1


def test_same_sample_pairing_and_paired_mcse():
    left=interval_frame([True,True,False,True],[True,False,None,True],[2.,4.,np.nan,6.],"wald")
    right=interval_frame([True,False,True,True],[False,None,True,True],[5.,np.nan,8.,10.])
    v=report.paired_group(left,right,4)
    assert v["both_available"]==2
    assert v["left_only_available"]==v["right_only_available"]==1
    assert v["neither_available"]==0
    assert v["left_coverage_on_common"]==1.
    assert v["right_coverage_on_common"]==.5
    assert v["coverage_difference"]==-.5
    assert v["coverage_difference_mcse"]==.5
    assert v["length_difference"]==3.5
    assert v["length_difference_mcse"]==.5
    assert v["left_mean_length_on_common"]==4.
    assert v["right_mean_length_on_common"]==7.5
    assert v["left_only_covers"]==1 and v["right_only_covers"]==0


def test_pairing_is_by_task_id_and_rejects_duplicate_ids():
    left=interval_frame([True,True],[True,False],[2.,3.],"wald")
    right=interval_frame([True,True],[False,True],[5.,6.])
    expected=report.paired_group(left,right,2)
    shuffled=report.paired_group(left,right.iloc[::-1],2)
    assert expected==shuffled
    with pytest.raises(pd.errors.MergeError):
        report.paired_group(pd.concat([left,left.iloc[:1]]),right,2)


def test_empty_common_sample_preserves_denominator_and_na():
    left=interval_frame([True,False],[True,None],[2.,np.nan],"wald")
    right=interval_frame([False,True],[None,False],[np.nan,4.])
    v=report.paired_group(left,right,2)
    assert v["planned"]==2 and v["both_available"]==0
    assert v["coverage_difference"] is None and v["coverage_difference_mcse"] is None
    assert v["length_difference"] is None and v["length_difference_mcse"] is None


def test_extreme_length_remains_visible_in_mean_mcse_and_tail():
    f=interval_frame([True]*3,[True]*3,[1.,1.,100.])
    v=report.summarize_group(f,3)
    assert v["mean_length"]==34
    assert v["mean_length_mcse"]==pytest.approx(33)
    assert v["median_length"]==1
    assert v["maximum_length"]==100
    assert v["largest_length_share"]==pytest.approx(100/102)


def test_full_draw_counts_exclude_intervals_that_are_unavailable():
    f=interval_frame([True,True,False],[True,False,None],[2.,4.,np.nan])
    f["full_draw_availability"]=[True,False,True]
    v=report.summarize_group(f,3)
    assert v["full_draw_available"]==1
    assert v["full_draw_availability"]==pytest.approx(1/3)
    assert v["full_draw_conditional_coverage"]==1


def test_boolean_strings_and_nonexclusive_flags_are_strict():
    assert report.strict_bool(pd.Series(["True","False",""]),missing=True).tolist()==[True,False,None]
    with pytest.raises(ValueError):report.strict_bool(pd.Series(["False","no"]))
    assert report.tokens("bound_hit|information_not_positive_definite")=={"bound_hit","information_not_positive_definite"}
    with pytest.raises(ValueError):report.tokens("bound_hit|unknown_reason")
    with pytest.raises(ValueError):report.tokens("bound_hit|bound_hit")


def test_one_available_length_has_unknown_mcse():
    assert report.mean_mcse([4.])==(4.,None)
    assert report.mean_mcse([])==(None,None)
