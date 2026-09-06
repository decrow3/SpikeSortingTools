import numpy as np
import pytest

from testing.luke_full_session_compare import common_time, correspondence, exclusive_count


def pop(times, clusters):
    return dict(st=np.array(times,dtype=np.int64), cl=np.array(clusters,dtype=np.int64))


def test_per_cluster_events_do_not_steal_each_other():
    a=pop([100,101,200,201],[1,2,1,2])
    b=pop([100,101,200,201],[11,12,11,12])
    result=correspondence(a,b,0)
    assert set(zip(result.baseline_cluster,result.candidate_cluster))=={(1,11),(2,12)}
    assert result.primary_match.all()
    assert (result.baseline_retention==1).all()


def test_duplicate_events_cannot_inflate_retention():
    assert exclusive_count(np.array([100,100]),np.array([100]),1)==1
    result=correspondence(pop([100,100,100],[1,1,1]),pop([100],[2]),1)
    assert result.matched_events.item()==1
    assert not result.primary_match.item()


def test_tied_split_is_ambiguous():
    result=correspondence(pop([100,200,300,400],[1]*4),pop([100,200,300,400],[2,2,3,3]),0)
    assert len(result)==2
    assert not result.primary_match.any()


def test_common_time_excludes_gaps_and_has_correct_direction():
    seconds,delta,pieces=common_time([[0,10,30],[20,30,10]],[[5,25,5]])
    assert seconds==10
    assert delta==15
    assert len(pieces)==2
    assert common_time([[0,1,10]],[[2,3,1]])[0]==0


def test_overlapping_windows_refused():
    with pytest.raises(ValueError):
        common_time([[0,10,5],[9,12,5]],[[0,20,5]])


def test_swapping_populations_preserves_pairs_and_exchanges_retention():
    a,b=pop([100,200,300],[1,1,1]),pop([100,200,301,500],[2,2,2,2])
    ab,ba=correspondence(a,b,1),correspondence(b,a,1)
    assert ab.jaccard.item()==ba.jaccard.item()
    assert ab.baseline_retention.item()==ba.candidate_retention.item()
