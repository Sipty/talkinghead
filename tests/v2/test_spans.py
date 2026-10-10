from talkinghead.spans import cluster_spans


def test_near_miss_boundaries_survive():
    out = cluster_spans([[(10, 20)], [(11, 20)], [(10, 19)]], agree=3, mode="narrow")
    assert [(s.a, s.b) for s in out] == [(11, 19)]


def test_wide_mode_takes_the_union():
    out = cluster_spans([[(10, 20)], [(11, 22)], [(9, 19)]], agree=3, mode="wide")
    assert [(s.a, s.b) for s in out] == [(9, 22)]


def test_below_threshold_is_dropped():
    assert cluster_spans([[(10, 20)], [(11, 20)]], agree=3, mode="narrow") == []


def test_disjoint_spans_cluster_separately():
    out = cluster_spans([[(1, 3), (40, 45)], [(2, 3), (41, 45)]], agree=2, mode="narrow")
    assert [(s.a, s.b) for s in out] == [(2, 3), (41, 45)]


def test_one_response_naming_a_span_twice_counts_once():
    out = cluster_spans([[(10, 20), (12, 18)], [(11, 19)]], agree=2, mode="narrow")
    assert [(s.a, s.b) for s in out] == [(11, 19)]


def test_votes_are_reported():
    out = cluster_spans([[(5, 9)], [(5, 9)], [(6, 9)]], agree=2, mode="narrow")
    assert out[0].votes == 3
