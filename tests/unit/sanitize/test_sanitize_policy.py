"""Policy engine: thresholds, overlap resolution, placeholders, drop_chunk."""

from __future__ import annotations

from zllf.stages.sanitize.policy import (
    Detection,
    PlaceholderMapper,
    Policy,
    apply_masks,
    chunk_should_drop,
    cluster_overlaps,
    normalize_value,
    policy_from_config,
    resolve,
)

POLICY = Policy(actions={"PERSON": "mask", "LOCATION": "keep",
                         "CLASSIFICATION_MARKING": "drop_chunk"},
                default_action="mask", score_threshold=0.5)


def test_threshold_filters_low_scores() -> None:
    spans = resolve([Detection(0, 4, "PERSON", 0.49), Detection(10, 14, "PERSON", 0.5)], POLICY)
    assert [(s.start, s.end) for s in spans] == [(10, 14)]


def test_overlap_highest_score_wins_and_union_is_masked() -> None:
    dets = [Detection(0, 10, "LOCATION", 0.6), Detection(5, 15, "PERSON", 0.9)]
    [span] = resolve(dets, POLICY)
    assert span.entity_type == "PERSON" and span.action == "mask"
    assert (span.start, span.end) == (0, 15)  # union: no fragment of the loser leaks
    assert span.members == 2


def test_overlap_tie_on_score_longest_wins() -> None:
    dets = [Detection(0, 5, "PHONE_NUMBER", 0.8), Detection(0, 9, "PATIENT_ID", 0.8)]
    [span] = resolve(dets, POLICY)
    assert span.entity_type == "PATIENT_ID"


def test_resolution_is_order_independent() -> None:
    dets = [Detection(0, 5, "A", 0.8), Detection(0, 5, "B", 0.8), Detection(3, 8, "C", 0.7),
            Detection(20, 25, "D", 0.9)]
    assert resolve(dets, POLICY) == resolve(list(reversed(dets)), POLICY)
    assert resolve(dets, POLICY)[0].entity_type == "A"  # final tie-break: type name


def test_keep_winner_keeps_only_its_span() -> None:
    dets = [Detection(0, 10, "LOCATION", 0.9), Detection(8, 12, "PERSON", 0.6)]
    [span] = resolve(dets, POLICY)
    assert span.action == "keep" and (span.start, span.end) == (0, 10)


def test_clusters_chain_transitively() -> None:
    dets = [Detection(0, 5, "A", 1), Detection(4, 9, "B", 1), Detection(8, 12, "C", 1),
            Detection(12, 14, "D", 1)]
    assert [len(c) for c in cluster_overlaps(dets)] == [3, 1]


def test_drop_chunk_is_strict() -> None:
    assert chunk_should_drop([Detection(0, 6, "CLASSIFICATION_MARKING", 0.9)], POLICY)
    assert not chunk_should_drop([Detection(0, 6, "CLASSIFICATION_MARKING", 0.4)], POLICY)
    # loses the overlap but still drops the chunk
    dets = [Detection(0, 6, "CLASSIFICATION_MARKING", 0.6), Detection(0, 8, "PERSON", 0.9)]
    assert resolve(dets, POLICY)[0].entity_type == "PERSON"
    assert chunk_should_drop(dets, POLICY)


def test_placeholders_consistent_within_document() -> None:
    mapper = PlaceholderMapper(b"salt")
    text = "Anna Muster met Ben Beispiel. Later ANNA  MUSTER called."
    dets = [Detection(0, 11, "PERSON", 0.9), Detection(16, 28, "PERSON", 0.9),
            Detection(36, 48, "PERSON", 0.9)]
    masked, phs = apply_masks(text, resolve(dets, POLICY), "doc1", mapper)
    assert masked == "[PERSON_1] met [PERSON_2]. Later [PERSON_1] called."
    assert phs == ["[PERSON_1]", "[PERSON_2]", "[PERSON_1]"]
    # another document restarts numbering; the same doc keeps its mapping
    assert mapper.placeholder("doc2", "PERSON", "Ben Beispiel") == "[PERSON_1]"
    assert mapper.placeholder("doc1", "PERSON", "Ben Beispiel") == "[PERSON_2]"
    assert mapper.placeholder("doc1", "EMAIL_ADDRESS", "a@b.de") == "[EMAIL_ADDRESS_1]"


def test_placeholder_mapping_holds_no_raw_values() -> None:
    mapper = PlaceholderMapper(b"salt")
    mapper.placeholder("d", "PERSON", "Very Secret Name")
    assert "Very Secret Name".lower() not in repr(mapper.__dict__).lower()


def test_keep_spans_are_not_masked() -> None:
    text = "Visit Berlin with Anna"
    dets = [Detection(6, 12, "LOCATION", 0.9), Detection(18, 22, "PERSON", 0.9)]
    masked, phs = apply_masks(text, resolve(dets, POLICY), "d", PlaceholderMapper(b"s"))
    assert masked == "Visit Berlin with [PERSON_1]"
    assert phs == [None, "[PERSON_1]"]


def test_normalize_value_compacts_ids() -> None:
    assert normalize_value("030 / 123-45", "PHONE_NUMBER") == "03012345"
    assert normalize_value("  Anna   Muster ", "PERSON") == "anna muster"


def test_policy_from_config_fills_defaults() -> None:
    pol = policy_from_config({"PERSON": "keep"}, "mask", 0.6, ["PERSON", "IBAN_CODE"])
    assert pol.action_for("PERSON") == "keep"
    assert pol.action_for("IBAN_CODE") == "mask"
    assert pol.action_for("UNKNOWN") == "mask"
    assert pol.score_threshold == 0.6
