import pytest

from ethrag.graph import CitationGraph, mentioned_numbers
from ethrag.ingest import Chunk


def test_mentioned_numbers_formats():
    assert mentioned_numbers("EIP-1559 and eip 4844, ERC20, erc-721.") == [1559, 4844, 20, 721]
    assert mentioned_numbers("nothing here") == []


@pytest.fixture(scope="module")
def graph(mini_chunks):
    return CitationGraph(mini_chunks)


def test_requires_edges_are_directed(graph):
    assert graph.requires["eip-1559"] == {"eip-2718", "eip-2930"}
    assert graph.required_by["eip-2718"] == {"eip-1559", "eip-2930", "eip-4844"}
    assert graph.requires["eip-2718"] == set()


def test_related_walks_both_directions_with_depth(graph):
    one = dict(graph.related("eip-1559", depth=1))
    assert one == {"eip-2718": 1, "eip-2930": 1, "eip-4844": 1}
    two = dict(graph.related("eip-1559", depth=2))
    assert two["eip-2929"] == 2 and "eip-1559" not in two
    assert graph.related("eip-1559", depth=1, kinds=("requires",)) == [("eip-2718", 1), ("eip-2930", 1)]
    assert graph.related("nope") == []


def test_related_is_sorted_by_distance_then_id(graph):
    rows = graph.related("eip-4844", depth=2)
    assert rows == sorted(rows, key=lambda kv: (kv[1], kv[0]))


def test_mentions_come_from_inline_text(graph):
    assert "eip-2718" in graph.mentions["eip-2930"] and "eip-1559" in graph.mentions["eip-4844"]
    assert "eip-4844" in graph.mentioned_by["eip-1559"]
    assert graph.neighbours("eip-4844", ("mentions",)) >= {"eip-1559"}


def test_unknown_edge_kind_raises(graph):
    with pytest.raises(ValueError, match="unknown edge kind"):
        graph.neighbours("eip-1559", ("bogus",))


def test_resolve_and_by_number(graph):
    assert graph.resolve(20) == "erc-20" and graph.resolve(1559) == "eip-1559"
    assert graph.resolve(123456) is None


def test_eip_copy_wins_over_erc_copy_for_same_number():
    chunks = [Chunk("erc-9#000", "erc-9", "t", "s", "x", number=9),
              Chunk("eip-9#000", "eip-9", "t", "s", "x", number=9)]
    assert CitationGraph(chunks).resolve(9) == "eip-9"


def test_edges_and_stats_are_deterministic(graph):
    e = graph.edges()
    assert ("eip-1559", "eip-2718", "requires") in e and e == graph.edges()
    s = graph.stats()
    assert s["docs"] == 7 and s["requires_edges"] == 7 and s["isolated"] >= 1


def test_requires_to_missing_doc_is_ignored():
    chunks = [Chunk("eip-5#000", "eip-5", "t", "s", "x", requires=(999,), number=5)]
    g = CitationGraph(chunks)
    assert g.requires["eip-5"] == set() and g.stats()["isolated"] == 1


def test_self_reference_is_not_an_edge():
    chunks = [Chunk("eip-5#000", "eip-5", "t", "s", "EIP-5 mentions EIP-5 itself", number=5,
                    requires=(5,))]
    g = CitationGraph(chunks)
    assert g.requires["eip-5"] == set() and not g.mentions["eip-5"]
