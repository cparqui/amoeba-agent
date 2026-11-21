from amoeba.agent import RelationOp, Store


def ops(*items: tuple[str, str, str, str]) -> list[RelationOp]:
    return [RelationOp(op=op, s=s, p=p, o=o) for op, s, p, o in items]


def test_self_is_seeded() -> None:
    store = Store()
    assert store.resolve("self") == "self"
    assert store.resolve("me") == "self"


def test_resolution_is_case_and_punctuation_insensitive() -> None:
    store = Store()
    store.upsert_entity("alpha0", ["Alpha0"])
    assert store.resolve("ALPHA-0") == "alpha0"
    assert store.resolve("alpha 0!") == "alpha0"
    assert store.resolve("beta") is None


def test_assert_creates_then_reinforces() -> None:
    store = Store()
    assert store.apply(ops(("ASSERT", "alpha0", "is_a", "cell")), step=1) == 1
    store.apply(ops(("ASSERT", "Alpha0", "is a", "Cell")), step=2)
    [rel] = store.about("alpha0")
    assert (rel.s, rel.p, rel.o) == ("alpha0", "is_a", "cell")
    assert rel.evidence == 2 and rel.weight > 1.0 and rel.confidence > 0.5
    assert rel.first_seen == 1 and rel.last_seen == 2
    assert store.resolve(rel.id) == rel.id  # relation ids are entities (reifiable)


def test_weaken_until_deleted_and_retract() -> None:
    store = Store()
    store.apply(ops(("ASSERT", "a", "near", "b"), ("ASSERT", "a", "near", "c")), step=1)
    for step in range(2, 6):
        store.apply(ops(("WEAKEN", "a", "near", "b")), step=step)
    assert [r.o for r in store.about("a")] == ["c"]
    store.apply(ops(("RETRACT", "a", "near", "c")), step=6)
    assert store.about("a") == []
    assert store.apply(ops(("RETRACT", "a", "near", "zzz")), step=7) == 0


def test_neighborhood_hops_literals_and_reified_relations() -> None:
    store = Store()
    store.apply(ops(("ASSERT", "a", "near", "b"), ("ASSERT", "b", "near", "c"), ("ASSERT", "c", "near", "d")), step=1)
    store.apply([RelationOp(op="ASSERT", s="a", p="energy", o="b", o_is_literal=True)], step=1)
    one = {(r.s, r.o) for r in store.neighborhood(["a"], hops=1)}
    two = {(r.s, r.o) for r in store.neighborhood(["a"], hops=2)}
    assert one == {("a", "b")}  # the literal "b" is not the entity b
    assert ("a", "b") in two and ("b", "c") in two and ("c", "d") not in two
    a_near_b = next(r for r in store.about("a") if r.o == "b" and not r.o_is_literal)
    store.apply(ops(("ASSERT", a_near_b.id, "holds_when", "dark")), step=2)
    assert any(r.s == a_near_b.id for r in store.neighborhood(["a"], hops=1))


def test_stats_and_file_persistence(tmp_path) -> None:
    path = tmp_path / "cell.db"
    store = Store(path)
    store.apply(ops(("ASSERT", "a", "near", "b")), step=1)
    stats = store.stats()
    assert stats["relations"] == 1 and stats["entities"] == 3 and stats["db_bytes"] > 0
    assert Store(path).stats()["relations"] == 1
