import hypothesis as hp

from proteinbee import motif

import pytest

from typing import Callable


type DrawType[T] = Callable[[hp.st.SearchStrategy[T]], T]


@hp.strategies.composite
def st_selector[T](draw: DrawType[T]) -> DrawType[T]:
    return draw(
        hp.strategies.from_regex(r"^[A-Z][0-9]{1,5}-[0-9]{1,5}$", fullmatch = True)
        .filter(
            lambda x: int(x[1 : x.find("-")]) <= int(x[x.find("-") + 1 :])
        )
    )


@hp.strategies.composite
def st_motif[T](draw: DrawType[T]) -> tuple[DrawType[T], DrawType[T]]:
    i = draw(hp.strategies.integers(0, 1000))
    sel = draw(st_selector().map(lambda x: motif.Selector.from_string(x)))
    lst = [i, sel]
    remaining_size = draw(hp.strategies.integers(2, 100))
    return (
        draw(
            hp.strategies
            .lists(hp.strategies.sampled_from(lst))
            .map(lambda sub_lst: sub_lst + lst)
            .flatmap(hp.strategies.permutations)
        ),
        draw(hp.strategies.sampled_from(["/", ",", "."]))
    )


@hp.given(st_selector())
def test_selector_init(sel: str) -> None:
    hp.assume(int(sel[1 : sel.find("-")]) <= int(sel[sel.find("-") + 1 :]))
    selector = motif.Selector.from_string(sel)
    assert selector.chain == sel[0]
    start, end = sel[1 :].split("-")
    assert selector.start == int(start)
    assert selector.end == int(end)


@pytest.mark.parametrize(
    "sel, length",
    [
        ("A12-24", 13),
        ("X9-10", 2),
        ("X10-10", 1),
    ],
)
def test_selector_init_special(sel: str, length: int) -> None:
    selector = motif.Selector.from_string(sel)
    assert len(selector) == length


@hp.given(st_motif())
def test_motif_init(motif_elements: tuple[list[str | motif.Selector], str]) -> None:
    m, delim = motif_elements
    hp.assume(len(m) != 0)
    print(m)
    m_str = delim.join(str(x) for x in m)
    mo = motif.Motif(m, delim)
    mo_str = motif.Motif.from_string(m_str, delim)
    assert mo == mo_str
    assert str(mo) == m_str
    assert str(mo_str) == m_str


@pytest.mark.parametrize(
    "s, expected",
    [
        (
            "A814-824/10/-1/25/B1-10",
            [
                motif.Selector.from_string("A814-824"),
                motif.Selector.from_string("B1-10"),
            ]
        ),
        (
            "1/A322-326/15/B531-539/12/-1/B551-562/10/B579-585/14",
            [
                motif.Selector.from_string("A322-326"),
                motif.Selector.from_string("B531-539"),
                motif.Selector.from_string("B551-562"),
                motif.Selector.from_string("B579-585"),
            ],
        ),
    ],
)
def test_motif_selector_iter(s: str, expected: list[motif.Selector]) -> None:
    m = motif.Motif.from_string(s)
    assert str(m) == s
    for sel_actual, sel_expected in zip(m.selector_iter(), expected):
        assert sel_actual == sel_expected


@pytest.mark.parametrize(
    "s, expected",
    [
        (
            "A814-824/10/-1/25/B1-10",
            [
                10,
                25,
            ],
        ),
        (
            "1/A322-326/-1/B531-539/12/-1/B551-562/10/B579-585/14",
            [
                1,
                12,
                10,
                14,
            ],
        ),
    ],
)
def test_motif_segment_iter(s: str, expected: list[motif.Selector]) -> None:
    m = motif.Motif.from_string(s)
    assert str(m) == s
    for seg_actual, seg_expected in zip(m.segment_iter(), expected):
        assert seg_actual == seg_expected


@pytest.mark.parametrize(
    "motif, expected",
    [
        (
            motif.Motif.from_string("25/A814-824/25"),
            motif.Motif.from_string("25/A26-36/25"),
        ),
        (
            motif.Motif.from_string("25/A1-10/10/B1-10/25"),
            motif.Motif.from_string("25/A26-35/10/A46-55/25"),
        ),
        (
            motif.Motif.from_string(
                "1/G1232-1232/1/G1234-1235/1/G1237-1239/1/G1241-1244/1",
            ),
            motif.Motif.from_string("1/A2-2/1/A4-5/1/A7-9/1/A11-14/1"),
        ),
        (
            motif.Motif.from_string("10/A322-326/5/B531-539/5/B551-562/5/B579-585/10"),
            motif.Motif.from_string("10/A11-15/5/A21-29/5/A35-46/5/A52-58/10"),
        ),
    ]
)
def test_get_motif_wrt_designed_structure(motif: motif.Motif, expected: motif.Motif) -> None:
    assert motif.get_motif_wrt_designed_structure() == expected


@pytest.mark.parametrize(
    "motif, expected",
    [
        (
            motif.Motif.from_string("10/A322-326/5/0/B531-539/5/B551-562"),
            [
                motif.Motif.from_string("10/A11-15/5"),
                motif.Motif.from_string("B1-9/5/B15-26"),
            ],
        ),
        (
            motif.Motif.from_string("16/A38-42/48/A291-293/10/B18-52/40/0/H1-111"),
            [
                motif.Motif.from_string("16/A17-21/48/A70-72/10/A83-117/40"),
                motif.Motif.from_string("B1-111"),
            ],
        ),
    ],
)
def test_get_motif_wrt_designed_structure_multi_chain(motif: motif.Motif, expected: list[str]) -> None:
    assert list(motif.get_motif_wrt_designed_structure_multi_chain()) == expected


@pytest.mark.parametrize(
    "motif, expected",
    [
        (
            motif.Motif.from_string("25/A814-824/25"),
            {
                motif.Selector.from_string("A814-824"): 1,
            },
        ),
        (
            motif.Motif.from_string("25/C1146-1165/10/C1146-1165/0/A814-824/10/A814-824/25"),
            {
                motif.Selector.from_string("A814-824"): 2,
                motif.Selector.from_string("C1146-1165"): 2,
            },
        ),
        (
            motif.Motif.from_string("10/A322-326/5/B531-539/5/B551-562/5/B579-585/10"),
            {
                motif.Selector.from_string("A322-326"): 1,
                motif.Selector.from_string("B531-539"): 1,
                motif.Selector.from_string("B551-562"): 1,
                motif.Selector.from_string("B579-585"): 1,
            },
        ),
    ],
)
def test_get_selector_counts(motif: motif.Motif, expected: dict[str, int]) -> None:
    assert motif.get_selector_counts() == expected
