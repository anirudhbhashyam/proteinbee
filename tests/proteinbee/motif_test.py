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
