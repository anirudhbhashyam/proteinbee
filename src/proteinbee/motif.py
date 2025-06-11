from dataclasses import dataclass

import re

from typing import (
    ClassVar,
    Iterator,
    Pattern,
    Self,
)


@dataclass(slots = True, frozen = True)
class Selector:
    chain: str
    start: int
    end: int
    _selector_pattern_re: ClassVar[Pattern[str]] = re.compile(
        r"^[A-Z][0-9]{1,5}-?[0-9]{1,5}?$",
    )

    @classmethod
    def from_string(cls, s: str) -> Self:
        cls._pattern_check(s)
        range_ = s[1: ].split("-")
        cls._range_check(range_)
        return cls(s[0], int(range_[0]), int(range_[1]))
    
    @classmethod
    def check_string(cls, s: str) -> bool:
        return cls._pattern_check(s) and cls._range_check(s[1 :].split())

    @classmethod
    def _pattern_check(cls, s: str) -> None:
        if not re.fullmatch(cls._selector_pattern_re, s):
            raise ValueError(f"Invalid selector string: {s}. The selector string should be of the form: A786-790.")

    @staticmethod
    def _range_check(range_: tuple[str, str]) -> None:
        r1, r2 = range_
        if not int(r1) <= int(r2):
            raise ValueError(
                f"Invalid range: {range_} in selector."
                f"The number left of '-' should be smaller than the number right of '-'."
            )
        
    def __len__(self) -> int:
        return self.end - self.start + 1
        
    def __str__(self) -> str:
        return f"{self.chain}{self.start}-{self.end}"


@dataclass
class Motif:
    components: list[Selector | int]
    delim: str 

    @classmethod
    def from_string(cls, s: str, delim: str = "/") -> Self:
        components = []
        for sel in s.split(delim):
            if sel.isnumeric():
                components.append(int(sel))
            else:
                components.append(
                    Selector.from_string(sel),
                )
        return cls(components, delim)
    
    def selector_iter(self) -> Iterator[str]:
        yield from (
            x
            for x in self.components
            if isinstance(x, Selector)
        )

    def segment_iter(self) -> Iterator[str]:
        yield from (
            x
            for x in self.components
            if isinstance(x, int) and x != -1
        )

    def get_motif_wrt_designed_structure(self, chain_id: str = "A") -> Self:
        m_str = ""
        curr_pos = 1
        for comp in self.components:
            if isinstance(comp, int):
                    m_str += f"{comp}/"
                    curr_pos += comp
            if isinstance(comp, Selector):
                design_chain = chain_id
                m_str += f"{design_chain}{curr_pos}-{curr_pos + len(comp) - 1}/"
                curr_pos += len(comp)
        return type(self).from_string(m_str[: -1])

    def get_motif_wrt_designed_structure_multi_chain(self) -> Iterator[Self]:
        chains = list(reversed("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        for chain_components in self.split_by_chain():
            yield chain_components.get_motif_wrt_designed_structure(chains.pop())

    def get_selector_counts(self) -> dict[Selector, int]:
        counter = collections.defaultdict(int)
        for selector in self.selector_iter():
            counter[selector] += 1
        return dict(counter)
    
    def split_by_chain(self) -> Iterator[list[int | Selector]]:
        motif_components_batch = []
        for i, comp in enumerate(self.components):
            if comp == 0 or i == len(self.components) - 1:
                if comp != 0:
                    motif_components_batch.append(comp)
                yield type(self)(motif_components_batch)
                motif_components_batch = []
            else:
                motif_components_batch.append(comp)
        
    def __str__(self) -> str:
        return self.delim.join(str(x) for x in self.components)