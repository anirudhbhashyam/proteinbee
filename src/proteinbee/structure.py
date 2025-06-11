from dataclasses import dataclass

from biotite.database import rcsb
from biotite.structure import (
    array as AtomArrayConstructor,
    AtomArray,
    residue_iter,
)
import biotite.structure.io as structureio

from io import BytesIO, StringIO

import numpy as np

import os

from proteinbee.motif import Selector
from proteinbee.motif import Motif

from tempfile import gettempdir

from typing import (
    Iterable,
    Iterator,
    Self,
)


three_res_to_one_res = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLU": "E",
    "GLN": "Q",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


type PathType = str | os.PathLike[str] 


@dataclass
class Structure:
    _atom_array: AtomArray

    @classmethod
    def from_file(cls, filepath: PathType) -> Self:
        return cls(
            structureio.load_structure(filepath)
        )

    @classmethod
    def from_pdb_code(cls, pdb_code: str, structure_format: str = "pdbx") -> Self:
        filepath = rcsb.fetch(pdb_code, structure_format, gettempdir())
        return cls.from_file(filepath)

    @classmethod
    def from_blob(cls, blob: bytes, structure_format: str) -> Self:
        # Need to figure out why the atom array here becomes a stack in the main object.
        match structure_format.casefold():
            case "pdb":
                stream = StringIO(blob.decode("utf-8"))
                file = structureio.pdb.PDBFile.read(stream)
                return cls(structureio.pdb.get_structure(file))
            case "cif" | "pdbx":
                stream = StringIO(blob.decode("utf-8"))
                file = structureio.pdbx.CIFFile.read(stream)
                return cls(structureio.pdbx.get_structure(file))
            case "bcif":
                stream = BytesIO(blob)
                file = structureio.pdbx.BinaryCIFFile.read(stream)
                return cls(structureio.pdbx.get_structure(file))
            case _:
                raise ValueError(f"Unsupported structure format: {structure_format}")
    
    def get_number_of_atoms(self) -> int:
        return len(self._atom_array)

    @property
    def atom_array(self) -> AtomArray:
        return self._atom_array
    
    def save_structure(self, filepath: PathType) -> None:
        structureio.save_structure(filepath, self._atom_array)

    def chain_iter(self) -> Iterator[str]:
        yield from set(self._atom_array.chain_id)

    def atom_iter(self) -> Iterator[str]:
        yield from self._atom_array

    def aa_iter(self) -> Iterator[str]:
        for res in residue_iter(self._atom_array):
            aa = res.res_name[0].upper()
            yield three_res_to_one_res[aa]

    def residue_id_iter(self) -> Iterator[str]:
        yield from self._atom_array.res_id

    def select_using_chains(self, chains: Iterable[str]) -> Self:
        mask = self._create_atom_array_mask("chain_id", chains)
        return type(self)(
            self._atom_array[mask]
        )

    def select_using_residue_range(self, start: int, end: int) -> Self:
        if start > end:
            raise ValueError("'start' cannot be greater than 'end'.")
        return type(self)(
            self._atom_array[
                np.logical_and(
                    start <= self._atom_array.res_id,
                    self._atom_array.res_id <= end,
                )
            ]
        )

    def select_using_atom_types(self, atom_types: Iterable[str]) -> Self:
        mask = self._create_atom_array_mask("atom_name", atom_types)
        return type(self)(
            self._atom_array[mask]
        )

    def select_using_selector(self, sel: Selector) -> Self:
        return (
            self
            .select_using_chains((sel.chain, ))
            .select_using_residue_range(sel.start, sel.end)
        )

    def select_using_motif(self, motif: Motif) -> Self:
        atoms = []
        for selector in motif.selector_iter():
            atoms.extend(
                self
                .select_using_selector(selector)
                .atom_iter()
            )
        return type(self)(
            AtomArrayConstructor(atoms),
        )
    
    def get_atom_data(self) -> Iterator[dict[str, str | float]]:
        for atom in self.atom_iter():
            yield {
                "res_id": atom._annot["res_id"],
                "res_name": atom._annot["res_name"],
                "atom": atom._annot["atom_name"],
                "x": atom.coord[0],
                "y": atom.coord[1],
                "z": atom.coord[2],
            }
    
    def is_empty(self) -> bool:
        return len(self._atom_array) == 0

    def get_submodel(self, num: int) -> Self:
        if self.model.atom_array.ndim <= 2:
            raise ValueError("The structure does not contain multiple models.")
        if num <= self.atom_array.shape[0] or self.atom_array.shape[0] < num:
            raise ValueError(
                "The specified model does not exist. Check the number of models that have been read by accessing the atom_array property."
            )
        return type(self)(
            self.atom_array[num]
        )
    
    def align(self, other: Self) -> Self:
        if not np.array_equal(self._atom_array, other._atom_array):
            raise ValueError("The structures do not have the same atoms.")
        raise NotImplementedError("Structure alignment not yet implemented.")

    def _create_atom_array_mask(self, attr: str, items: Iterable[str]) -> np.array:
        mask = np.array([False] * len(self.atom_array), dtype = bool)
        for item in items:
            mask |= (getattr(self.atom_array, attr) == item)
        return mask