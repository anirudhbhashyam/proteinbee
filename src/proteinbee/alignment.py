from dataclasses import (
    dataclass,
    field,
)

from biotite.structure import (
    rmsd,
    superimpose,
)

import numpy as np

from proteinbee.motif import Motif

from proteinbee.structure import Structure

from typing import Self


@dataclass(slots = True, frozen = True)
class StructureAlignment:
    structure: Structure
    motif: Motif
    _motif_structure: Structure = field(
        init = False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "_motif_structure", self.structure.select_using_motif(self.motif))

    def get_motif_structure(self) -> Structure:
        return self._motif_structure

    def get_motif_deviation(self, reference: Self) -> float:
        reference_motif_structure = reference.get_motif_structure()
        assert len(self._motif_structure.atom_array) == len(reference_motif_structure.atom_array)
        reference_coords = reference_motif_structure.atom_array.coord
        mobile_coords = self._motif_structure.atom_array.coord
        deviation = self.root_mean_square_deviation(
            reference_coords,
            mobile_coords,
        )
        return deviation

    @staticmethod
    def root_mean_square_deviation(reference_coords: np.array, mobile_coords: np.array) -> float:
        assert reference_coords.shape[0] == mobile_coords.shape[0]
        aligned_coords, _ = superimpose(
            reference_coords,
            mobile_coords,
        )
        return rmsd(
            reference_coords,
            aligned_coords,
        )