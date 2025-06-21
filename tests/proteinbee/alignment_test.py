import numpy as np

from proteinbee import alignment

from proteinbee.motif import Motif

from proteinbee.structure import Structure

import pytest


@pytest.mark.parametrize(
    "struc, motif",
    [
        (
            Structure.from_pdb_code("7u0a"),
            Motif.from_string("A814-824"),
        ),
        (
            Structure.from_pdb_code("7t6x"),
            Motif.from_string("E192-256"),
        ),
    ],
)
def test_structure_alignment_creation(struc: Structure, motif: Motif) -> None:
    struc_al = alignment.StructureAlignment(struc, motif)
    assert isinstance(struc_al.structure, Structure)
    assert isinstance(struc_al.motif, Motif)


@pytest.mark.parametrize(
    "ref_coords, mobile_coords, expected",
    [
        (
            np.zeros((100, 3), dtype = np.float32),
            np.zeros((100, 3), dtype = np.float32),
            0.0,
        ),
        (
            np.ones((100, 3), dtype = np.float32),
            np.ones((100, 3), dtype = np.float32),
            0.0,
        ),
        (
            np.full((100, 3), fill_value = -2.0, dtype = np.float32),
            np.full((100, 3), fill_value = -2.0, dtype = np.float32),
            0.0,
        ),
    ],
)
def test_structure_alignment_root_mean_square_deviation(
    ref_coords: np.array,
    mobile_coords: np.array,
    expected: float
) -> None:
    rmsd = alignment.StructureAlignment.root_mean_square_deviation(ref_coords, mobile_coords)
    assert pytest.approx(rmsd) == expected



@pytest.mark.parametrize(
    "ref_struc, mobile_struc, ref_motif, mobile_motif, expected",
    [
        (
            Structure.from_pdb_code("7nab"),
            Structure.from_pdb_code("7raq"),
            Motif.from_string("C1152-1165"),
            Motif.from_string("P1152-1165"),
            0.9224294,
        ),
    ],
)
def test_structure_alignment_root_mean_square_deviation_using_pdbs(
    ref_struc: Structure,
    mobile_struc: Structure,
    ref_motif: Motif,
    mobile_motif: Motif,
    expected: float,
) -> None:
    ref_al = alignment.StructureAlignment(ref_struc, ref_motif)
    mobile_al = alignment.StructureAlignment(mobile_struc, mobile_motif)
    rmsd = mobile_al.get_motif_deviation(ref_al)
    assert pytest.approx(rmsd) == expected