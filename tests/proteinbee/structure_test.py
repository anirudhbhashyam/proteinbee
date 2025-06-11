import biotite.structure.io as structureio

from proteinbee import structure
from proteinbee.motif import Selector, Motif

from pathlib import Path

import pytest

from io import BytesIO, StringIO


@pytest.mark.parametrize(
    "pdb_code, structure_format",
    [
        (
            "7u0a",
            "pdbx",
        ),
        (
            "7t6x",
            "cif",
        ),
        (
            "7nab",
            "pdb",
        ),
        (
            "8fsj",
            "pdbx",
        ),
    ],
)
def test_structure_from_pdb_code(pdb_code: str, structure_format: str) -> None:
    struc = structure.Structure.from_pdb_code(
        pdb_code,
        structure_format = structure_format,
    )
    assert isinstance(struc._atom_array, structure.AtomArray)
    assert struc.get_number_of_atoms() > 0


@pytest.mark.parametrize(
    "pdb_code, structure_format",
    [
        (
            "4fqi",
            "pdb",
        ),
        (
            "6mb3",
            "cif",
        ),
    ],
)
def test_structure_from_blob(pdb_code: str, structure_format: str) -> None:
    struc = structure.Structure.from_pdb_code(
        pdb_code,
        structure_format = structure_format,
    )
    match structure_format:
        case "pdb":
            stream = StringIO()
            file = structureio.pdb.PDBFile()
            structureio.pdb.set_structure(file, struc.atom_array)
        case "cif" | "pdbx":
            stream = StringIO()
            file = structureio.pdbx.CIFFile()
            structureio.pdbx.set_structure(file, struc.atom_array)
        case "bcif":
            stream = BytesIO()
            file = structureio.pdbx.BinaryCIFFile()
            structureio.pdbx.set_structure(file, struc.atom_array)
    file.write(stream)
    blob = stream.getvalue().encode("utf-8")
    struc_blob = structure.Structure.from_blob(blob, structure_format = structure_format)
    assert struc_blob.atom_array[0] == struc.atom_array


@pytest.mark.parametrize(
    "pdb_code, expected",
    [
        (
            "7u0a",
            set("AHL"),
        ),
        (
            "7t6x",
            set("ABCDEUHLFGIJKMNOP"),
        ),
        (
            "7nab",
            set("ABCDHL"),
        ),
        (
            "8fsj",
            set("ABCDEFGHIJKLMN"),
        ),
    ],
)
def test_chain_iter(pdb_code: str, expected: set[str]) -> None:
    struc = structure.Structure.from_pdb_code(pdb_code)
    assert set(struc.chain_iter()) == expected


@pytest.mark.parametrize(
    "pdb_code, chain",
    [
        (
            "7u0a",
            "L",
        ),
        (
            "7t6x",
            "U",
        ),
        (
            "7nab",
            "H",
        ),
        (
            "8fsj",
            "A",
        ),
        (
            "6mb3",
            "A",
        )
    ],
)
def test_select_using_chains(pdb_code: str, chain: str) -> None:
    struc = structure.Structure.from_pdb_code(pdb_code)
    sub_struc = struc.select_using_chains(chain)
    set(sub_struc.chain_iter()) == set(chain)

@pytest.mark.parametrize(
    "pdb_code, chain, res_range",
    [
        (
            "7u0a",
            "A",
            (814, 824)
        ),
        (
            "7t6x",
            "H",
            (5, 26),
        ),
        (
            "7nab",
            "C",
            (1146, 1157),
        ),
        (
            "8fsj",
            "E",
            (565, 619),
        ),
    ],
)
def test_select_using_residue_number(
    pdb_code: str,
    chain: str,
    res_range: tuple[int, int],
) -> None:
    struc = structure.Structure.from_pdb_code(pdb_code)
    sub_struc = (
        struc
        .select_using_chains((chain, ))
        .select_using_residue_range(*res_range)
    )
    l, h = res_range
    assert l == min(sub_struc.residue_id_iter())
    assert h == max(sub_struc.residue_id_iter())


@pytest.mark.parametrize(
    "pdb_code, chain, res_range, expected",
    [
        (
            "7u0a",
            "A",
            (814, 824),
            "KRSFIEDLLFN",
        ),
        (
            "7t6x",
            "H",
            (5, 26),
            "LEQSGPEVKKPGDSLRISCKMSG",
        ),
        (
            "7nab",
            "C",
            (1146, 1157),
            "DSFKEELDKYFK",
        ),
        (
            "8fsj",
            "E",
            (565, 619),
            "GGPPCNIGGVGNNTLTCPTDCFRKHPEATYTKCGSGPWLTPRCLVDYPYRLWHYP",
        ),
    ],
)
def test_aa_iter(
    pdb_code: str,
    chain: str,
    res_range: tuple[int, int],
    expected: str,
) -> None:
    struc = structure.Structure.from_pdb_code(pdb_code)
    sub_struc = (
        struc
        .select_using_chains((chain, ))
        .select_using_residue_range(*res_range)
    )
    assert "".join(sub_struc.aa_iter()) == expected


@pytest.mark.parametrize(
    "pdb_code, atom_types",
    [
        (
            "7u0a",
            {
                "CA",
                "N",
                "O",
            },
        ),
        (
            "7t6x",
            {
                "CA",
                "C",
            }
        )
    ],
)
def test_select_using_atom_types(
    pdb_code: str,
    atom_types: list[str],
) -> None:
    struc = structure.Structure.from_pdb_code(pdb_code)
    sub_struc = (
        struc
        .select_using_atom_types(atom_types)
    )
    assert set(sub_struc._atom_array.atom_name) == atom_types


@pytest.mark.parametrize(
    "pdb_code, sel, expected",
    [
        (
            "7u0a",
            Selector.from_string("A814-824"),
            "KRSFIEDLLFN",
        ),
        (
            "7t6x",
            Selector.from_string("H5-26"),
            "LEQSGPEVKKPGDSLRISCKMSG",
        ),
        (
            "7nab",
            Selector.from_string("C1146-1157"),
            "DSFKEELDKYFK",
        ),
        (
            "8fsj",
            Selector.from_string("E565-619"),
            "GGPPCNIGGVGNNTLTCPTDCFRKHPEATYTKCGSGPWLTPRCLVDYPYRLWHYP",
        ),
    ]
)
def test_select_using_selector(
    pdb_code: str,
    sel: Selector,
    expected: str,
) -> None:
    struc = structure.Structure.from_pdb_code(pdb_code)
    sub_struc = struc.select_using_selector(sel)
    assert "".join(sub_struc.aa_iter()) == expected


@pytest.mark.parametrize(
    "pdb_code, motif, expected_seq, expected_n_atoms",
    [
        (
            "4fqi",
            Motif.from_string("A38-42/A291-293/B18-21/B36-52"),
            "HAQDISMPVDGWADKESTQKAIDGVTNKV",
            219,
        ),
        (
            "7u0a",
            Motif.from_string("A814-824"),
            "KRSFIEDLLFN",
            97,
        )
    ],
)
def test_select_using_motif(
    pdb_code: str,
    motif: Motif,
    expected_seq: str,
    expected_n_atoms: int,
) -> None:
    struc = structure.Structure.from_pdb_code(pdb_code)
    sub_struc = struc.select_using_motif(motif)
    assert "".join(sub_struc.aa_iter()) == expected_seq
    assert sub_struc.atom_array.shape[0] == expected_n_atoms


@pytest.mark.parametrize(
    "pdb_code",
    [
        "7U0A",
        "6MRR",
    ]
)
def test_save_structure(pdb_code: str, tmp_path: Path) -> None:
    struc = structure.Structure.from_pdb_code(pdb_code)
    filepath = tmp_path / f"{pdb_code}.cif"
    struc.save_structure(filepath)
    assert filepath.exists()
