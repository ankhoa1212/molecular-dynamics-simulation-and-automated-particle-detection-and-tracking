"""
Shared utilities for parsing LAMMPS dump files.
"""


def parse_lammps_dump(filename):
    """
    Generator that yields simulation frames from a LAMMPS dump file.

    Yields:
        dict: containing:
            - timestep (int)
            - num_atoms (int)
            - box_bounds (list of strings)
            - atoms (list of strings or generator)
    """
    with open(filename, "r", encoding="utf-8") as dump_file:
        while True:
            line = dump_file.readline()
            if not line:
                break

            if "ITEM: TIMESTEP" in line:
                try:
                    timestep = int(dump_file.readline().strip())
                except ValueError:
                    continue

                line = dump_file.readline()
                while line and "ITEM: NUMBER OF ATOMS" not in line:
                    line = dump_file.readline()
                if not line:
                    break
                num_atoms = int(dump_file.readline().strip())

                line = dump_file.readline()
                while line and "ITEM: BOX BOUNDS" not in line:
                    line = dump_file.readline()
                if not line:
                    break
                box_lines = [dump_file.readline() for _ in range(3)]

                line = dump_file.readline()
                while line and "ITEM: ATOMS" not in line:
                    line = dump_file.readline()
                if not line:
                    break

                atom_header = line.strip()
                atom_lines = [dump_file.readline() for _ in range(num_atoms)]

                yield {
                    "timestep": timestep,
                    "num_atoms": num_atoms,
                    "box_bounds": box_lines,
                    "atom_header": atom_header,
                    "atoms": atom_lines,
                }
