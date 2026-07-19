"""
Parse a Motive/OptiTrack export (e.g. data/Static_Tests/10mm_0.3.csv) into a
pandas DataFrame:

    - row index  -> "Time (Seconds)" column
    - columns    -> 3-level MultiIndex: (Name, Component, Axis)
                    e.g. ('SpiritNewest', 'Rotation', 'X')

The file layout (0-indexed lines) is:
    0: metadata           ("Format Version,1.23,Take Name,...")
    1: blank
    2: ,Type,Rigid Body,Rigid Body,...
    3: ,Name,SpiritNewest,SpiritNewest,...
    4: ,ID,506381AA...,506381AA...
    5: ,,Rotation,Rotation,Position,Position,...
    6: Frame,Time (Seconds),X,Y,Z,W,X,Y,Z,...
    7+: numeric data

Rows 5 and 6 ("the two rows immediately below the ID row") are exactly the
Component (Rotation/Position) and Axis (X/Y/Z/W) info you want attached to
each column. The trick is to let pandas parse them as header rows (via the
`header=[...]` argument) rather than data rows -- that keeps every data
column purely numeric (float64) instead of getting upcast to `object` the
moment a string like "Rotation" lands in it.
"""

import pandas as pd
from pathlib import Path

def load_motive_csv(path: str) -> pd.DataFrame:
    # skiprows=1 skips the ragged metadata line (line 0), which has a
    # different field count than the rest of the file and would otherwise
    # break the tokenizer.
    #
    # header=[1, 3, 4] (indices *after* skiprows=1 is applied, and after
    # pandas' default skip_blank_lines=True silently drops the now-empty
    # line 0) picks out:
    #   1 -> Name row       ('SpiritNewest', 'SpiritNewest', ...)
    #   3 -> Rotation/Position row
    #   4 -> X/Y/Z/W row
    # and pandas automatically skips the ID row (index 2) since it's not
    # in the header list, combining the rest into a 3-level MultiIndex.
    df = pd.read_csv(path, skiprows=1, header=[1, 3, 4])

    # Column 0 is Frame, column 1 is Time (Seconds) -- both came through
    # with junky "Unnamed: ..." labels on their unused header levels.
    time_col = df.columns[1]

    df = df.set_index(time_col)
    df.index.name = "Time (Seconds)"

    # Drop the now-redundant Frame column.
    df = df.drop(columns=[df.columns[0]])

    # Clean up the MultiIndex: give the levels sensible names and drop the
    # "Unnamed: ..." placeholders that pandas invented for the blank
    # header cells (there shouldn't be any left after removing Frame/Time,
    # but this is a harmless safety net if e.g. a column has no ID/name).
    df.columns = pd.MultiIndex.from_tuples(
        [tuple(lvl if not str(lvl).startswith("Unnamed") else "" for lvl in col)
         for col in df.columns],
        names=["Name", "Component", "Axis"],
    )

    # Every remaining column is now purely numeric.
    df = df.apply(pd.to_numeric, errors="coerce")

    # Drop any column whose Name level mentions "SpiritNewest" or
    # "Unlabeled" (case-insensitive), e.g. 'SpiritNewest', 'SpiritNewest:Marker1',
    # 'Unlabeled 1138', etc.
    name_level = df.columns.get_level_values("Name").astype(str)
    drop_mask = name_level.str.contains("spiritnewest|unlabeled", case=False, regex=True)
    df = df.loc[:, ~drop_mask]

    return df


if __name__ == "__main__":
    output_dir = "processed_data\\"
    input_dir = "raw_data\\"

    dir_path = Path(input_dir)

    # rglob("*") recursively finds everything inside the directory
    for file_path in dir_path.rglob("*"):
        if file_path.is_file():
            print(f"Processing nested file: {file_path}")
            print(f"{output_dir}{file_path.parent.name}\\{file_path.name}")
            df = load_motive_csv(file_path)
            df.to_csv(output_dir + file_path.parent.name + "\\" + file_path.name, index=True)

    # Note: since df.columns is a MultiIndex (Name, Component, Axis), the
    # output CSV will have 3 header rows (one per level) above the data,
    # e.g.:
    #   Name,FrontBody,FrontBody,...
    #   Component,Rotation,Rotation,...
    #   Axis,X,Y,...
    #   Time (Seconds),,
    #   0.0,0.051245,...
    #
    # If you'd rather flatten those 3 levels into a single header row
    # (e.g. "FrontBody_Rotation_X") for easier re-reading with a plain
    # pd.read_csv(), do this before calling to_csv():
    #
    #   flat = df.copy()
    #   flat.columns = ["_".join(col).strip("_") for col in flat.columns]
    #   flat.to_csv("cleaned_10mm_0.3_flat.csv")   # index=True by default