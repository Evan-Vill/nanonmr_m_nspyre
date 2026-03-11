from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, Union

from nspyre import DataSink
from rpyc.utils.classic import obtain
from nspyre.gui.widgets.save import save_json, save_pickle

def flexSave(
    datasetName: str,
    expType: str,
    filename: str,
    dirs: Union[str, Path, Iterable[Union[str, Path]]] = r"E:\Data",
    file_format: str = "json",
):
    """
    Save data from dataserv into one or more directories using a structure:

        <base_dir>\<YY_MM_DD>\<expType>\<expType>_<filename>.json

    - Uses pathlib to avoid slash mixing on Windows (incl. UNC paths).
    - Accepts dirs as a single path or a list/iterable of paths.
    - Avoids GUI-ellipsized paths (contains "...") by raising early.
    """

    if not dirs:
        raise ValueError("No directories specified for custom autosaver")

    # Normalize dirs into a list of strings/Paths
    if isinstance(dirs, (str, Path)):
        dirs_list = [dirs]
    else:
        dirs_list = list(dirs)

    now = datetime.now()
    date_folder = now.strftime("%y_%m_%d")

    with DataSink(datasetName) as dataSink:
        # Ensure we have at least one data point ready
        dataSink.pop(1)

        for save_dir in dirs_list:
            base = Path(save_dir)

            # Catch the exact GUI bug: passing ellipsized display text
            if "..." in str(base):
                raise ValueError(
                    f"Directory looks ellipsized/truncated: {base}\n"
                    "Store the full path separately in the GUI (e.g. widget property) "
                    "and pass that to flexSave."
                )

            # Build platform-correct paths (UNC-safe)
            date_path = base / date_folder
            exp_path = date_path / expType

            # If a non-directory file blocks the path, fail with a clear error
            if exp_path.exists() and not exp_path.is_dir():
                raise FileExistsError(
                    f"Cannot create directory {exp_path}: a non-directory file exists at that path."
                )

            # Create directories
            exp_path.mkdir(parents=True, exist_ok=True)

            # Build a unique filename
            base_name = f"{expType}_{filename}".strip()
            if not base_name:
                base_name = expType

            file_path = exp_path / f"{base_name}.json"
            counter = 1
            while file_path.exists():
                file_path = exp_path / f"{base_name}_{counter}.json"
                counter += 1

            # Helpful debug prints
            print("Trying to save in flexSave...")
            print(str(date_path), str(exp_path), str(file_path))

            # Save
            if file_format == "json":
                save_json(str(file_path), obtain(dataSink.data))
            elif file_format == "pickle":
                save_pickle(str(file_path), obtain(dataSink.data))
