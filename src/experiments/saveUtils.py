from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, Union

from nspyre import DataSink
from rpyc.utils.classic import obtain
from nspyre.data.save import save_json, save_pickle

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

    if file_format == "json":
        ext = ".json"
        save_fun = save_json
    elif file_format == "pickle":
        ext = ".pkl"
        save_fun = save_pickle
    else:
        raise ValueError(f"Unknown file format: {file_format}")
    
    # Remove any extension the user may have provided
    filename_clean = Path(filename).stem if filename else ""
    base_name = f"{expType}_{filename_clean}".strip()
    if not base_name:
        base_name = str(expType)

    with DataSink(datasetName) as dataSink:
        # Ensure we have at least one data point ready
        # For large datasets from long experiments, increase timeout substantially
        print(f"Waiting for data from DataSink '{datasetName}'... (this may take a while for large datasets)")
        dataSink.pop(600)  # 600-second timeout (10 min) for very large datasets

        data_obtained = None
        pickle_direct_ok = (file_format == "pickle")

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
            exp_path = base / date_folder / expType

            # If a non-directory file blocks the path, fail with a clear error
            if exp_path.exists() and not exp_path.is_dir():
                raise FileExistsError(
                    f"Cannot create directory {exp_path}: a non-directory file exists at that path."
                )

            # Create directories
            exp_path.mkdir(parents=True, exist_ok=True)

            file_path = exp_path / f"{base_name}{ext}"
            counter = 1
            while file_path.exists():
                file_path = exp_path / f"{base_name}_{counter}{ext}"
                counter += 1

            # Helpful debug prints
            print("Trying to save in flexSave...")
            print(str(file_path))

            # Save
            if file_format == "json":
                if data_obtained is None:
                    data_obtained = obtain(dataSink.data)
                save_fun(str(file_path), data_obtained)

            else:  # pickle
                if pickle_direct_ok:
                    try:
                        save_fun(str(file_path), dataSink.data)
                        continue
                    except Exception:
                        pickle_direct_ok = False

                if data_obtained is None:
                    data_obtained = obtain(dataSink.data)
                save_fun(str(file_path), data_obtained)
