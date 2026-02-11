import pickle
from pathlib import Path


def save_to_pickle_file(data, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle_file(file_path: Path):
    with open(file_path, "rb") as file:
        return pickle.load(file)