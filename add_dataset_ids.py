import pandas as pd

from constants import DATA_FILE


def add_ids_in_place():
    dataset = pd.read_csv(DATA_FILE)

    if "id" in dataset.columns:
        print(f"'{DATA_FILE}' already contains an 'id' column. No changes made.")
        return

    dataset.insert(0, "id", range(len(dataset)))
    dataset.to_csv(DATA_FILE, index=False)
    print(f"Added persistent ids to {len(dataset)} rows in '{DATA_FILE}'.")


if __name__ == "__main__":
    add_ids_in_place()
