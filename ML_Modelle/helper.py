from typing import Union

import numpy as np
import pandas as pd


def _validate_race_split_inputs(
    features: pd.DataFrame,
    target: Union[pd.Series, pd.DataFrame],
) -> None:
    required_columns = {"season", "round"}
    missing_columns = required_columns - set(features.columns)
    if missing_columns:
        raise ValueError(
            "features muss folgende Spalten enthalten: "
            + ", ".join(sorted(missing_columns))
        )

    if len(features) != len(target):
        raise ValueError("features und target müssen gleich lang sein.")


def _get_n_test_races(test_size: Union[int, float], n_races: int) -> int:
    if n_races < 2:
        raise ValueError(
            "Für einen Train/Test-Split werden mindestens zwei Rennen benötigt."
        )

    if isinstance(test_size, float):
        if not 0 < test_size < 1:
            raise ValueError(
                "Wenn test_size ein float ist, muss der Wert zwischen 0 und 1 liegen."
            )
        return int(np.ceil(n_races * test_size))

    if isinstance(test_size, int):
        if not 1 <= test_size < n_races:
            raise ValueError(
                "Wenn test_size ein int ist, muss der Wert mindestens 1 und kleiner "
                "als die Anzahl Rennen sein."
            )
        return test_size

    raise TypeError("test_size muss ein int oder float sein.")


def _split_by_races(
    features: pd.DataFrame,
    target: Union[pd.Series, pd.DataFrame],
    train_races: pd.DataFrame,
    test_races: pd.DataFrame,
):
    train_keys = set(map(tuple, train_races[["season", "round"]].to_numpy()))
    race_keys = list(map(tuple, features[["season", "round"]].to_numpy()))
    train_mask = pd.Series([key in train_keys for key in race_keys])

    X_train = features.loc[train_mask].reset_index(drop=True)
    X_test = features.loc[~train_mask].reset_index(drop=True)
    y_train = target.loc[train_mask].reset_index(drop=True)
    y_test = target.loc[~train_mask].reset_index(drop=True)

    return X_train, X_test, y_train, y_test, train_races, test_races


def chronological_race_split(
    features: pd.DataFrame,
    target: Union[pd.Series, pd.DataFrame],
    test_size: Union[int, float] = 0.2,
):
    """
    Teilt Features und Target nach chronologischer Rennreihenfolge auf.

    Die ersten Rennen nach `season` und `round` werden für das Training
    verwendet. Die restlichen Rennen werden für das Testing verwendet.
    `test_size` funktioniert ähnlich wie bei sklearn:
    - float zwischen 0 und 1: Anteil der Rennen für das Testing
    - int >= 1: Anzahl Rennen für das Testing
    """
    _validate_race_split_inputs(features, target)

    features_sorted = features.reset_index(drop=True).sort_values(
        ["season", "round"], kind="stable"
    )
    target_sorted = target.reset_index(drop=True).loc[features_sorted.index]
    features_sorted = features_sorted.reset_index(drop=True)
    target_sorted = target_sorted.reset_index(drop=True)

    races = (
        features_sorted[["season", "round"]]
        .drop_duplicates()
        .sort_values(["season", "round"], kind="stable")
        .reset_index(drop=True)
    )

    n_races = len(races)
    n_test_races = _get_n_test_races(test_size, n_races)

    n_train_races = n_races - n_test_races
    train_races = races.iloc[:n_train_races]
    test_races = races.iloc[n_train_races:]

    return _split_by_races(
        features_sorted,
        target_sorted,
        train_races,
        test_races,
    )


def random_race_split(
    features: pd.DataFrame,
    target: Union[pd.Series, pd.DataFrame],
    test_size: Union[int, float] = 0.2,
    random_state: int | None = 42,
):
    """
    Teilt Features und Target zufällig auf Rennwochenend-Ebene auf.

    Alle Zeilen mit gleicher Kombination aus `season` und `round` bleiben
    zusammen im Training oder im Testing. `test_size` funktioniert ähnlich
    wie bei sklearn:
    - float zwischen 0 und 1: Anteil der Rennen für das Testing
    - int >= 1: Anzahl Rennen für das Testing
    """
    _validate_race_split_inputs(features, target)

    features_reset = features.reset_index(drop=True)
    target_reset = target.reset_index(drop=True)

    races = (
        features_reset[["season", "round"]]
        .drop_duplicates()
        .sort_values(["season", "round"], kind="stable")
        .reset_index(drop=True)
    )

    n_races = len(races)
    n_test_races = _get_n_test_races(test_size, n_races)

    rng = np.random.default_rng(random_state)
    shuffled_indices = rng.permutation(n_races)
    test_indices = shuffled_indices[:n_test_races]
    train_indices = shuffled_indices[n_test_races:]

    train_races = (
        races.iloc[train_indices]
        .sort_values(["season", "round"], kind="stable")
        .reset_index(drop=True)
    )
    test_races = (
        races.iloc[test_indices]
        .sort_values(["season", "round"], kind="stable")
        .reset_index(drop=True)
    )

    return _split_by_races(
        features_reset,
        target_reset,
        train_races,
        test_races,
    )
