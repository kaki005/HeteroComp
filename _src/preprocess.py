import pathlib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import logging


def prepare_event_tensor(
    given_data: pd.DataFrame,
    categorical_idxs: list[str],
    time_idx,
    freq,
    outdir: pathlib.Path,
    save_encoders=False,
):
    data = given_data
    # data = given_data.copy("deep")
    data = data.dropna(subset=(categorical_idxs + [time_idx]))
    # Encode timestamps
    sorted_indice = data[time_idx].argsort(kind='mergesort') # stable sort
    data = data.iloc[sorted_indice]
    if freq != "":
        logging.info(F"{freq=}")
        data[time_idx] = data[time_idx].dt.round(freq)
    timestamps = pd.unique(data[time_idx])
    timepoint_encoder = LabelEncoder()
    data[time_idx] = timepoint_encoder.fit_transform(data[time_idx])
    data[time_idx] = data[time_idx].astype(int)

    # Encode categorical features
    oe = OrdinalEncoder()
    data[categorical_idxs] = oe.fit_transform(data[categorical_idxs].astype(str))
    data[categorical_idxs] = data[categorical_idxs].astype(int)

    if save_encoders:
        time_encoder = pd.DataFrame(
            timepoint_encoder.classes_,
            index=range(len(timepoint_encoder.classes_)),
            columns=["timestamp"],
        )  # Timestamps
        time_encoder.to_csv(outdir / f"{time_idx}.csv.gz", index=False)
        # Categorical features
        for key, feature_elem in zip(categorical_idxs, oe.categories_):
            ctg_encoder = pd.DataFrame(
                feature_elem, index=range(len(feature_elem)), columns=[key]
            )
            ctg_encoder.to_csv(outdir / f"{key}.csv.gz", index=False)
    return data.reset_index(drop=True), oe, timepoint_encoder, timestamps, sorted_indice
