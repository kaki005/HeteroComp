import logging
import shutil
import time
from omegaconf import OmegaConf


import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys
import hydra
import tqdm
import gc
from _src import (
    prepare_event_tensor,
    plot_regimeassignment,
    load_dataset,
    print_dataset,
    load_model,
    log_init,
    Config,
    tree_patrition,
)



@hydra.main(version_base=None, config_path="_src/configs", config_name="base_config")
def main(config: Config):
    try:
        log_init()
        np.random.seed(0)
        logger = logging.getLogger("main")
        config.model.alpha = 1 / config.model.k
        config.model.beta = 1 / config.model.k
        config.model.tol_r = 2.0 ** (-config.model.FB)
        outputdir = Path(f"./_out/{config.model.name}/")
        outputdir /= config.data.name
        outputdir /= f"seed{config.data.seed}"
        outputdir /= f"topic{config.model.k}_scale{config.data.time_scale}_width{config.model.width}_initlen{config.data.init_len}"
        # assert not outputdir.exists()
        outputdir.mkdir(exist_ok=True, parents=True)
        OmegaConf.save(config, outputdir / "config.yaml")
        logger.info(f"{outputdir = }")

        categorical_idxs = list(config.data.categorical_idxs)
        continuous_idxs = list(config.data.continuous_idxs)
        time_idx = config.data.time_idx
        raw_df = load_dataset(config.data.name, time_idx, continuous_idxs, categorical_idxs)
        mode_bounds = []
        if config.model.name == "heterocomp":  # grid
            for idx in continuous_idxs:
                bounds, counter = tree_patrition(
                    raw_df[idx].to_numpy(),
                    config.model.num_bins,
                    min_points=1,
                    min_width=1.0,
                )
                mode_bounds.append(bounds)
                raw_df[idx] = pd.cut(
                    raw_df[idx],
                    bins=np.hstack([bounds[:, 0] - 1e-10, bounds[-1, 1]]),
                    labels=False,
                )
                raw_df[idx] = raw_df[idx].astype(int)
        elif config.model.name != "cybercscope":  # discretize
            for idx in continuous_idxs:
                uniques = pd.unique(raw_df[idx].astype(float))
                for val in uniques:
                    if val < 0:
                        logger.info(f"{idx} : ;{val}")
                if uniques.shape[0] > 50:
                    raw_df[idx] = pd.cut(raw_df[idx].astype(float), 10).astype(str)
                logger.info(f"{idx}: {pd.unique(raw_df[idx])}")
            categorical_idxs += continuous_idxs
            continuous_idxs = []

        tensor, category_encoder, timepoint_encoder, timestamps, sorted_indice = prepare_event_tensor(
            raw_df,
            categorical_idxs,
            time_idx,
            freq=config.data.freq,
            outdir=outputdir,
        )
        np.savetxt(outputdir / "sorted_indice.txt", sorted_indice, fmt='%d')
        np.savetxt(outputdir / "timestamps.txt",pd.to_datetime(timestamps).strftime('%Y-%m-%dT%H:%M:%S'), fmt='%s')
        del raw_df

        # Set inlier tensor to train tensor
        if config.model.anomaly and config.data.label_col != None:
            anom_series = tensor[config.data.label_col]
        else:
            anom_series = np.zeros(len(tensor))
        assert len(tensor) == len(anom_series)

        tensor = tensor[[time_idx] + categorical_idxs + continuous_idxs]
        gc.collect()
        print_dataset(
            logger,
            tensor,
            categorical_idxs + continuous_idxs,
            time_idx,
        )
        tensor_shape = tensor.max().values + 1
        tensor_Train = tensor.loc[
            (tensor[config.data.time_idx] < config.data.init_len) & (anom_series == 0),
            :,
        ].reset_index(drop=True)
        train_timestamps = timestamps[pd.unique(tensor_Train[config.data.time_idx])]
        tensor_Train[time_idx] = (
            tensor_Train[time_idx].rank(method="dense", numeric_only=True).astype(int)
            - 1
        )
        logger.info(f"train tensor shape: {tensor_Train.max().values + 1}")

        ### Batch processing (Initialize) ###################
        model = load_model(
            tensor,
            config,
            config.model.name,
            tensor_shape,
            anom_series,
            categorical_idxs,
            mode_bounds,
        )
        start_time = time.perf_counter()
        regime_assignments = model.init_infer(tensor_Train, train_timestamps, 40)
        elapsed_time = time.perf_counter() - start_time
        logger.info(f"Elapsed time(train): {elapsed_time:.2f} [sec]")
        if config.save_batch or config.plot_batch:
            outputdir_s = outputdir / "save" / "train"
            if outputdir_s.exists():
                shutil.rmtree(outputdir_s)
            outputdir_s.mkdir(parents=True)
            # model.save(
            #     outputdir_s, tensor_Train, regime_assignments, [elapsed_time]
            # )
            if config.save_batch:
                model.save_online(outputdir_s, tensor_Train, category_encoder)
            if config.plot_batch:
                model.plot_online(outputdir_s, tensor_Train, train_timestamps)
        del tensor_Train
        del train_timestamps
        gc.collect()
        ### Batch processing (Initialize) ###################

        ### Stream processing ###############################
        start_time_stream_process = time.process_time()
        elapsed_times = []
        max_ = tensor[time_idx].max() + 1
        model.data_len = max_
        for i in tqdm.tqdm(range(0, max_, config.model.width)):
            start_time = time.perf_counter()
            current_tensor = tensor[
                (tensor[time_idx] >= i) & (tensor[time_idx] < (i + config.model.width))
            ]
            current_tensor.loc[:, time_idx] -= i
            stamp = timestamps[i : i + config.model.width]
            shift_id = model.infer_online(
                current_tensor, config.model.iter_num, stamp
            )
            elapsed_time = time.perf_counter() - start_time
            logger.info(f"Elapsed time(online#{i}): {elapsed_time:.2f} [sec]")
            elapsed_times.append(elapsed_time)
            if config.save_batch or config.plot_batch:
                outputdir_s = outputdir / "save" / f"t_{str(i)}/"
                if outputdir_s.exists():
                    shutil.rmtree(outputdir_s)
                outputdir_s.mkdir(parents=True)
                if config.save_batch:
                    model.save_online(outputdir_s, current_tensor, category_encoder)
                if config.plot_batch:
                    model.plot_online(outputdir_s, current_tensor, stamp)
            del current_tensor
            del stamp
            if (i+1) %100 == 0:
                gc.collect()
        model.rgm_update_fin()
        elapsed_time_stream_process = time.process_time() - start_time_stream_process
        logger.info(
            f"Elapsed time(all stream processing): {elapsed_time_stream_process:.2f} [sec]"
        )
        ### Stream processing ###############################

        # save overall results
        model.save(outputdir, tensor, regime_assignments, elapsed_times, category_encoder)
        logger.info(f"result in {outputdir}")

        # viz temporal pattern segmetation
        # time_index = timepoint_encoder.inverse_transform(pd.unique(tensor[time_idx]))
        model.plot(outputdir, category_encoder, timestamps)
        logger.info(regime_assignments)
        # モデルを保存
        try:
            with open(outputdir / "model.pickle", mode="wb") as fo:
                pickle.dump(model, fo)
        except Exception as ex:
            logger.exception(ex)
    except Exception as ex:
        logger.exception(ex)


main()
