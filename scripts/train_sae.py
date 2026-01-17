import numpy as np

from sparse_autoencoder import (
    ActivationResampler,
    AdamWithReset,
    L2ReconstructionLoss,
    LearnedActivationsL1Loss,
    LossReducer,
    SparseAutoencoder,
)

from dncbm.custom_pipeline import Pipeline
import os
from pathlib import Path

import torch
import numpy as np
import math
import datetime

from sparse_autoencoder import (
    ActivationResampler,
    AdamWithReset,
    L2ReconstructionLoss,
    LearnedActivationsL1Loss,
    LossReducer,
    SparseAutoencoder,
)
import wandb
from time import time
import numpy as np
import torch
import os.path as osp
from dncbm.arg_parser import get_common_parser
from dncbm.utils import common_init
import os.path as osp
import pickle
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="ham")
    args = parser.parse_args()
    data = args.data
    data_dir_activations = {}
    data_dir_activations["img"] = f"file/{data}"
    train = torch.from_numpy(np.load(f"file/{data}/image_emb_train.npy"))
    torch.save(train, osp.join(data_dir_activations["img"], "train"))
    train_val = torch.from_numpy(np.load(f"file/{data}/image_emb_val.npy"))
    torch.save(train_val, osp.join(data_dir_activations["img"], "train_val"))
    result = []
    start_time = time()
    for sparsity in [3e-5, 1.5e-4, 3e-4, 1.5e-3, 3e-3]:
        for learning in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]:
            autoencoder_input_dim: int = 768
            n_learned_features = int(autoencoder_input_dim * 4)
            autoencoder = SparseAutoencoder(n_input_features=autoencoder_input_dim,
                                            n_learned_features=n_learned_features, n_components=1).to("cuda")
            print(f"Autoencoder created at {time() - start_time} seconds")
            print(f"------------Getting Image activations from model: {'densenet121'}")


            # We use a loss reducer, which simply adds up the losses from the underlying loss functions.
            loss = LossReducer(LearnedActivationsL1Loss(
                l1_coefficient=float(sparsity),), L2ReconstructionLoss(),)
            print(f"Loss created at {time() - start_time} seconds")

            optimizer = AdamWithReset(
                params=autoencoder.parameters(),
                named_parameters=autoencoder.named_parameters(),
                lr=float(learning),
                betas=(0.9,
                       0.999),
                eps= 1e-8,
                weight_decay= 0.0,
                has_components_dim=True,
            )

            print(f"Optimizer created at {time() - start_time} seconds")
            actual_resample_interval = 1
            activation_resampler = ActivationResampler(
                resample_interval=actual_resample_interval,
                n_activations_activity_collate=actual_resample_interval,
                max_n_resamples=math.inf,
                n_learned_features=n_learned_features, resample_epoch_freq=500000,
                resample_dataset_size=819200,
            )

            print(f"Activation resampler created at {time() - start_time} seconds")

            pipeline = Pipeline(
                activation_resampler=activation_resampler,
                autoencoder=autoencoder,
                checkpoint_directory=Path(data_dir_activations["img"]),
                loss=loss,
                optimizer=optimizer,
                device="cuda",
                args={},
            )
            print(f"Pipeline created at {time() - start_time} seconds")

            fnames = os.listdir(data_dir_activations["img"])

            train_fnames = []
            train_val_fnames = []
            for fname in fnames:
                if fname.startswith(f"train_val"):
                    train_val_fnames.append(os.path.join(
                        os.path.abspath(data_dir_activations["img"]), fname))
                elif fname.startswith(f"train"):
                    train_fnames.append(os.path.join(
                        os.path.abspath(data_dir_activations["img"]), fname))

            print(f"Train and Train_val fnames created at {time() - start_time} seconds")

            # It takes the train activations and inside split it into train_activations and train_val_activations
            pipeline.run_pipeline(
                train_batch_size=int(4096),
                checkpoint_frequency=500000,
                val_frequency=50000,
                num_epochs=200,
                train_fnames=train_fnames,
                train_val_fnames=train_val_fnames,
                start_time=start_time,
                resample_epoch_freq=500000,
            )

            print(f"-------total time taken------ {np.round(time()-start_time,3)}")
            autoencoder.eval()
            with torch.no_grad():
                recon = autoencoder(train_val.cuda().float()).decoded_activations.squeeze()
                error = torch.mean((recon-train_val.cuda())**2)
                result.append({"sparsity":sparsity, "learning":learning, "error":error})
    learning = result[np.argmin([i['error'].item() for i in result])]['learning']
    sparsity = result[np.argmin([i['error'].item() for i in result])]['sparsity']
    autoencoder_input_dim: int = 768
    n_learned_features = int(autoencoder_input_dim * 4)
    autoencoder = SparseAutoencoder(n_input_features=autoencoder_input_dim,
                                    n_learned_features=n_learned_features, n_components=1).to("cuda")
    print(f"Autoencoder created at {time() - start_time} seconds")
    print(f"------------Getting Image activations from model: {'densenet121'}")


    # We use a loss reducer, which simply adds up the losses from the underlying loss functions.
    loss = LossReducer(LearnedActivationsL1Loss(
        l1_coefficient=float(sparsity),), L2ReconstructionLoss(),)
    print(f"Loss created at {time() - start_time} seconds")

    optimizer = AdamWithReset(
        params=autoencoder.parameters(),
        named_parameters=autoencoder.named_parameters(),
        lr=float(learning),
        betas=(0.9,
               0.999),
        eps= 1e-8,
        weight_decay= 0.0,
        has_components_dim=True,
    )

    print(f"Optimizer created at {time() - start_time} seconds")
    actual_resample_interval = 1
    activation_resampler = ActivationResampler(
        resample_interval=actual_resample_interval,
        n_activations_activity_collate=actual_resample_interval,
        max_n_resamples=math.inf,
        n_learned_features=n_learned_features, resample_epoch_freq=500000,
        resample_dataset_size=819200,
    )

    print(f"Activation resampler created at {time() - start_time} seconds")

    pipeline = Pipeline(
        activation_resampler=activation_resampler,
        autoencoder=autoencoder,
        checkpoint_directory=Path(data_dir_activations["img"]),
        loss=loss,
        optimizer=optimizer,
        device="cuda",
        args={},
    )
    print(f"Pipeline created at {time() - start_time} seconds")

    fnames = os.listdir(data_dir_activations["img"])

    train_fnames = []
    train_val_fnames = []
    for fname in fnames:
        if fname.startswith(f"train_val"):
            train_val_fnames.append(os.path.join(
                os.path.abspath(data_dir_activations["img"]), fname))
        elif fname.startswith(f"train"):
            train_fnames.append(os.path.join(
                os.path.abspath(data_dir_activations["img"]), fname))

    print(f"Train and Train_val fnames created at {time() - start_time} seconds")

    # It takes the train activations and inside split it into train_activations and train_val_activations
    pipeline.run_pipeline(
        train_batch_size=int(4096),
        checkpoint_frequency=500000,
        val_frequency=50000,
        num_epochs=200,
        train_fnames=train_fnames,
        train_val_fnames=train_val_fnames,
        start_time=start_time,
        resample_epoch_freq=500000,
    )

    print(f"-------total time taken------ {np.round(time()-start_time,3)}")
    autoencoder.eval()
    with torch.no_grad():
        activations = []
        for i in range(len(train)//4096+1):
            activations.append(autoencoder(train[i*4096:(i*4096+4096)].cuda().float()).learned_activations)
        activations = torch.cat(activations, dim=0).cpu().detach().squeeze().numpy()
        s = activations.sum(axis=0)
        idx = 0
        rank = {}
        for i in range(len(s)):
            if s[i] > 0:
                rank[idx] = np.argsort(activations[:, i])[::-1]
                idx += 1
        with open(f"file/{data}/rank.pkl", "wb") as f:
            pickle.dump(rank, f)