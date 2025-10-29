import os.path as osp
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import wandb
from sklearn.manifold import TSNE
from torch import optim

from in_out.arguments import parse_train_test_pc_ae_arguments
from in_out.basics import load_state_dicts, pickle_data, save_state_dicts
from in_out.pointcloud import (
    deterministic_data_loader,
    prepare_pointcloud_dataloaders,
    prepare_vanilla_pointcloud_datasets,
)
from models.model_descriptions import describe_pc_ae

# Argument-handling.
args = parse_train_test_pc_ae_arguments(save_args=True)

# Prepare pointcloud data.
datasets, _ = prepare_vanilla_pointcloud_datasets(args)
data_loaders = prepare_pointcloud_dataloaders(datasets, args)

# Make an AE.
device = torch.device("cuda:" + str(args.gpu_id))
model = describe_pc_ae(args).to(device)

if args.load_pretrained_model:
    best_epoch = load_state_dicts(args.pretrained_model_file, model=model)
    print("Loading pretrained model @epoch", best_epoch)
    print("Losses for this model/epoch:")
    for split in ["train", "val", "test"]:
        loss = model.reconstruct(data_loaders[split], device=device)[-1]
        print(split, loss)

# Train it.
if args.do_training:
    model_name = "best_model.pt"
    save_new_model_file = osp.join(args.log_dir, model_name)

    # Optimization
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=args.lr_patience, min_lr=5e-7
    )

    start_epoch = 1
    min_val_loss = np.inf
    val_not_improved = 0

    with wandb.init(project="Train PC AE", config=args) as run:
        run.watch(model, log_freq=100)

        for epoch in tqdm.tqdm(range(start_epoch, start_epoch + args.max_train_epochs)):
            np.random.seed()
            train_loss = model.train_for_one_epoch(
                data_loaders["train"], optimizer, device=device
            )
            val_loss = model.reconstruct(data_loaders["val"], device=device)[-1]
            lr_scheduler.step(val_loss)

            test_recons, _, _, test_loss = model.reconstruct(
                data_loaders["test"], device=device
            )
            print(
                "{}, {:.6f}, {:.6f}, {:.6f}".format(
                    epoch, train_loss, test_loss, val_loss
                ),
                end=" ",
            )
            run.log(
                {"train_loss": train_loss, "test_loss": test_loss, "val_loss": val_loss}
            )

            if val_loss < min_val_loss:
                print("* validation loss improved *")
                min_val_loss = val_loss
                save_state_dicts(
                    save_new_model_file,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                )
                val_not_improved = 0
            else:
                val_not_improved += 1
                if val_not_improved == args.train_patience:
                    print(
                        f"Validation loss did not improve for {val_not_improved} consecutive epochs. Training is "
                        f"stopped."
                    )
                    break
                print()

        # Load model with best per-validation loss.
        best_epoch = load_state_dicts(osp.join(args.log_dir, model_name), model=model)
        print("per-validation optimal epoch", best_epoch)
        print("losses at this epoch:", best_epoch)
        for split in ["train", "test"]:
            reconstructions, inputs, losses_per_example, loss = model.reconstruct(
                data_loaders[split], device=device
            )

            table = wandb.Table(["Input", "Output"])

            for i in range(0, 46, 5):
                input_shape = wandb.Object3D(np.array(inputs[0][i]))
                output_shape = wandb.Object3D(np.array(reconstructions[0][i]))
                table.add_data(input_shape, output_shape)

            run.log({f"{split}_examples": table})
            print(split, loss)

        train_loader = deterministic_data_loader(
            data_loaders["train"],
            **{
                "batch_size": args.batch_size,
                "worker_init_fn": lambda x: np.random.seed(seed=int(args.random_seed)),
            },
        )
        train_latents, train_classes = model.embed_dataset(train_loader, device=device)
        unique_classes = np.unique(train_classes)

        tsne = TSNE(n_components=2, random_state=int(args.random_seed))
        z2d = tsne.fit_transform(train_latents)

        fig, ax = plt.subplots(figsize=(10, 8))
        for cls in unique_classes:
            mask = train_classes == cls
            ax.scatter(z2d[mask, 0], z2d[mask, 1], s=6, alpha=0.9)

        ax.set_title("Latent Space Visualization using t-SNE")
        ax.set_xlabel("Latent Dimension 1")
        ax.set_ylabel("Latent Dimension 1")

        run.log({"tsne_train_latents": wandb.Image(fig)})
        plt.close(fig)

    wandb.finish()


# Extracting latent codes of the above trained system.
if args.extract_latent_codes:
    uid_to_latent = dict()
    n_data_points = 0
    for split in data_loaders:
        loader = data_loaders[split]
        n_data_points += len(loader.dataset)

        if split == "train":
            loader = deterministic_data_loader(
                loader,
                **{
                    "batch_size": args.batch_size,
                    "worker_init_fn": lambda x: np.random.seed(
                        seed=int(args.random_seed)
                    ),
                },
            )

        latents, _ = model.embed_dataset(loader, device=device)
        data_uids = loader.dataset.model_metadata["model_uid"]

        if len(data_uids) != len(latents):
            raise ValueError(
                "The pointcloud dataset/loader has to have the model_uid attribute set. "
                "See: "
            )

        for k, v in zip(data_uids, latents):
            uid_to_latent[k] = v

    if n_data_points != len(uid_to_latent):
        warnings.warn(
            "The uids in the underlying pointcloud dataset are -not- unique. This can lead to unexpected "
            "behavior."
        )

    save_out_latent_file = osp.join(args.log_dir, "latent_codes.pkl")
    pickle_data(save_out_latent_file, uid_to_latent)
