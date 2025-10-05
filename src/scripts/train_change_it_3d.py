"""
Script to train/test a Language-Assisted 3D Shape Edit/Deformation System (ChangeIt3D)
Finished sometime around 2022.
By Panos Achlioptas (https://optas.github.io)
"""

import os.path as osp

import numpy as np
import torch
import wandb
from torch import nn, optim

from in_out.arguments import parse_train_changeit3d_arguments
from in_out.basics import create_logger, load_state_dicts, save_state_dicts
from in_out.changeit3d_net import prepare_input_data
from in_out.language_contrastive_dataset import LanguageContrastiveDataset
from models.model_descriptions import ablations_changeit3d_net, load_pretrained_pc_ae

##
# Read arguments
##
args = parse_train_changeit3d_arguments()
logger = create_logger(args.log_dir)

##
# Prepare the input data
##
df, shape_to_latent_code, shape_latent_dim, vocab = prepare_input_data(args, logger)


##
# Prepare the data loaders
##
def to_stimulus_func(x):
    return shape_to_latent_code[x]


dataloaders = dict()
for split in ["train", "val", "test"]:
    ndf = df[df.changeit_split == split].copy()
    ndf.reset_index(inplace=True, drop=True)

    seed = None if split == "train" else args.random_seed
    batch_size = args.batch_size if split == "train" else 2 * args.batch_size

    dataset = LanguageContrastiveDataset(
        ndf, to_stimulus_func, n_distractors=1, shuffle_items=False
    )  # important, target *always* last

    dataloaders[split] = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=split == "train",
        num_workers=args.num_workers,
        worker_init_fn=lambda _: np.random.seed(seed),
    )

##
# Build Shape Transformation Model
##
model = ablations_changeit3d_net(
    vocab, shape_latent_dim, args.shape_editor_variant, args.self_contrast
)
device = torch.device("cuda:" + str(args.gpu_id))
model = model.to(device)

##
# Loading pretrained Shape Generator (AutoEncoder).
##
if args.shape_generator_type == "pcae":
    pc_ae, pc_ae_args = load_pretrained_pc_ae(args.pretrained_shape_generator)
    pc_ae = pc_ae.to(device)
    pc_ae = pc_ae.eval()


def from_stimulus_func(latent_shapes, pc_ae, device):
    all_recons = []

    for vals in latent_shapes:
        recons = pc_ae.decoder(torch.from_numpy(vals).to(device))
        recons = recons.view([len(recons), -1, 3]).cpu()
        all_recons.append(recons)

    return all_recons


##
# Optimization
##
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay
)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=args.lr_patience,
    verbose=True,
    min_lr=5e-7,
)

# Load pre-trained listener that will be used for optimizing the changer.
pretrained_listener = torch.load(args.pretrained_listener_file).to(device)
for param in pretrained_listener.parameters():
    param.requires_grad = False


##
# Train it.
##

# torch.backends.cudnn.enabled = False  # uncomment if pretrained listener is based on an LSTM

if args.train:
    epochs_val_not_improved = 0
    best_val_loss = np.Inf
    start_epoch = 0
    checkpoint_file = osp.join(args.log_dir, "best_model.pt")

    logger.info(
        "Start training of the Language Assisted Shape Editing (ChangeIt3DNet)."
    )

    with wandb.init(project="Train Latent Listener", config=args) as run:
        run.watch(model, log_freq=100)

        for epoch in range(start_epoch + 1, start_epoch + args.max_train_epochs + 1):
            np.random.seed()

            # training
            train_losses = model.single_epoch_train(
                pretrained_listener,
                dataloaders["train"],
                criterion,
                optimizer,
                gamma=args.identity_penalty,
                adaptive_id_penalty=args.adaptive_id_penalty,
                device=device,
            )

            run.log({"train_total_loss": train_losses["total_loss"]})
            run.log({"train_listening_loss": train_losses["listening_loss"]})
            run.log({"train_identity_loss": train_losses["identity_loss"]})

            logger.info(f"@epoch-{epoch}")
            logger.info("train/val losses:")
            train_losses_str = " ".join(
                ["{:15} {:.5f}".format(key, val) for key, val in train_losses.items()]
            )
            logger.info(train_losses_str)

            # validation
            val_losses = model.evaluate(
                pretrained_listener,
                dataloaders["val"],
                criterion,
                gamma=args.identity_penalty,
                adaptive_id_penalty=args.adaptive_id_penalty,
                device=device,
            )

            run.log({"val_total_loss": val_losses["total_loss"]})
            run.log({"val_listening_loss": val_losses["listening_loss"]})
            run.log({"val_identity_loss": val_losses["identity_loss"]})

            val_losses_str = " ".join(
                ["{:15} {:.5f}".format(key, val) for key, val in val_losses.items()]
            )
            logger.info(val_losses_str)

            # test
            test_losses = model.evaluate(
                pretrained_listener,
                dataloaders["test"],
                criterion,
                gamma=args.identity_penalty,
                adaptive_id_penalty=args.adaptive_id_penalty,
                device=device,
            )

            run.log({"test_total_loss": test_losses["total_loss"]})
            run.log({"test_listening_loss": test_losses["listening_loss"]})
            run.log({"test_identity_loss": test_losses["identity_loss"]})

            lr_scheduler.step(val_losses["total_loss"])
            if val_losses["total_loss"] < best_val_loss:
                best_val_loss = val_losses["total_loss"]
                epochs_val_not_improved = 0
                save_state_dicts(
                    checkpoint_file,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                )
            else:
                epochs_val_not_improved += 1
                logger.info("* validation loss did not improved *")

            if epochs_val_not_improved == args.train_patience:
                logger.warning(
                    f"Validation loss did not improve for {epochs_val_not_improved} consecutive epochs. Training is stopped."
                )
                break

        logger.info("Training is done!")

        # Load best model per validation set
        best_epoch = load_state_dicts(checkpoint_file, model=model)
        logger.info(f"per-validation optimal epoch {best_epoch}")

        result = model.evaluate(
            pretrained_listener,
            dataloaders["test"],
            criterion,
            gamma=args.identity_penalty,
            adaptive_id_penalty=args.adaptive_id_penalty,
            device=device,
        )

        table = wandb.Table(["Input", "Text", "Output", "Probabilities"])
        for i in range(10):
            input_shapes = from_stimulus_func(result["inputs"], pc_ae, device)
            output_shapes = from_stimulus_func(result["outputs"], pc_ae, device)

            texts = vocab.decode_print(result["texts"][i])
            input_shape_3dobject = wandb.Object3D(input_shapes[i])
            output_shape_3dobject = wandb.Object3D(input_shapes[i])
            probabilities = result["probabilities"][i]
            table.add_data(
                input_shape_3dobject, texts, output_shape_3dobject, probabilities
            )

        run.log({"examples": table})

    wandb.finish()
