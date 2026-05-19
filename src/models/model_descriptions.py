import os.path as osp

from torch import nn

from in_out.basics import load_state_dicts, read_saved_args
from in_out.changeit3d_net import load_pickled_shape_latent_codes
from language.vocabulary import Vocabulary

from .basic_ops_as_modules import ReLU
from .beta_vae import PointcloudBetaVAE
from .changeit3d_net import LatentDirectionFinder
from .dgcnn import DGCNN
from .folding_net import FoldingNet
from .listening_oriented import TransformerModel, TransformerModelFeature
from .mlp import MLP
from .pc_ae_cls import PointcloudAutoencoderCls
from .point_net import PointNet
from .pointcloud_autoencoder import PointcloudAutoencoder
from .language_encoder import EmbeddingLangEncoder


def describe_pc_ae(args):
    # Make an AE.
    if args.encoder_net == "pointnet":
        ae_encoder = PointNet(init_feat_dim=3, conv_dims=args.encoder_conv_layers)
        encoder_latent_dim = args.encoder_conv_layers[-1]
    elif args.encoder_net == "dgcnn":
        ae_encoder = DGCNN(
            initial_dim=3,
            out_dim=512,
            k_neighbors=20,
            intermediate_feat_dim=[64, 64, 128, 256],
            subtract_from_self=True,
        )
        encoder_latent_dim = 512
    else:
        raise NotImplementedError()

    if args.decoder_net == "mlp":
        ae_decoder = MLP(
            in_feat_dims=encoder_latent_dim,
            out_channels=args.decoder_fc_neurons + [args.n_pc_points * 3],
            b_norm=False,
        )
    elif args.decoder_net == "foldingnet":
        ae_decoder = FoldingNet(encoder_latent_dim, num_points=args.n_pc_points)
    else:
        raise NotImplementedError()

    model = PointcloudAutoencoder(ae_encoder, ae_decoder)
    return model


def describe_pc_ae_cls(args):
    # Make an AE.
    if args.encoder_net == "pointnet":
        ae_encoder = PointNet(init_feat_dim=3, conv_dims=args.encoder_conv_layers)
        encoder_latent_dim = args.encoder_conv_layers[-1]
    else:
        raise NotImplementedError()

    if args.decoder_net == "mlp":
        ae_decoder = MLP(
            in_feat_dims=encoder_latent_dim,
            out_channels=args.decoder_fc_neurons + [args.n_pc_points * 3],
            b_norm=False,
        )
    else:
        raise NotImplementedError()

    ae_classifier = MLP(
        in_feat_dims=encoder_latent_dim,
        out_channels=args.classifier_fc_neurons + [args.n_cls],
        b_norm=False,
    )

    model = PointcloudAutoencoderCls(ae_encoder, ae_decoder, ae_classifier, args.alfa)
    return model


def describe_pc_beta_vae(args):
    # Make an beta-VAE.
    if args.encoder_net == "pointnet":
        ae_encoder = PointNet(init_feat_dim=3, conv_dims=args.encoder_conv_layers)
        encoder_latent_dim = args.encoder_conv_layers[-1]
    else:
        raise NotImplementedError()

    if args.decoder_net == "mlp":
        ae_decoder = MLP(
            in_feat_dims=encoder_latent_dim,
            out_channels=args.decoder_fc_neurons + [args.n_pc_points * 3],
            b_norm=False,
        )
    else:
        raise NotImplementedError()

    model = PointcloudBetaVAE(
        ae_encoder, ae_decoder, args.encoder_conv_layers[-1], args.beta
    )
    return model


def load_pretrained_pc_ae(model_file):
    config_file = osp.join(osp.dirname(model_file), "config.json.txt")
    pc_ae_args = read_saved_args(config_file)

    if pc_ae_args.latent_backbone == "pc_ae":
        pc_ae = describe_pc_ae(pc_ae_args)
    elif pc_ae_args.latent_backbone == "pc_ae_cls":
        pc_ae = describe_pc_ae_cls(pc_ae_args)
    elif pc_ae_args.latent_backbone == "pc_beta_vae":
        pc_ae = describe_pc_beta_vae(pc_ae_args)
    else:
        raise NotImplementedError()

    # if osp.join(pc_ae_args.log_dir, "best_model.pt") != osp.abspath(model_file):
    #     warnings.warn(
    #         "The saved best_model.pt in the corresponding log_dir is not equal to the one requested."
    #     )

    best_epoch = load_state_dicts(model_file, model=pc_ae)
    print(f"Pretrained PC-AE is loaded at epoch {best_epoch}.")
    return pc_ae, pc_ae_args


##
# ChangeIt Models and Ablations
##


def ablations_changeit3d_net(
    vocab, shape_latent_dim, ablation_version, self_contrast=True
):
    d_lang_model = 128
    in_dim = d_lang_model + shape_latent_dim

    editor = MLP(
        in_dim,
        [256, shape_latent_dim, shape_latent_dim, shape_latent_dim],
        b_norm=True,
        remove_final_bias=True,
    )
    stimulus_encoder = MLP(shape_latent_dim, [shape_latent_dim, shape_latent_dim])
    closure = ReLU()

    if ablation_version == "decoupling_mag_direction":
        magnitude_encoder = MLP(in_dim, [256, 128, 64, 1], closure=closure)
        unit_normalize_direction = True
    elif ablation_version == "coupled":
        unit_normalize_direction = False
        magnitude_encoder = None
    else:
        raise ValueError("ablation version of ChangeIt3D not understood.")

    print("Doing ST ablation", ablation_version, "with self contrast", self_contrast)
    
    nhead = 2
    d_hid = 128
    nlayers = 2
    language_dropout = 0.2
    language_model = TransformerModel(len(vocab), d_lang_model, nhead, d_hid, nlayers, language_dropout)
    language_encoder = TransformerModelFeature(language_model)

    model = LatentDirectionFinder(
        language_encoder,
        stimulus_encoder,
        editor,
        magnitude_unit=magnitude_encoder,
        unit_normalize_direction=unit_normalize_direction,
        self_contrast=self_contrast,
    )

    return model


def load_pretrained_changeit3d_net(checkpoint_file, shape_latent_dim=None, vocab=None):
    config_file = osp.join(osp.dirname(checkpoint_file), "config.json.txt")
    args = read_saved_args(config_file)

    if shape_latent_dim is None:
        shape_latent_dim = load_pickled_shape_latent_codes(args)

    if vocab is None:
        vocab = Vocabulary.load(args.vocab_file)

    model = ablations_changeit3d_net(
        vocab, shape_latent_dim, args.shape_editor_variant, args.self_contrast
    )

    best_epoch = load_state_dicts(checkpoint_file, model=model, map_location="cpu")
    model = model.eval()

    return model, best_epoch, args
