from .mlp import MLP
from .point_net import PointNet
from .pointcloud_autoencoder import PointcloudAutoencoder


def describe_pc_ae(args):
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

    model = PointcloudAutoencoder(ae_encoder, ae_decoder)
    return model
