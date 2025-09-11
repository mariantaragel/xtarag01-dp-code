from .point_net import PointNet
from .mlp import MLP
from .pointcloud_autoencoder import PointcloudAutoencoder
from .embedding import Embedding

def describe_pc_ae(args):
    # Make an AE.
    if args.encoder_net == 'pointnet':
        ae_encoder = PointNet(init_feat_dim=3, conv_dims=args.encoder_conv_layers)
        encoder_latent_dim = args.encoder_conv_layers[-1]
    else:
        raise NotImplementedError()

    if args.conditional_net == 'embedding':
        ae_conditional = Embedding(args.num_embeddings, args.embedding_dim)
    else:
        raise NotImplementedError()

    if args.decoder_net == 'mlp':
        decoder_fc_neurons = [x + args.embedding_dim for x in args.decoder_fc_neurons]
        ae_decoder = MLP(in_feat_dims=encoder_latent_dim + args.embedding_dim,
                         out_channels=decoder_fc_neurons + [args.n_pc_points * 3],
                         b_norm=False)
    else:
        raise NotImplementedError()

    model = PointcloudAutoencoder(ae_encoder, ae_conditional, ae_decoder)
    return model