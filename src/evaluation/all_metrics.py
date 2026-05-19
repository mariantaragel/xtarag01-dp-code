"""
Functions implementing metric evaluations for ChangeIt3d (C3D).

Notice. If you want to use your own pre-trained shape_classifier, or shape_part_classifier, be sure to adapt the input/output
to/of the provided ChangeIt3D model so the 3D data (e.g., PointClouds) are consistently scaled/aligned across those neural-networks.

To this end, see the input `network_input_transformations' functions used below by the ``run_all_metrics'' function.

circa 2022, Panos Achlioptas (https://optas.github.io)
"""

import os.path as osp
from functools import partial

import numpy as np
import pandas as pd
import torch

from evaluation.fpd import calculate_fpd
from evaluation.generic_metrics import (
    chamfer_dists,
    difference_in_probability,
    get_clf_probabilities,
)
from evaluation.listening_based import listening_fit_on_raw_pcs
from in_out.basics import unpickle_data
from in_out.pointcloud import (
    center_in_unit_sphere,
    swap_axes_of_pointcloud,
)

network_transformations = dict()

## this is the transformation we use in our pretrained pc-based-classifier
## YOU MUST Change this accordingly if you use a different classifier
network_transformations["shape_classifier"] = partial(
    center_in_unit_sphere, in_place=False
)


def pc_transform_for_part_predictor(pc):
    pc = swap_axes_of_pointcloud(pc, [0, 2, 1])
    pc = center_in_unit_sphere(pc)
    return pc


## this is the transformation we use in our pretrained part-predictor-classifier
## YOU MUST Change this accordingly if you use a different part-predictor
network_transformations["part_predictor"] = pc_transform_for_part_predictor


@torch.no_grad()
def run_all_metrics(
    transformed_shapes,
    gt_pcs,
    gt_classes,
    sentences,
    vocab,
    args,
    logger,
    network_input_transformations=network_transformations,
):
    """
    transformed_shapes:  (np.array): N_shapes, N_points_per_shape, 3
    gt_pcs: (np.array): N_shapes, N_points_per_shape, 3

    gt_classes: (1-D iterable) with names of classes per shape

    sentences: (list of strings) strings used to transform gt_pcs into the transformed_shapes, they are assumed to
                                 be space tokenizable already (i.e., by a simple .split() operation)
    """
    print(transformed_shapes.max(), gt_pcs.max())
    print(transformed_shapes.mean(axis=(0,1)), gt_pcs.mean(axis=(0,1)))
    results_on_metrics = dict()

    device = torch.device("cuda:" + str(args.gpu_id))

    # Prepare input:
    gt_classes = pd.Series(
        gt_classes, name="shape_class"
    )  # convert to pandas for ease of use via .groupby
    tokens = pd.Series([sent.split() for sent in sentences])
    max_len = tokens.apply(len).max()
    tokens_encoded = np.stack(
        tokens.apply(lambda x: vocab.encode(x, max_len=max_len))
    )  # encode to ints and put them in a N-shape array

    #############################################
    # Test-1: Holistic (regular) Chamfer distance
    #############################################
    scale_chamfer_by = 1000
    batch_size_for_cd = 2048
    for cmp_with in [gt_pcs]:
        holistic_cd_mu, holistic_cds = chamfer_dists(
            cmp_with,
            transformed_shapes,
            bsize=min(len(transformed_shapes), batch_size_for_cd),
            device=device,
        )
        torch.cuda.empty_cache()
        score = round(holistic_cd_mu * scale_chamfer_by, 3)
        score_per_class = (
            pd.concat([gt_classes, pd.DataFrame(holistic_cds.tolist())], axis=1)
            .groupby("shape_class")
            .mean()
            * scale_chamfer_by
        ).round(3)
        score_per_class = score_per_class.reset_index().rename(
            columns={0: "holistic-chamfer"}
        )

        logger.info(f"Chamfer Distance (all pairs), Average: {score}")
        logger.info("Chamfer Distance (all pairs), Average, per class:")
        logger.info(score_per_class)

        results_on_metrics["Chamfer_holistic_cds"] = holistic_cds
        results_on_metrics["Chamfer_all_pairs_average"] = score
        results_on_metrics["Chmafer_all_pairs_per_class"] = score_per_class

    #############################################
    # Test-2: LAB
    #############################################
    # Loading (optionally) a separately trained ORACLE neural listener, to be used for LAB measurements.
    oracle_listener = None
    if args.pretrained_oracle_listener:
        oracle_listener = torch.load(args.pretrained_oracle_listener).eval().to(device)
        _, all_boosts, avg_wins, avg_boost = listening_fit_on_raw_pcs(
            gt_pcs, transformed_shapes, tokens_encoded, oracle_listener, device=device
        )

        torch.cuda.empty_cache()
        logger.info(f"LAB Average:{avg_boost}")
        logger.info(
            f"LAB-related-metric: Times edit is favored by listener against the original input, Average:{avg_wins}"
        )
        results_on_metrics["LAB_avg"] = avg_boost
        results_on_metrics["LAB_wins"] = avg_wins

        score_per_class = (
            pd.concat([gt_classes, pd.DataFrame(all_boosts.tolist())], axis=1)
            .groupby("shape_class")
            .mean()
        )
        score_per_class = score_per_class.reset_index().rename(columns={0: "LAB"})
        logger.info("LAB (all pairs), Average, per class:")
        logger.info(score_per_class)

        results_on_metrics["LAB_per_class"] = score_per_class

    #############################################
    # Test-3: Class-Distortion
    #############################################

    # Loading (optionally) a shape-clf to measure the Class-Distortion (CD) score
    shape_clf = None
    if args.pretrained_shape_classifier is not None:
        shape_clf = torch.load(args.pretrained_shape_classifier).to(device)
        clf_idx_file = osp.join(
            osp.dirname(args.pretrained_shape_classifier), "class_name_to_idx.pkl"
        )
        clf_name_to_idx = next(unpickle_data(clf_idx_file))
        logger.info(
            f"A classifier trained to recognize {len(clf_name_to_idx)} shape classes was loaded."
        )

    if shape_clf is not None:
        collected_probs = dict()  # get the classification probabilities for the shapes * pre and post* the edit
        for shapes, tag in zip([gt_pcs, transformed_shapes], ["gt", "transformed"]):
            shapes_normalized = np.array(
                [network_input_transformations["shape_classifier"](s) for s in shapes]
            )
            collected_probs[tag] = get_clf_probabilities(
                shape_clf,
                shapes_normalized,
                clf_feed_key="pointcloud",
                clf_res_key="class_logits",
                channel_last=True,
                bsize=500,
                device=device,
            )

        gt_class_labels = gt_classes.apply(lambda x: clf_name_to_idx.get(x, None))

        if gt_class_labels.isna().sum() > 0:
            raise ValueError(
                "The classifier was not trained on some of the object classes you are trying to use it for evaluation."
            )
        gt_class_labels = torch.Tensor(gt_class_labels.tolist()).long()

        scores_per_class = dict()
        total_avg_class_distortion = 0
        for u in gt_classes.unique():
            idx_per_class = (gt_classes[gt_classes == u].index).tolist()

            diff = difference_in_probability(
                collected_probs["gt"][idx_per_class],
                collected_probs["transformed"][idx_per_class],
                gt_class_labels[idx_per_class],
            )

            scores_per_class[u] = diff
            logger.info(f"\n (Average) Class Distortion for {u}: {scores_per_class[u]}")
            total_avg_class_distortion += diff * len(idx_per_class)

        total_avg_class_distortion /= len(gt_classes)
        logger.info(
            f"\n (Average) Class Distortion (all classes): {total_avg_class_distortion}"
        )

        # print more stats
        for pcs in ["gt", "transformed"]:
            pred = collected_probs[pcs].argmax(1)
            guessed_correct = pred == gt_class_labels
            logger.info(
                f"\nThe classifier guesses the classes of the ** {pcs} pointclouds ** with accuracy {guessed_correct.double().mean()}"
            )
            per_class_guessing = pd.concat(
                [gt_classes, pd.Series(guessed_correct, name="guessed_correct")], axis=1
            )
            logger.info(
                per_class_guessing.groupby("shape_class")["guessed_correct"]
                .mean()
                .to_markdown()
            )
            logger.info("\n")

        results_on_metrics["Class-Distortion (all classes, average)"] = (
            total_avg_class_distortion
        )
        results_on_metrics["Class-Distortion (per class average)"] = pd.DataFrame(
            scores_per_class, index=["CD"]
        ).transpose()
        torch.cuda.empty_cache()

    #############################################
    # Test-5: Frechet Pointcloud Distance
    #############################################

    if args.pretrained_shape_classifier and args.compute_fpd:
        fpd_scores_per_class = dict()
        total_avg_fpd = 0
        for u in gt_classes.unique():
            idx_per_class = (gt_classes[gt_classes == u].index).tolist()
            input_gt = torch.from_numpy(
                np.array(
                    [
                        network_input_transformations["shape_classifier"](s)
                        for s in gt_pcs[idx_per_class]
                    ]
                )
            )
            input_trans = torch.from_numpy(
                np.array(
                    [
                        network_input_transformations["shape_classifier"](s)
                        for s in transformed_shapes[idx_per_class]
                    ]
                )
            )

            fpd_scores_per_class[u] = calculate_fpd(
                input_gt,
                input_trans,
                pretrained_model_file=args.pretrained_shape_classifier,
                batch_size=500,
            )
            logger.info(f"Class = {u}, FPD-score = {round(fpd_scores_per_class[u], 3)}")

            total_avg_fpd += fpd_scores_per_class[u] * len(idx_per_class)

        total_avg_fpd /= len(gt_classes)
        logger.info(f"Average across all classes={round(total_avg_fpd, 3)}\n")

        results_on_metrics["FPD (all classes, average)"] = total_avg_fpd
        results_on_metrics["FPD (per class average)"] = pd.DataFrame(
            fpd_scores_per_class, index=["FPD"]
        ).transpose()
        torch.cuda.empty_cache()

    #############################################
    # Test-6: Part-Based (Localized) Metrics
    #############################################

    if args.shape_part_classifiers_top_dir:
        raise NotImplementedError()

    return results_on_metrics
