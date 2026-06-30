"""
Adopted from ChangeIt3D by Achlioptas et al.
Modified by Marián Tarageľ (xtarag01)
"""

import json
import logging
import os
import os.path as osp
import pprint
import sys
import warnings
from argparse import ArgumentParser

import torch
from six.moves import cPickle


def create_dir(dir_path):
    """Creates a directory (or nested directories) if they don't exist."""
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path


def pickle_data(file_name, *args):
    """Using (c)Pickle to save multiple python objects in a single file."""
    out_file = open(file_name, "wb")
    cPickle.dump(len(args), out_file, protocol=2)
    for item in args:
        cPickle.dump(item, out_file, protocol=2)
    out_file.close()


def save_state_dicts(checkpoint_file, epoch=None, **kwargs):
    """Save torch items with a state_dict"""
    checkpoint = dict()

    if epoch is not None:
        checkpoint["epoch"] = epoch

    for key, value in kwargs.items():
        checkpoint[key] = value.state_dict()

    torch.save(checkpoint, checkpoint_file)


def read_saved_args(config_file, override_or_add_args=None, verbose=False):
    """
    :param config_file: json file containing arguments
    :param override_args: dict e.g., {'gpu': '0'} will set the resulting arg.gpu to be 0
    :param verbose:
    :return:
    """
    parser = ArgumentParser()
    args = parser.parse_args([])
    with open(config_file, "r") as f_in:
        args.__dict__ = json.load(f_in)

    if override_or_add_args is not None:
        for key, val in override_or_add_args.items():
            args.__setattr__(key, val)

    if verbose:
        args_string = pprint.pformat(vars(args))
        print(args_string)

    return args


def load_state_dicts(checkpoint_file, map_location=None, **kwargs):
    """Load torch items from saved state_dictionaries"""
    if map_location is None:
        checkpoint = torch.load(checkpoint_file)
    else:
        checkpoint = torch.load(checkpoint_file, map_location=map_location)

    for key, value in kwargs.items():
        value.load_state_dict(checkpoint[key])

    epoch = checkpoint.get("epoch")
    if epoch:
        return epoch


def create_logger(log_dir, std_out=True):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")

    # Add logging to file handler
    file_handler = logging.FileHandler(osp.join(log_dir, "log.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Add stdout to also print statements there
    if std_out:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def torch_save_model(model, path):
    """Wrap torch.save to catch standard warning of not finding the nested implementations.
    :param model:
    :param path:

    :return:
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.save(model, path)


def unpickle_data(file_name, python2_to_3=False):
    """Restore data previously saved with pickle_data().
    :param file_name: file holding the pickled data.
    :param python2_to_3: (boolean), if True, pickle happened under python2x, unpickling under python3x.
    :return: an generator over the un-pickled items.
    Note, about implementing the python2_to_3 see
        https://stackoverflow.com/questions/28218466/unpickling-a-python-2-object-with-python-3
    """
    in_file = open(file_name, "rb")
    if python2_to_3:
        size = cPickle.load(in_file, encoding="latin1")
    else:
        size = cPickle.load(in_file)

    for _ in range(size):
        if python2_to_3:
            yield cPickle.load(in_file, encoding="latin1")
        else:
            yield cPickle.load(in_file)
    in_file.close()
