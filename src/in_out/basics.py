import logging
import os
import os.path as osp
import sys
import warnings

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
