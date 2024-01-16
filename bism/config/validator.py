import inspect
import os.path

import torch
from yacs.config import CfgNode

import bism.config.valid as v


def assert_keyword_valid(kwarg, fn):
    valid_args = inspect.getfullargspec(fn).args
    valid_args = [v for v in valid_args if v != "self"]
    assert kwarg in valid_args, (
        f"keyword argument: `{kwarg}` is not a valid keyword argument for function {fn}. "
        f"Expected valid keywords are: {valid_args}. Please check source documentation for more details."
    )


def validate_model_config(cfg: CfgNode):

    mdl = cfg.MODEL

    # model shape makes sense
    assert (
        len(mdl.DIMS) == len(mdl.DEPTHS)
    ), f"cfg.MODEL.DIMS must have the same number of elements as cfg.MODEL.DEPTHS"

    # model is available
    assert (
        mdl.MODEL in v._valid_models.keys()
    ), f"{cfg.MODEL.MODEL=} is not a valid key. Options: {v._valid_models.keys()}"

    # model backbone is available
    assert (
        mdl.BACKBONE in v._valid_backbone_constructors.keys()
    ), f"{cfg.MODEL.BACKBONE=} is not a valid key. Options: {v._valid_backbone_constructors.keys()}"

    # model activation is available
    assert (
        mdl.ACTIVATION in v._valid_activations.keys()
    ), f"{cfg.MODEL.ACTIVATION=} is not a valid key. Options: {v._valid_activations.keys()}"

    # model kernel size is positive
    assert mdl.KERNEL_SIZE >= 1, f"{cfg.MODEL.KERNEL_SIZE=} must be >= 1"

    # model in channelse is positive
    assert mdl.IN_CHANNELS >= 1, f"{cfg.MODEL.IN_CHANNELS=} must be >= 1"

    # model out channels is positive
    assert mdl.OUT_CHANNELS >= 1, f"{cfg.MODEL.OUT_CHANNELS=} must be > 1"

    # model drop path between 0 and 1
    assert 1 >= mdl.DROP_PATH_RATE >= 0, f"{cfg.MODEL.OUT_CHANNELS=} must be > 1"

    # model block is available
    assert (
        mdl.BLOCK in v._valid_model_blocks.keys()
    ), f"{cfg.MODEL.BLOCK=} is not a valid key. Options: {v._valid_model_blocks.keys()}"

    # model norm is available
    assert (
        mdl.NORMALIZATION in v._valid_normalization.keys()
    ), f"{cfg.MODEL.NORMALIZATION=} is not a valid key. Options: {v._valid_normalization.keys()}"

    assert (
        len(mdl.OUTPUT_ACTIVATIONS) > 0
    ), "cfg.MODEL.OUTPUT_ACTIVATIONS must be a list of valid activations or a list containing `None` = [None]"

    # make sure all output activations are available
    for i, act in enumerate(mdl.OUTPUT_ACTIVATIONS):
        if act is not None:
            assert act in v._valid_activations.keys(), (
                f"Activation: {act} at position {i} of cfg.MODEL.OUTPUT_ACTIVATIONS if not a valid activation."
                f" Options: {v._valid_activations.keys()}"
            )

def validate_train_config(cfg: CfgNode):
    trn = cfg.TRAIN

    # Loss function works
    assert (
        trn.LOSS_FN in v._valid_loss_functions
    ), f"{cfg.TRAIN.LOSS_FN} is not a valid key. Options: {v._valid_loss_functions.keys()}"

    # validate keyword
    for kw in trn.LOSS_KEYWORDS:
        assert_keyword_valid(kw, v._valid_loss_functions[kw])

    # assert the same num of loss keyword and val
    assert len(trn.LOSS_KEYWORDS) == len(trn.LOSS_VALUES), (
        f"There must be a valid keyword for each value in cfg.TRAIN.LOSS_KEYWORDS and cfg.TRAIN.LOSS_VALUES. "
        f"Found {len(trn.LOSS_KEYWORDS)} keywords and {len(trn.LOSS_VALUES)} values values."
    )

    # positive num of data dir

    assert (
        len(trn.TRAIN_DATA_DIR) > 0
    ), f"cfg.TRAIN.DATA_DIR must be a list containing at least one path string of a training data dir. {cfg.TRAIN.DATA_DIR=}"

    # assert the same num of loss keyword and val
    assert len(trn.TRAIN_DATA_DIR) == len(
        trn.TRAIN_SAMPLE_PER_IMAGE
    ), f"There must be a valid integer multiple in {cfg.TRAIN.TRAIN_SAMPLE_PER_IMAGE=} for each directory in cfg.TRAIN.TRAIN_DATA_DIR."

    # assert the same num of loss keyword and val
    assert len(trn.VALIDATION_DATA_DIR) == len(
        trn.VALIDATION_SAMPLE_PER_IMAGE
    ), f"There must be a valid integer multiple in {cfg.TRAIN.VALIDATION_SAMPLE_PER_IMAGE=} for each directory in cfg.TRAIN.VALIDATION_DATA_DIR."

    assert trn.TRAIN_BATCH_SIZE >= 1, f"{cfg.TRAIN.TRAIN_BATCH_SIZE}>=1"

    assert (
        trn.OPTIMIZER in v._valid_optimizers.keys()
    ), f"{cfg.TRAIN.OPTIMIZER =} is not a valid key. Options: {v._valid_optimizers.keys()}"
    assert (
        trn.SCHEDULER in v._valid_lr_schedulers.keys()
    ), f"{cfg.TRAIN.SCHEDULER =} is not a valid key. Options: {v._valid_lr_schedulers.keys()}"

    assert os.path.exists(trn.SAVE_PATH), f"{cfg.TRAIN.SAVE_PATH=} does not exist"

    assert (
        trn.TARGET in v._valid_targets.keys()
    ), f"{cfg.TRAIN.TARGET=} is not a valid key. Options: {v._valid_targets.keys()}"

    # validate pretrained_model will work...
    pretrained_cfg = torch.load(cfg.TRAIN.PRETRAINED_MODEL_PATH, map_location="cpu")
    pass
    assert (
        "cfg" in pretrained_cfg
    ), f"file {cfg.TRAIN.PRETRAINED_MODEL_PATH=} is not a valid model file."
    pretrained_cfg = pretrained_cfg["cfg"]

    assert (
        pretrained_cfg.TRAIN.TARGET == cfg.TRAIN.TARGET
    ), f"Cannot load a model trained with target: {pretrained_cfg.TRAIN.TARGET} to a new model with to be trained against target: {trn.TARGET}."

    assert (
        pretrained_cfg.MODEL.IN_CHANNELS == cfg.MODEL.IN_CHANNELS
    ), f"Pretrained Model with value: {pretrained_cfg.MODEL.IN_CHANNELS=} cannot be mapped to current configuration value: {cfg.MODEL.IN_CHANNELS}"
    assert (
        pretrained_cfg.MODEL.OUT_CHANNELS == cfg.MODEL.OUT_CHANNELS
    ), f"Pretrained Model with value: {pretrained_cfg.MODEL.OUT_CHANNELS=} cannot be mapped to current configuration value: {cfg.MODEL.OUT_CHANNELS}"
    assert (
        pretrained_cfg.MODEL.DIMS == cfg.MODEL.DIMS
    ), f"Pretrained Model with value: {pretrained_cfg.MODEL.DIMS=} cannot be mapped to current configuration value: {cfg.MODEL.DIMS}"
    assert (
        pretrained_cfg.MODEL.DEPTHS == cfg.MODEL.DEPTHS
    ), f"Pretrained Model with value: {pretrained_cfg.MODEL.DEPTHS=} cannot be mapped to current configuration value: {cfg.MODEL.DEPTHS}"
    assert (
        pretrained_cfg.MODEL.KERNEL_SIZE == cfg.MODEL.KERNEL_SIZE
    ), f"Pretrained Model with value: {pretrained_cfg.MODEL.KERNEL_SIZE=} cannot be mapped to current configuration value: {cfg.MODEL.KERNEL_SIZE}"
    assert (
        pretrained_cfg.MODEL.BLOCK == cfg.MODEL.BLOCK
    ), f"Pretrained Model with value: {pretrained_cfg.MODEL.BLOCK=} cannot be mapped to current configuration value: {cfg.MODEL.BLOCK}"


def validate_config(cfg: CfgNode):
    validate_model_config(cfg)
    validate_train_config(cfg)

if __name__ == '__main__':
    from bism.config.config import get_cfg_defaults

    def load_cfg_from_filename(filename: str):
        """Load configurations.
        """
        # Set configurations
        cfg = get_cfg_defaults()
        if os.path.exists(filename):
            cfg.merge_from_file(filename)
        else:
            raise ValueError('Could not find config file from path!')
        cfg.freeze()

        return cfg

    cfg = load_cfg_from_filename('/home/chris/Dropbox (Partners HealthCare)/bism/Configs/test_iadb_config.yaml')

    validate_model_config(cfg)
    validate_train_config(cfg)
