from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Training config definition
# -----------------------------------------------------------------------------
_C = CN()

# -----------------------------------------------------------------------------
# System
# -----------------------------------------------------------------------------
_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 2
_C.SYSTEM.NUM_CPUS = 1


# Define a BISM Model
_C.MODEL = CN()
_C.MODEL.BACKBONE = 'bism_unext'
_C.MODEL.MODEL = 'generic'
_C.MODEL.IN_CHANNELS = 1
_C.MODEL.OUT_CHANNELS = 3
_C.MODEL.DIMS = [32, 64, 128, 64, 32]
_C.MODEL.DEPTHS = [2, 2, 2, 2, 2]
_C.MODEL.KERNEL_SIZE = 7
_C.MODEL.DROP_PATH_RATE = 0.0
_C.MODEL.LAYER_SCALE_INIT_VALUE = 1.
_C.MODEL.ACTIVATION = 'gelu'
_C.MODEL.BLOCK = 'block3d'
_C.MODEL.CONCAT_BLOCK = 'concatconv3d'
_C.MODEL.UPSAMPLE_BLOCK = 'upsamplelayer3d'
_C.MODEL.NORMALIZATION ='layernorm'
_C.MODEL.COMPILE = True

# A list of activations, one for each output channel
# Only matters if model name is "generic"
_C.MODEL.OUTPUT_ACTIVATIONS = [None]


# Only useful for torchvision maskRCNN
_C.MODEL.ANCHOR_SIZES = ((32,), (64,), (128,), (256,), (512,))
_C.MODEL.ANCHOR_ASPECT_RATIOS = ((0.5, 1.0, 2.0),)

# Training Configurations
_C.TRAIN = CN()
_C.TRAIN.DISTRIBUTED = True
_C.TRAIN.PRETRAINED_MODEL_PATH = ''


# Choose here what kind of training you want to do! Default is to predict 3D affinities
# Supported values are: affinities, aclsd, lsd, omnipose mtlsd, semantic, torchvision
_C.TRAIN.TARGET = 'affinities'


# Loss function and their constructor keyowrds
_C.TRAIN.LOSS_FN = 'mse'
_C.TRAIN.LOSS_KEYWORDS = []
_C.TRAIN.LOSS_VALUES = []
_C.TRAIN.TRAIN_DATA_DIR = []
_C.TRAIN.TRAIN_SAMPLE_PER_IMAGE = [1]
_C.TRAIN.TRAIN_BATCH_SIZE = 1
_C.TRAIN.VALIDATION_DATA_DIR = []
_C.TRAIN.VALIDATION_SAMPLE_PER_IMAGE = [1]
_C.TRAIN.VALIDATION_BATCH_SIZE = 1
_C.TRAIN.STORE_DATA_ON_GPU = False
_C.TRAIN.NUM_EPOCHS = 10000
_C.TRAIN.LEARNING_RATE = 5e-4
_C.TRAIN.WEIGHT_DECAY = 1e-6
_C.TRAIN.OPTIMIZER = 'adamw'  # Adam, AdamW, SGD,
_C.TRAIN.OPTIMIZER_EPS = 1e-8
_C.TRAIN.SCHEDULER = 'cosine_annealing_warm_restarts'
_C.TRAIN.SCHEDULER_T0 = 10000 + 1
_C.TRAIN.MIXED_PRECISION = True
_C.TRAIN.N_WARMUP = 1500
_C.TRAIN.SAVE_PATH = './models'
_C.TRAIN.SAVE_INTERVAL = 100
_C.TRAIN.VALIDATE_EPOCH_SKIP = 10
_C.TRAIN.CUDNN_BENCHMARK = True
_C.TRAIN.AUTOGRAD_PROFILE = False
_C.TRAIN.AUTOGRAD_EMIT_NVTX = False
_C.TRAIN.AUTOGRAD_DETECT_ANOMALY = False


## this forces the transform to cast the input data_dict
## to a certain device for calculations. Useful if using dataloader prefetch
_C.TRAIN.TRANSFORM_DEVICE = 'default' # or 'cpu'
_C.TRAIN.DATASET_OUTPUT_DEVICE = 'default' # or 'cpu'
_C.TRAIN.DATALOADER_NUM_WORKERS = 0
_C.TRAIN.DATALOADER_PREFETCH_FACTOR = 0


# Augmentation
_C.AUGMENTATION = CN()
_C.AUGMENTATION.CROP_WIDTH = 300
_C.AUGMENTATION.CROP_HEIGHT = 300
_C.AUGMENTATION.CROP_DEPTH = 20
_C.AUGMENTATION.FLIP_RATE = 0.5
_C.AUGMENTATION.BRIGHTNESS_RATE = 0.4
_C.AUGMENTATION.BRIGHTNESS_RANGE = [-0.1, 0.1]
_C.AUGMENTATION.NOISE_GAMMA = 0.1
_C.AUGMENTATION.NOISE_RATE = 0.2
_C.AUGMENTATION.CONTRAST_RATE = 0.33
_C.AUGMENTATION.CONTRAST_RANGE = [0.75, 2.0]
_C.AUGMENTATION.AFFINE_RATE = 0.66
_C.AUGMENTATION.AFFINE_SCALE = [0.85, 1.1]
_C.AUGMENTATION.AFFINE_YAW = [-180, 180]
_C.AUGMENTATION.AFFINE_SHEAR = [-7, 7]


_C.TARGET = CN()
_C.TARGET.LSD = CN()  # only influences bism.targets.local_shape_descriptors.py
_C.TARGET.LSD.SIGMA = (8, 8, 8)
_C.TARGET.LSD.VOXEL_SIZE = (1, 1, 5)

_C.TARGET.AFFINITIES = CN()  # only influences bism.targets.affinities.py
_C.TARGET.AFFINITIES.N_ERODE = 1
_C.TARGET.AFFINITIES.PAD = 'replicate'  # either 'replicate' or None
_C.TARGET.AFFINITIES.NHOOD = 1

_C.TARGET.OMNIPOSE = CN()
_C.TARGET.OMNIPOSE.EPS = 1e-5
_C.TARGET.OMNIPOSE.MIN_EIKONAL_STEPS = 200  # effectivley the max diameter
_C.TARGET.OMNIPOSE.MAX_DISTANCE = 60

_C.TARGET.SEMANTIC = CN()
_C.TARGET.SEMANTIC.THR = 0.5

_C.TARGET.TORCHVISION = CN()

_C.TARGET.IADB = CN()
_C.TARGET.IADB.MASK_CHANNELS = 1

def get_cfg_defaults():
    r"""Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
