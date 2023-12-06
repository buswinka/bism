from bism.targets.affinities import affinities
from bism.targets.local_shape_descriptors import lsd
from bism.targets.aclsd import aclsd
from bism.targets.mtlsd import mtlsd
from bism.targets.omnipose import omnipose
from bism.targets.semantic import semantic
from bism.targets.torchvision import maskrcnn

_valid_targets = {
    'lsd': lsd,
    'affinities': affinities,
    'mtlsd': mtlsd,
    'aclsd': aclsd,
    'omnipose': omnipose,
    'semantic': semantic,
    'torchvision': maskrcnn
}