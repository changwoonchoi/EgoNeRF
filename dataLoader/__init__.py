# from .llff import LLFFDataset
from .dataset_llff import LLFFDataset
from .dataset_egocentric_video import EgocentricVideoDataset
from .dataset_omniblender import OmniBlenderDataset
from .dataset_omniscenes import OmniscenesDataset


dataset_dict = {
    'llff': LLFFDataset,
    'egocentric': EgocentricVideoDataset,
    'omniblender': OmniBlenderDataset,
    'omniscenes': OmniscenesDataset,
}
