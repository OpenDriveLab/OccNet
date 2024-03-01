import numpy as np
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadOccGTFromFile(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    note that we read image in BGR style to align with opencv.imread
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self):
        pass

    def __call__(self, results):
        occ_labels = np.load(results['occ_path'])
        semantics = occ_labels['semantics']
        flow = occ_labels['flow']
        results['voxel_semantics'] = semantics
        results['voxel_flow'] = flow

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return "{} (data_root={}')".format(
            self.__class__.__name__, self.data_root)