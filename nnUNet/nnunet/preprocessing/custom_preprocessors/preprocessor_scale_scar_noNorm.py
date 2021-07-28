# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 11:30:31 2021

@author: s166895
"""

import numpy as np
from nnunet.preprocessing.preprocessing import PreprocessorFor2D, resample_patient


class GenericPreprocessor_scale_scar_noNorm(PreprocessorFor2D):
    """
        The images are already normalized and thus normalization here is not necessary.
        That is why is interfered with the nnUnet and the normalization is taken away.
        This is only interfered in the 2D approach!!
    """
    def resample_and_normalize(self, data, target_spacing, properties, seg=None, force_separate_z=None):
        ############ THIS PART IS IDENTICAL TO PARENT CLASS ################

        original_spacing_transposed = np.array(properties["original_spacing"])[self.transpose_forward]
        before = {
            'spacing': properties["original_spacing"],
            'spacing_transposed': original_spacing_transposed,
            'data.shape (data is transposed)': data.shape
        }
        target_spacing[0] = original_spacing_transposed[0]
        data, seg = resample_patient(data, seg, np.array(original_spacing_transposed), target_spacing, 3, 1,
                                     force_separate_z=force_separate_z, order_z_data=0, order_z_seg=0,
                                     separate_z_anisotropy_threshold=self.resample_separate_z_anisotropy_threshold)
        after = {
            'spacing': target_spacing,
            'data.shape (data is resampled)': data.shape
        }
        print("before:", before, "\nafter: ", after, "\n")

        if seg is not None:  # hippocampus 243 has one voxel with -2 as label. wtf?
            seg[seg < -1] = 0

        properties["size_after_resampling"] = data[0].shape
        properties["spacing_after_resampling"] = target_spacing
        use_nonzero_mask = self.use_nonzero_mask

        assert len(self.normalization_scheme_per_modality) == len(data), "self.normalization_scheme_per_modality " \
                                                                         "must have as many entries as data has " \
                                                                         "modalities"
        assert len(self.use_nonzero_mask) == len(data), "self.use_nonzero_mask must have as many entries as data" \
                                                        " has modalities"

        print("normalization...")

        ############ Make a change in your normalization below

        # this is where normally normalization takes place but that's not necessary now

        return data, seg, properties