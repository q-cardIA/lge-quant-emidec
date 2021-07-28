# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 11:32:04 2021

@author: s166895
"""

from nnunet.experiment_planning.experiment_planner_baseline_2DUNet_v21 import ExperimentPlanner2D_v21
from nnunet.paths import *

class ExperimentPlanner2D_v21_scar_noNorm(ExperimentPlanner2D_v21):
    """
    used by tutorial nnunet.tutorials.custom_preprocessing
    """
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super().__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.data_identifier = "nnUNet_scar_noNorm"
        self.plans_fname = join(self.preprocessed_output_folder, "nnUNet_scar_noNorm" + "_plans_2D.pkl")

        # The custom preprocessor class we intend to use is GenericPreprocessor_scale_uint8_to_0_1. It must be located
        # in nnunet.preprocessing (any file and submodule) and will be found by its name. Make sure to always define
        # unique names!
        self.preprocessor_name = 'GenericPreprocessor_scale_scar_noNorm'