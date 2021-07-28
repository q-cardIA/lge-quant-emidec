# -*- coding: utf-8 -*-
"""
Name: D.R.P.R.M. Lustermans
Version: 1
Date: 21-06-2021
Email: d.r.p.r.m.lustermans@student.tue.nl
GitHub: https://github.com/DidierLustermans
"""

import os 
import numpy as np
import shutil

import utils
import models

DATA_DIR = "emidec-segmentation-testset-1.0.0"
NNUNET_DIR = os.path.abspath('21_06_nnUNet')
MYO_IM_SHAPE = 128 # Shape of image used for myocardium segmentation
SCAR_IM_SHAPE = 64 # Shape of image used for scar segmentation
AVERAGE_BB = 134 # precomputed from training set

# Bounding Box Model parameters
SAVED_BB_MODEL = "Bounding_box_2021.h5"
BB_STRIDE = (1,1)
BB_KERNEL = (3,3)
BB_BATCH_SIZE = 32
BB_EPOCHS = 3000
BB_PATIENCE = 300
BB_SHAPE = [256,256]
LAMBDA_L2 = 0.001   
LEARNING_RATE = 0.001
DOUBLE_CONV = True

# Set up environmental variables
os.environ['nnUNet_raw_data_base'] = os.path.join(NNUNET_DIR, "nnUNet_data")
os.environ['nnUNet_preprocessed'] = os.path.join(NNUNET_DIR, 'segmentation_preprocessed')
os.environ['RESULTS_FOLDER'] = os.path.join(NNUNET_DIR, 'segmentation_output')

data, second_basal_data, patients_begin_idx, voxel_info, original_image_pos, switch_labels = utils.read_images(BB_SHAPE,DATA_DIR)
num_patients = second_basal_data.shape[0]
patients_begin_idx.append(len(data))
second_basal_data_norm = utils.normalize_images(second_basal_data)

bb_model = models.bounding_box_CNN(BB_STRIDE, BB_KERNEL, BB_SHAPE, LAMBDA_L2, LEARNING_RATE, DOUBLE_CONV)
bb_model.load_weights(SAVED_BB_MODEL)

predictions = bb_model.predict(second_basal_data_norm[...,np.newaxis])

x_normalization = data.shape[1]//2
y_normalization = data.shape[2]//2
predictions = utils.denormalize_transformations(predictions, x_normalization, y_normalization)
predictions[:, :2] = np.around(predictions[:, :2])

segmentation_input = np.zeros((data.shape[0], MYO_IM_SHAPE, MYO_IM_SHAPE))
box_positions = []
box_scales = []
for pt in range(num_patients):
    slices = int(patients_begin_idx[pt+1] - patients_begin_idx[pt])
    start_idx = patients_begin_idx[pt]         
    for sl in range(slices):   
        segmentation_input[start_idx+sl,:,:], box_pos, box_scale = utils.post_process_bb(data[start_idx+sl,:,:], predictions[pt,:], AVERAGE_BB, MYO_IM_SHAPE)
        box_positions.append(box_pos)
        box_scales.append(box_scale)

normalized_segmentation_input = utils.normalize_images(segmentation_input)[...,np.newaxis]

if not os.path.isdir("tmp_cropped_images"):
    os.mkdir("tmp_cropped_images")
utils.save_test_myo(normalized_segmentation_input, "tmp_cropped_images")

if not os.path.isdir("tmp_myo_segs"):
    os.mkdir("tmp_myo_segs")

os.system("nnUNet_predict -i tmp_cropped_images -o tmp_myo_segs -t 801 -m 2d -f 0 -p nnUNet_scar_noNorm")

nifti_file_name = "Myocardium"
myo_task = "myocardium"
normalized_segmentation_input = normalized_segmentation_input[:,:,:,0]
# Load the myocardium prediction
myocardium_labels = utils.read_predicted_labels(normalized_segmentation_input, nifti_file_name, "tmp_myo_segs", myo_task)

shutil.rmtree("tmp_cropped_images")
shutil.rmtree("tmp_myo_segs")

# checks if myocardium segmentations are too small or not closed loops 
# if this is the case, it creates augmented versions of the image to be re-segmented
augmented_myocardium, incorrect_myocardium_idx, save_shifts_x, save_shifts_y, box_pos_shifts, box_scale_shifts = utils.myocardium_seg_check(data, myocardium_labels, predictions, patients_begin_idx, AVERAGE_BB, MYO_IM_SHAPE)

if incorrect_myocardium_idx != []:
    normalized_augmented_myocardium = utils.normalize_images(augmented_myocardium)
    if not os.path.isdir("tmp_cropped_images"):
        os.mkdir("tmp_cropped_images")
    utils.save_test_myo(normalized_augmented_myocardium, "tmp_cropped_images")
    if not os.path.isdir("tmp_myo_segs"):
        os.mkdir("tmp_myo_segs")
    os.system("nnUNet_predict -i tmp_cropped_images -o tmp_myo_segs -t 801 -m 2d -f 0 -p nnUNet_scar_noNorm")

    # Load the augmented myocardium prediction
    augmented_myocardium_labels = utils.read_predicted_labels(normalized_augmented_myocardium, nifti_file_name, "tmp_myo_segs", myo_task)
    myocardium_labels = utils.maj_vote_augmentations(normalized_augmented_myocardium, myocardium_labels, augmented_myocardium_labels, incorrect_myocardium_idx, save_shifts_x, save_shifts_y)

    shutil.rmtree("tmp_cropped_images")
    shutil.rmtree("tmp_myo_segs")


# Crop the images to 64x64 based on the gravity center of the predicted myocardium
scar_input, scar_myocardium_labels, uncut_positions = utils.cut_images(segmentation_input,myocardium_labels)
scar_input = utils.normalize_images(scar_input)
normalize_scar_input = utils.scar_normalize(scar_input, scar_myocardium_labels) 

if not os.path.isdir("tmp_scar_segs"):
    os.mkdir("tmp_scar_images")

utils.save_test_scar(normalize_scar_input, "tmp_scar_images")
if not os.path.isdir("tmp_scar_segs"):
    os.mkdir("tmp_scar_segs")

os.system("nnUNet_predict -i tmp_scar_images -o tmp_scar_segs -t 803 -m 2d -f 0 -p nnUNet_scar_noNorm")


nifti_file_name = "ScaringMyocardium"
myo_task = "scar"
# # Load the myocardium prediction
scar_labels = utils.read_predicted_labels(normalize_scar_input[:,:,:,0], nifti_file_name, "tmp_scar_segs", myo_task)

shutil.rmtree("tmp_scar_images")
shutil.rmtree("tmp_scar_segs")

processed_scar_labels, scar_percentage_slice, scar_percentage_subject, scar_volume_slice, scar_volume_subject, myo_volume_slice, myo_volume_subject = utils.scar_post_processing(normalize_scar_input, myocardium_labels, scar_labels, patients_begin_idx, voxel_info)

predicted_labels = utils.return_image_size_and_save(scar_myocardium_labels, processed_scar_labels, uncut_positions, box_positions, box_scales, original_image_pos, patients_begin_idx, DATA_DIR, switch_labels)
