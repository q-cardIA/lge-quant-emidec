import os
import numpy as np
from glob import glob
import nibabel as nib
import SimpleITK as sitk
import cv2

from skimage import measure
from skimage.morphology import area_opening
from skimage.morphology import convex_hull_image


def read_images(shape, direct):

    im_files = sorted(glob(os.path.join(direct, "*", "Images", "*.nii.gz")))

    info = []   
    original_positions = []
    switches = [] # keep track of whether axes of images have been transposed
    patient_begin_idx = []
    for num, im_file in enumerate(im_files):
        img = nib.load(im_file)
        img_data = img.get_data()        
        header = img.header
        info.append((num,header['pixdim']))

        # Switch the axis of the image, so the images are in the same direction 
        if img_data.shape[0] > img_data.shape[1]:
            img_data = np.swapaxes(img_data,0,1)
            switches.append(1)
        else:
            switches.append(0)

        if num == 0:
            # Create empty list for data, masks and box sizes
            data = np.zeros((img_data.shape[2], shape[0], shape[1]))
            second_basal_data = np.zeros((len(im_files), shape[0], shape[1]))
            # Integer indicating first position in the list of patients
            length = 0
        else:
            # Integer indicating first position in the list of patients
            length = data.shape[0]
            # Increase the existing array with zeros amount depending on the amount of slices 
            data = np.concatenate((data,np.zeros((img_data.shape[2], shape[0], shape[1]))))

        for z_axis in range(img_data.shape[2]):
            image = img_data[:,:,z_axis]

            data[length + z_axis,:,:], ori_pos = resize_image_2D(image, shape)
            original_positions.append(ori_pos)
        second_basal_data[num, ...], _ = resize_image_2D(img_data[..., -2], shape)
        patient_begin_idx.append(length)
    
    return data, second_basal_data, patient_begin_idx, info, original_positions, switches

def resize_image_2D(img, shape):
    xdim, ydim = shape
    img_final = np.zeros((xdim,ydim))
    
    xIndex = xdim - img.shape[0]
    yIndex = ydim - img.shape[1]
    if xIndex < 0:
        # Check whether it contains even numbers
        if (xIndex % 2 ) == 0:
            remove_rows_x = abs(xIndex)//2
            if remove_rows_x == 0:
                xBegin = 0
                xEnd = int(img.shape[0] - 1)
            else:
                xBegin = int(remove_rows_x - 1)
                xEnd = int(img.shape[0] - 1 - remove_rows_x)
        else:
            remove_rows_x = abs(xIndex)//2
            if remove_rows_x == 0:
                xBegin = 0
                xEnd = int(img.shape[0] - 1)
            else:
                xBegin = int(remove_rows_x - 1)
                # Now substract 2 because of the odd numbers
                xEnd = int(img.shape[0] - 2 - remove_rows_x)
    
    elif xIndex >= 0:
        # Check whether it contains even numbers
        if (xIndex % 2 ) == 0:
            add_rows_x = abs(xIndex)//2 
            if add_rows_x == 0:
                xBegin = 0
                xEnd = int(img_final.shape[0])
            else:
                xBegin = int(add_rows_x - 1)
                xEnd = int(img_final.shape[0] - 1 - add_rows_x)
        else:
            add_rows_x = abs(xIndex)//2
            if add_rows_x == 0:
                xBegin = 0
                xEnd = int(img_final.shape[0]-1)
            else:
                xBegin = int(add_rows_x - 1)
                # Now don't substract 1 because of the odd numbers
                xEnd = int(img_final.shape[0] - 2 - add_rows_x)
    
    
    if yIndex < 0:
        # Check whether it contains even numbers
        if (yIndex % 2 ) == 0:
            remove_rows_y = abs(yIndex)//2 
            if remove_rows_y == 0:
                yBegin = 0
                yEnd = int(img.shape[1] - 1)
            else:
                yBegin = int(remove_rows_y - 1)
                yEnd = int(img.shape[1] - 1 - remove_rows_y)
        else:
            remove_rows_y = abs(yIndex)//2
            if remove_rows_y == 0:
                yBegin = 0
                yEnd = int(img.shape[1] - 1)
            else:
                yBegin = int(remove_rows_y - 1)
                # Now don't substract 1 because of the odd numbers
                yEnd = int(img.shape[1] - 2 - remove_rows_y)
    
    elif yIndex >= 0:
        # Check whether it contains even numbers
        if (yIndex % 2 ) == 0:
            add_rows_y = abs(yIndex)//2
            if add_rows_y == 0:
                yBegin = 0
                yEnd = int(img_final.shape[1])
            else:
                yBegin = int(add_rows_y - 1)
                yEnd = int(img_final.shape[1] - 1 - add_rows_y)
        else:
            add_rows_y = abs(yIndex)//2
            if add_rows_y == 0:
                yBegin = 0
                yEnd = int(img_final.shape[1]-1)
            else:
                yBegin = int(add_rows_y - 1)
                # Now don't substract 1 because of the odd numbers
                yEnd = int(img_final.shape[1] - 2 - add_rows_y)
    
    original_positions = []
    if (xIndex < 0) and (yIndex < 0):
        img_final[:,:] = img[xBegin:xEnd,yBegin:yEnd]
    elif (xIndex < 0) and (yIndex >= 0):
        img_final[:,yBegin:yEnd] = img[xBegin:xEnd,:]
    elif (xIndex >= 0) and (yIndex < 0):
        img_final[xBegin:xEnd,:] = img[:,yBegin:yEnd]
    else:
        img_final[xBegin:xEnd,yBegin:yEnd] = img[:,:]

    original_positions.append(xBegin)
    original_positions.append(xEnd)
    original_positions.append(yBegin)
    original_positions.append(yEnd)
    original_positions.append(img.shape[0])
    original_positions.append(img.shape[1])
    original_positions.append(xIndex)
    original_positions.append(yIndex)
    return img_final, original_positions


def normalize_images(images):

    norm_data = np.zeros_like(images)
    for i in range(images.shape[0]):
        image = images[i,:,:]

        minMat = np.percentile(image[image>0], 5)
        maxMat = np.percentile(image[image>0], 95)
        image = (image - minMat)/ (maxMat - minMat)
        norm_data[i, ...] =  np.clip(image, 0,1)
    
    return norm_data


def denormalize_transformations(transform, x_bound, y_bound):
    old_transform = np.zeros((transform.shape))
    # Normalization of the translation between -1 and 1
    x_tr = transform[:,0]
    y_tr = transform[:,1]
    old_transform[:,0] = minmax_denorm(x_tr, x_bound)
    old_transform[:,1] = minmax_denorm(y_tr, y_bound)
    
    scale = transform[:,2:] + 1
    old_transform[:,2:] = scale
    return old_transform

def minmax_denorm(transform, bound):
    # Values between -1 and 1 back to original dataset
    old = (((transform + 1) / 2) * (bound - (-bound))) + (-bound)
    return old


def post_process_bb(image,transformation, average_size, image_size = 128):
    # Save positions
    positions = np.zeros((1, 4))
    box_positions = np.zeros((1, 4))
    # Calculate the center of the image
    img_center_x = int((image.shape[0]-1)/2)
    img_center_y = int((image.shape[1]-1)/2)
    
    # Calculate the positions of the bounding box
    x_center = img_center_x + transformation[0]
    y_center = img_center_y + transformation[1]
    x_top_left = int(x_center - (average_size * transformation[2]+1)//2)
    y_top_left = int(y_center + (average_size * transformation[3]+1)//2)
    x_bottom_right = int(x_center + (average_size * transformation[2]+1)//2)
    y_bottom_right = int(y_center - (average_size * transformation[3]+1)//2)

    # When the bounding box found is to large than the boundaries are chosen
    if (x_top_left < 0 or x_top_left >= image.shape[0]):
        x_top_left = 0
    if (y_top_left >= image.shape[1] or y_top_left < 0):
        y_top_left = image.shape[1] - 1
    if (x_bottom_right >= image.shape[0] or x_bottom_right < 0):
        x_bottom_right = image.shape[0] - 1
    if (y_bottom_right < 0 or y_bottom_right >= image.shape[1]):
        y_bottom_right = 0

    new_picture = image[x_top_left:x_bottom_right,y_bottom_right:y_top_left]
    box_positions[0,0] = x_top_left
    box_positions[0,1] = x_bottom_right
    box_positions[0,2] = y_bottom_right
    box_positions[0,3] = y_top_left
    
    # Calculate how much of the image is too big ore to small
    x_over = image_size - new_picture.shape[0]
    y_over = image_size - new_picture.shape[1]

    if x_over < 0:
        # Calculate the how much you need to add/subtract to the image
        x_correct1, x_correct2 = adding_subtracting(x_over)
        # The transformed image is too big thus you need to get a smaller one
        new_picture = new_picture[-x_correct1:new_picture.shape[0]+x_correct2,:]
        positions[0,0] = -x_correct1
        positions[0,1] = -x_correct2
    else:
        # Calculate the how much you need to add/subtract to the image
        x_correct1, x_correct2 = adding_subtracting(x_over)
        # The image is too small thus insert zero rows/columns
        new_picture = np.concatenate((np.zeros((x_correct1,new_picture.shape[1])),new_picture))
        new_picture = np.concatenate((new_picture,np.zeros((x_correct2,new_picture.shape[1]))))
        positions[0,0] = -x_correct1
        positions[0,1] = -x_correct2

    if y_over < 0:
        # Calculate the how much you need to add/subtract to the image
        y_correct1, y_correct2 = adding_subtracting(y_over)
        # The transformed image is too big thus you need to get a smaller one        
        new_picture = new_picture[:,-y_correct1:new_picture.shape[1]+y_correct2]
        positions[0,2] = -y_correct1
        positions[0,3] = -y_correct2

    else:
        # Calculate the how much you need to add/subtract to the image
        y_correct1, y_correct2 = adding_subtracting(y_over)
        # The image is too small thus insert zero rows/columns
        new_picture = np.concatenate((np.zeros((new_picture.shape[0], y_correct1)),new_picture,np.zeros((new_picture.shape[0],y_correct2))),axis = 1)
        positions[0,2] = -y_correct1
        positions[0,3] = -y_correct2

    if new_picture.shape[0] != image_size:
        print("Something went wrong", new_picture.shape)
    if new_picture.shape[1] != image_size:
        print("Something went wrong", new_picture.shape)

    return new_picture, positions, box_positions

def adding_subtracting(over):
    if over % 2 == 0:
        correct_1 = over/2
        correct_2 = over/2
    else:
        correct_1 = (over+1)//2
        correct_2 = over//2
    return int(correct_1), int(correct_2)


def save_test_myo(data, target_base):
    # target_imagesTs = (target_base + "/imagesTs")
    for i in range(data.shape[0]):
        image = data[i,:,:]
        if i < 10:
            output_image_file = (target_base + "/Myocardium_00{}_0000".format(i))  # do not specify a file ending! This will be done for you
        elif i < 100:
            output_image_file = (target_base + "/Myocardium_0{}_0000".format(i))  # do not specify a file ending! This will be done for you
        else:
            output_image_file = (target_base + "/Myocardium_{}_0000".format(i))  # do not specify a file ending! This will be done for you

        # this utility will convert 2d images that can be read by skimage.io.imread to nifti. You don't need to do anything.
        # if this throws an error for your images, please just look at the code for this function and adapt it to your needs
        convert_2d_image_to_nifti(image, output_image_file, is_seg=False)


# This function is made by the writers of the nnUnet
def convert_2d_image_to_nifti(img, output_filename_truncated: str, spacing=(999, 1, 1),
                              transform=None, is_seg: bool = False) -> None:
    """
    Reads an image (must be a format that it recognized by skimage.io.imread) and converts it into a series of niftis.
    The image can have an arbitrary number of input channels which will be exported separately (_0000.nii.gz,
    _0001.nii.gz, etc for images and only .nii.gz for seg).
    Spacing can be ignored most of the time.
    !!!2D images are often natural images which do not have a voxel spacing that could be used for resampling. These images
    must be resampled by you prior to converting them to nifti!!!
    Datasets converted with this utility can only be used with the 2d U-Net configuration of nnU-Net
    If Transform is not None it will be applied to the image after loading.
    Segmentations will be converted to np.uint32!
    :param is_seg:
    :param transform:
    :param input_filename:
    :param output_filename_truncated: do not use a file ending for this one! Example: output_name='./converted/image1'. This
    function will add the suffix (_0000) and file ending (.nii.gz) for you.
    :param spacing:
    :return:
    """

    if transform is not None:
        img = transform(img)

    if len(img.shape) == 2:  # 2d image with no color channels
        img = img[None, None]  # add dimensions
    else:
        assert len(img.shape) == 3, "image should be 3d with color channel last but has shape %s" % str(img.shape)
        # we assume that the color channel is the last dimension. Transpose it to be in first
        img = img.transpose((2, 0, 1))
        # add third dimension
        img = img[:, None]

    # image is now (c, x, x, z) where x=1 since it's 2d
    if is_seg:
        assert img.shape[0] == 1, 'segmentations can only have one color channel, not sure what happened here'

    for j, i in enumerate(img):
        if is_seg:
            i = i.astype(np.uint32)

        itk_img = sitk.GetImageFromArray(i)
        itk_img.SetSpacing(list(spacing)[::-1])
        if not is_seg:
            sitk.WriteImage(itk_img, output_filename_truncated + ".nii.gz")
        else:
            sitk.WriteImage(itk_img, output_filename_truncated + ".nii.gz")


def get_largestCC(segmentation):
    labels = measure.label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def opening_scar_one_class(predicted):
    scar = np.copy(predicted[:,:])    
    biggest_scar = area_opening(scar,10)    
    new_image = biggest_scar

    return new_image

def read_predicted_labels(data, name, link, task):
    # Create list of the predicted labels
    labels = np.zeros((data.shape))
    
    for i in range(data.shape[0]):
        if i < 10:
            pred_seg_file0 = (link + "/{}_00{}.nii.gz".format(name,i))  # do not specify a file ending! This will be done for you
        elif i < 100:
            pred_seg_file0 = (link + "/{}_0{}.nii.gz".format(name,i))  # do not specify a file ending! This will be done for you
        else:
            pred_seg_file0 = (link + "/{}_{}.nii.gz".format(name,i))  # do not specify a file ending! This will be done for you
        
        # Read in your data
        predict = nib.load(pred_seg_file0)
        predict_data = predict.get_data()
        
        # Because the nnUnet rotates the images during saving
        predict_data = predict_data[:,:,0]
        predict_data = cv2.flip(predict_data, 1)
        predict_data = cv2.rotate(predict_data, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if task == "myocardium":
            if np.max(predict_data) != 0:
                predict_data = get_largestCC(predict_data)
        if task == "scar":
            predict_data = opening_scar_one_class(predict_data)
            
        labels[i,:,:] = predict_data
    return labels



def myocardium_seg_check(images, myocardium_predictions, shift_predictions, patients, average_box, im_shape):
    incorrect_myocardium = []
    augmented_myocardium = []
    save_shifts_x = []
    save_shifts_y = []
    box_pos_shifts = []
    box_scale_shifts = []
    new_pos = 0
    patient = 0
    
    for i in range(myocardium_predictions.shape[0]):
        prediction = myocardium_predictions[i,:,:]
        # If the area of the predicted myocardium is below 90 then make 10
        # new pictures with shifted bounding boxes
        if i == int(patients[patient+1]):
            patient += 1
        
        if prediction.sum() < 90 or np.amax(measure.label(get_largestCC(prediction),connectivity=2,background=-1)) != 3:
            # Append which slice it is
            incorrect_myocardium.append(i)
            
            if augmented_myocardium == []:
                augmented_myocardium = np.zeros((10,myocardium_predictions.shape[1], myocardium_predictions.shape[2]))
            else:
                augmented_myocardium = np.concatenate((augmented_myocardium, np.zeros((10,myocardium_predictions.shape[1], myocardium_predictions.shape[2])))) 
            
            original_shift = np.copy(shift_predictions)
            
            x, y = np.where(prediction == 1)
            
            if x != []:
                distance_x = int(max(x)-min(x))
                center_x = int(min(x)+int(distance_x/2))
        
                distance_y = int(max(y)-min(y))
                center_y = int(min(y)+int(distance_y/2))
                
                center_x_image = int(myocardium_predictions.shape[1]/2)
                center_y_image = int(myocardium_predictions.shape[2]/2)
   
                diff_center_x = center_x_image - center_x
                diff_center_y = center_y_image - center_y
    
                if diff_center_x <= 0:
                    x_shifts = [0, diff_center_x, diff_center_x-8, diff_center_x-4, diff_center_x-8, diff_center_x-4, diff_center_x, diff_center_x, diff_center_x-8, diff_center_x-4]
                if diff_center_y <= 0:
                    y_shifts = [0, diff_center_y, diff_center_y-8, diff_center_y-4, diff_center_y, diff_center_y, diff_center_y-8, diff_center_y-4, diff_center_y-4, diff_center_y-8]
                if diff_center_x > 0:
                    x_shifts = [0, diff_center_x, diff_center_x+8, diff_center_x+4, diff_center_x+8, diff_center_x+4, diff_center_x, diff_center_x, diff_center_x+8, diff_center_x+4]
                if diff_center_y > 0:
                    y_shifts = [0, diff_center_y, diff_center_y+8, diff_center_y+4, diff_center_y, diff_center_y, diff_center_y+8, diff_center_y+4, diff_center_y+4, diff_center_y+8]

                save_shifts_x.append(np.copy(x_shifts))
                save_shifts_y.append(np.copy(y_shifts))


                for t in range(10):
                    original_shift[patient,0] = shift_predictions[patient,0]+int(x_shifts[t])
                    original_shift[patient,1] = shift_predictions[patient,1]+int(y_shifts[t])
                        
                    augmented_myocardium[new_pos,:,:], box_pos, box_scale = post_process_bb(images[i,:,:], original_shift[patient,:], average_box, im_shape)
                    box_pos_shifts.append(box_pos)
                    box_scale_shifts.append(box_scale)
                    new_pos += 1
    return augmented_myocardium, incorrect_myocardium, save_shifts_x, save_shifts_y, box_pos_shifts, box_scale_shifts
  

def maj_vote_augmentations(double_checked_myocardium, myocardium_labels, double_check_myocardium_labels, incorrect_myocardium, shifts_x, shifts_y):
    incorrect = 0
    for i in range(0,double_checked_myocardium.shape[0], 10):
        labels = double_check_myocardium_labels[i:i+10,:,:]

        summed_myo_mask = np.zeros((labels.shape[1], labels.shape[2]))
        for z in range(labels.shape[0]):
            
            label = labels[int(z),:,:]
            new_label = np.zeros((label.shape))
            if int(shifts_x[incorrect][z]) >= 0 and int(shifts_y[incorrect][z]) >= 0:    
                new_label[int(shifts_x[incorrect][z]):,int(shifts_y[incorrect][z]):] = label[:int(label.shape[0]-int(shifts_x[incorrect][z])),:int(label.shape[0]-int(shifts_y[incorrect][z]))]
            elif int(shifts_x[incorrect][z]) < 0 and int(shifts_y[incorrect][z]) < 0:    
                new_label[:int(label.shape[0]+int(shifts_x[incorrect][z])),:int(label.shape[0]+int(shifts_y[incorrect][z]))] = label[int(-shifts_x[incorrect][z]):,int(-shifts_y[incorrect][z]):]
            elif int(shifts_x[incorrect][z]) >= 0 and int(shifts_y[incorrect][z]) < 0:    
                new_label[int(shifts_x[incorrect][z]):,:int(label.shape[0]+int(shifts_y[incorrect][z]))] = label[:int(label.shape[0]-int(shifts_x[incorrect][z])),int(-shifts_y[incorrect][z]):]
            elif int(shifts_x[incorrect][z]) < 0 and int(shifts_y[incorrect][z]) >= 0:  
                new_label[:int(label.shape[0]+int(shifts_x[incorrect][z])),int(shifts_y[incorrect][z]):] = label[int(-shifts_x[incorrect][z]):,:int(label.shape[0]-int(shifts_y[incorrect][z]))]
            summed_myo_mask += new_label

        k = 5
        not_finished = True
        while k>=1 and not_finished:
            proposed_myo = (summed_myo_mask > k)*1.
            if proposed_myo.sum() != 0:
                if np.amax(measure.label(get_largestCC(proposed_myo),connectivity=2,background=-1)) == 3 or k == 1:
                    not_finished = False
                    myocardium = proposed_myo
            k -= 1

        slice_nm = int(incorrect_myocardium[incorrect])
        myocardium_labels[slice_nm,:,:] = myocardium
        incorrect += 1
    return myocardium_labels


def cut_images(data, myos):        
    new_data = np.zeros((data.shape[0],64,64))
    new_myos = np.zeros((myos.shape[0],64,64))
    position = np.zeros((myos.shape[0], 4))
    
    for i in range(data.shape[0]):
        image = data[i,:,:]
        myo = myos[i,:,:]
        
        x, y = np.where(myo==1)
        if x != []:
            distance_x = int(max(x)-min(x))
            center_x = int(min(x)+int(distance_x/2))
    
            distance_y = int(max(y)-min(y))
            center_y = int(min(y)+int(distance_y/2))
            
            if (center_x - 32) > 0 and (center_y - 32) > 0:
                new_data[i,:,:] = image[center_x-32:center_x+32,center_y-32:center_y+32]
                new_myos[i,:,:] = myo[center_x-32:center_x+32,center_y-32:center_y+32]
                position[i,0] = center_x-32
                position[i,1] = center_x+32
                position[i,2] = center_y-32
                position[i,3] = center_y+32
                
            else:
                if center_x - 32 < 0:
                    new_data[i,:,:] = image[0:64,center_y-32:center_y+32]
                    new_myos[i,:,:] = myo[0:64,center_y-32:center_y+32]
                    position[i,0] = 0
                    position[i,1] = 64
                    position[i,2] = center_y-32
                    position[i,3] = center_y+32
                else:
                    new_data[i,:,:] = image[center_x-32:center_x+32,0:64]
                    new_myos[i,:,:] = myo[center_x-32:center_x+32,0:64]
                    position[i,0] = center_x-32
                    position[i,1] = center_x+32
                    position[i,2] = 0
                    position[i,3] = 64
        else:
            new_data[i,:,:] = image[32:96,32:96]
            new_myos[i,:,:] = myo[32:96,32:96]
            position[i,0] = 32
            position[i,1] = 96
            position[i,2] = 32
            position[i,3] = 96
            
    return new_data, new_myos, position



def get_cavity(myocardium_labels):
    # Turn the complete myocardium to 1 value and make use of convex hull to
    # include the cavity
    myo = np.copy(myocardium_labels)
    LV = convex_hull_image(myo)
    
    # Now only find the cavity
    cavity = LV - myo
    cavity = get_largestCC(cavity)
    return cavity

def change_image(image, LV, cavity, option = []):
    if option == []:
        image = image * LV
        # Add a fixed value to the intensities so there a no zero values
        LV[LV==1] = 0.1
        image = image + LV
        
        cavity[cavity>1] = 0
        x, y = np.where(cavity==1)
        for i in range(x.shape[0]):
            image[x[i],y[i]] = 2.5                          
    return image

def scar_normalize(data, myo):
    new_data = np.zeros((data.shape[0], data.shape[1], data.shape[2], 1))

    for i in range(data.shape[0]):
        image = np.copy(data[i,:,:])
        cavity_labels = get_cavity(myo[i,:,:])
        LV = np.copy(myo[i,:,:]) + np.copy(cavity_labels)
        
        # Create a masked version and brighten the cavity values
        image = change_image(image, LV, cavity_labels)
        new_data[i,:,:,0] = image                              
    return new_data

def save_test_scar(data, target_base):

    for i in range(data.shape[0]):
        image = data[i,:,:]
        if i < 10:
            output_image_file = (target_base + "/ScaringMyocardium_00{}_0000".format(i))  # do not specify a file ending! This will be done for you
        elif i < 100:
            output_image_file = (target_base + "/ScaringMyocardium_0{}_0000".format(i))  # do not specify a file ending! This will be done for you
        else:
            output_image_file = (target_base + "/ScaringMyocardium_{}_0000".format(i))  # do not specify a file ending! This will be done for you

        # this utility will convert 2d images that can be read by skimage.io.imread to nifti. You don't need to do anything.
        # if this throws an error for your images, please just look at the code for this function and adapt it to your needs
        convert_2d_image_to_nifti(image, output_image_file, is_seg=False)

def scar_post_processing(data,myocardium,scar,patients, voxel_info):

    # Make a list for the volumes and scar percentages per-slice and per-subject
    scar_percentage_slice = np.zeros((scar.shape[0],1))
    scar_percentage_subject = np.zeros((len(patients)-1,1))
    scar_volume_slice = np.zeros((scar.shape[0],1))
    scar_volume_subject = np.zeros((len(patients)-1,1))
    myo_volume_slice = np.zeros((scar.shape[0],1))
    myo_volume_subject = np.zeros((len(patients)-1,1))
   
    position = 0
    # Now look on patient level
    for pat in range(len(patients)-1):

        myo_labels = myocardium[int(patients[int(pat)]):int(patients[int(pat+1)]),:,:]
        scar_labels = scar[int(patients[int(pat)]):int(patients[int(pat+1)]),:,:]

        # Calculate the percentage scar per patient
        sum_myo = myo_labels.sum()
        sum_scar = scar_labels.sum()
        patient_scar_percentage = (sum_scar/sum_myo)*100
        
        # If the scar percentage is below 3% then take away the scar prediction
        if patient_scar_percentage <= 3:
            scar_labels = np.zeros((scar_labels.shape))
            scar[int(patients[int(pat)]):int(patients[int(pat+1)]),:,:] = scar_labels

                
        # Append per-subject scar percentage
        sum_scar = scar_labels.sum()
        scar_percentage_subject[pat,0] = (sum_scar/sum_myo)*100
        
        # Calculate the scar volume
        voxel_info_pat = voxel_info[pat]
        voxel_dim = voxel_info_pat[1][1]
        
        scar_volume = 0     
        myo_volume = 0
        for i in range(myo_labels.shape[0]):
            myo_slice = myo_labels[i,:,:]
            scar_slice = scar_labels[i,:,:]
            
            # Append the per-slice scar percentage
            scar_percentage_slice[position,0] = (scar_labels[i,:,:].sum()/myo_labels[i,:,:].sum())*100
            
            volume_myo = myo_slice.sum()
            if i == myo_labels.shape[0]-1:
                # Calculate the volume per-slice, however add them to the previous slice to get the value per-subject
                myo_volume += volume_myo*voxel_dim*voxel_dim*18                
                # Take 18, because you need to take the slice distance and thickness into account
                myo_volume_slice[position,0] = volume_myo*voxel_dim*voxel_dim*18
            else:
                # Calculate the volume per-slice, however add them to the previous slice to get the value per-subject                
                myo_volume += volume_myo*voxel_dim*voxel_dim*18
                # Take 18, because you need to take the slice distance and thickness into account
                myo_volume_slice[position,0] = volume_myo*voxel_dim*voxel_dim*18
            
            # If you find scar calculate the scar volume
            if 1 in scar_slice:
                volume = scar_slice.sum()            
                if i != myo_labels.shape[0]-1:
                    # If you find scar in the next slice than include the thickness and distance
                    # if you don't find it then only use the thickness
                    if 1 in scar[i+1,:,:]:
                        scar_volume += volume*voxel_dim*voxel_dim*18
                        scar_volume_slice[position,0] = volume*voxel_dim*voxel_dim*18
                    else:
                        scar_volume += volume*voxel_dim*voxel_dim*8
                        scar_volume_slice[position,0] = volume*voxel_dim*voxel_dim*8
                else:
                    scar_volume += volume*voxel_dim*voxel_dim*8
                    scar_volume_slice[position,0] = volume*voxel_dim*voxel_dim*8
            position += 1
            
        scar_volume_subject[pat,0] = scar_volume      
        myo_volume_subject[pat,0] = myo_volume      
    
    return scar, scar_percentage_slice, scar_percentage_subject, scar_volume_slice, scar_volume_subject, myo_volume_slice, myo_volume_subject   



def uncut_64_64(label_predictions, uncut_position):
    new_labels = np.zeros((label_predictions.shape[0],128,128))
    
    for i in range(label_predictions.shape[0]):
        label = label_predictions[i,:,:]
        new_labels[i,int(uncut_position[i,0]):int(uncut_position[i,1]),int(uncut_position[i,2]):int(uncut_position[i,3])] = label

    return new_labels

def save_images(labels, filename_lab, affine, head1):
    lab = labels.astype('uint8')   
    lab1 = nib.Nifti1Image(lab, affine = affine, header = head1)    
    nib.save(lab1, filename_lab)

def return_image_size_and_save(label_predictions_myo, label_predictions_scar, uncut_position, box_positions, box_scales, original_image_position, patients, direct, switchs):
    
    im_files = sorted(glob(os.path.join(direct, "*", "Images", "*.nii.gz")))

    label_predictions = np.zeros((label_predictions_scar.shape))
    for z in range(label_predictions_myo.shape[0]):
        label_predictions_cavity = get_cavity(label_predictions_myo[z,:,:])
        
        x, y = np.where(label_predictions_cavity==1)
        for i in range(x.shape[0]):
            label_predictions[z, x[i],y[i]] = 1  
        x, y = np.where(label_predictions_myo[z,:,:]==1)
        for i in range(x.shape[0]):
            label_predictions[z, x[i],y[i]] = 2                          
        x, y = np.where(label_predictions_scar[z,:,:]==1)
        for i in range(x.shape[0]):
            label_predictions[z, x[i],y[i]] = 3                          
    
    # Get the 128x128 images back
    resized_labels = uncut_64_64(label_predictions,uncut_position)
    
    resized_labels_256_256 = np.zeros((resized_labels.shape[0], 256, 256))
    for z in range(resized_labels.shape[0]):
        bigger_image = np.zeros((256,256))
        
        if box_positions[z][0][0] > 0:
            box_image = np.concatenate((np.zeros((int(box_positions[z][0][0]),resized_labels.shape[1])),resized_labels[z,:,:]))
            box_image = np.concatenate((box_image,np.zeros((int(box_positions[z][0][1]),resized_labels.shape[1]))))

        elif box_positions[z][0][0] < 0:
            box_image = resized_labels[z,-int(box_positions[z][0][0]):int(resized_labels.shape[1]+int(box_positions[z][0][0])),:]
        elif box_positions[z][0][0]==0 and box_positions[z][0][1] == 0:
            box_image = resized_labels[z,:,:]
        if box_positions[z][0][2] > 0:
            box_image = np.concatenate((np.zeros((box_image.shape[0], int(box_positions[z][0][2]))),box_image,np.zeros((box_image.shape[0], int(box_positions[z][0][3])))),axis = 1)
        elif box_positions[z][0][2] < 0:
            box_image = box_image[:,-int(box_positions[z][0][2]):int(resized_labels.shape[1]+int(box_positions[z][0][3]))]

        bigger_image[int(box_scales[z][0][0]):int(box_scales[z][0][1]), int(box_scales[z][0][2]):int(box_scales[z][0][3])] = box_image
        resized_labels_256_256[z,:,:] = bigger_image
    
    
    patient_slice = 0    
    for pat in range(len(patients)-1):
        predict_pat = resized_labels_256_256[int(patients[int(pat)]):int(patients[int(pat+1)]),:,:]
        original_label = np.zeros((int(original_image_position[patient_slice][4]), int(original_image_position[patient_slice][5]), predict_pat.shape[0]))
        for z_slice in range(predict_pat.shape[0]):
            if (original_image_position[patient_slice][6] < 0) and (original_image_position[patient_slice][7] < 0):
                original_label[int(original_image_position[patient_slice][0]):int(original_image_position[patient_slice][1]),int(original_image_position[patient_slice][2]):int(original_image_position[patient_slice][3]),z_slice] = predict_pat[z_slice,:,:]
            elif (original_image_position[patient_slice][6] < 0) and (original_image_position[patient_slice][7] >= 0):
                original_label[int(original_image_position[patient_slice][0]):int(original_image_position[patient_slice][1]), :, z_slice] = predict_pat[z_slice,:,int(original_image_position[patient_slice][2]):int(original_image_position[patient_slice][3])]
            elif (original_image_position[patient_slice][6] >= 0) and (original_image_position[patient_slice][7] < 0):
                original_label[:,int(original_image_position[patient_slice][2]):int(original_image_position[patient_slice][3]), z_slice] = predict_pat[z_slice,int(original_image_position[patient_slice][0]):int(original_image_position[patient_slice][1]),:]
            else:
                original_label[:,:, z_slice] = predict_pat[z_slice,int(original_image_position[patient_slice][0]):int(original_image_position[patient_slice][1]),int(original_image_position[patient_slice][2]):int(original_image_position[patient_slice][3])]
            
            patient_slice += 1

        img = nib.load(im_files[pat])
        header = img.header
        affine = img.affine

        # Switch the axis of the picture, so the pictures are in the same direction 
        if switchs[pat]==1:
            original_label = np.swapaxes(original_label,0,1)
                
        filename_lab = im_files[pat]
        pat_direct = os.path.split(os.path.split(filename_lab)[0])[0]
        pat_direct = os.path.join(pat_direct, "Contours")
        if not os.path.isdir(pat_direct):
            os.mkdir(pat_direct)

        label_direct = os.path.join(pat_direct, os.path.split(filename_lab)[1])
        
        save_images(original_label, label_direct, affine, header)
        
    
    return resized_labels

