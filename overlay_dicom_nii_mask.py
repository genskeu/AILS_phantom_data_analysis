# function to overlay a nifti mask on a dicom image
# save image as png
# input: dicom image, nifti mask, output path, z slice, mask color, mask opacity
# output: overlay image

import os
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import matplotlib.pyplot as plt

def overlay_dicom_nii_mask(vol_path, nii_path, output_path, z_slice=None, color_scale=None, mask_opacity=0.5, window_width = 350, window_ceter=50, 
                           bb_cutoff_dicom=-500, show=False, bb_mask=False, bb_mask_padding=0, title="", mask_labels_to_ignore=None,
                           delte_px = None, flipr=False, flipud=False, show_image=True):
    nii_data = sitk.ReadImage(nii_path)
    nii_data = sitk.GetArrayFromImage(nii_data)
    if mask_labels_to_ignore is not None:
        for label in mask_labels_to_ignore:
            nii_data[nii_data == label] = 0
    if z_slice is None:
        bb = np.argwhere(nii_data > 0)
        min_z = np.min(bb[:, 0])
        max_z = np.max(bb[:, 0])
        z_slice = (min_z + max_z) // 2
        print(f'z_slice not provided, using {z_slice}')
    if os.path.isdir(vol_path):
        dicom_path = vol_path
        dicom_series_reader = sitk.ImageSeriesReader()
        dicom_series_ids = dicom_series_reader.GetGDCMSeriesIDs(dicom_path)
        dicom_series_file_names = dicom_series_reader.GetGDCMSeriesFileNames(dicom_path, dicom_series_ids[0])
        dicom_data = sitk.ReadImage(dicom_series_file_names)
        dicom_data = sitk.GetArrayFromImage(dicom_data)
    elif os.path.isfile(vol_path) and vol_path.endswith('.nii.gz'):
        dicom_data = sitk.ReadImage(vol_path)
        dicom_data = sitk.GetArrayFromImage(dicom_data)
    else:
        raise ValueError('Invalid vol path')

    mask = nii_data[z_slice]
    dicom = dicom_data[z_slice]

    if flipr:
        dicom = np.fliplr(dicom)
        mask = np.fliplr(mask)
    
    if flipud:
        dicom = np.flipud(dicom)
        mask = np.flipud(mask)

    if bb_cutoff_dicom is not None:
        # get the bounding box of the dicom image
        bb = np.argwhere(dicom > bb_cutoff_dicom)
        min_x = np.min(bb[:, 0])
        max_x = np.max(bb[:, 0]) + 1
        min_y = np.min(bb[:, 1])
        max_y = np.max(bb[:, 1]) + 1

        dicom = dicom[min_x:max_x, min_y:max_y]
        mask = mask[min_x:max_x, min_y:max_y]

    if bb_mask:
        bb = np.argwhere(mask > 0)
        min_x = np.min(bb[:, 0])
        min_x = max(0, min_x - bb_mask_padding)
        max_x = np.max(bb[:, 0]) + 1
        max_x = min(mask.shape[0], max_x + bb_mask_padding)
        min_y = np.min(bb[:, 1])
        min_y = max(0, min_y - bb_mask_padding)
        max_y = np.max(bb[:, 1]) + 1
        max_y = min(mask.shape[1], max_y + bb_mask_padding)
        
        mask = mask[min_x:max_x, min_y:max_y]
        dicom = dicom[min_x:max_x, min_y:max_y]

    mask = np.ma.masked_where(mask == 0, mask)
    if delte_px:
        dicom = dicom[delte_px["top"]:-delte_px["bottom"], delte_px["left"]:-delte_px["right"]]
        mask = mask[delte_px["top"]:-delte_px["bottom"], delte_px["left"]:-delte_px["right"]]

    if show_image:
        plt.imshow(dicom, cmap='gray', vmin=window_ceter - window_width/2, vmax=window_ceter + window_width/2)
    plt.imshow(mask, cmap=color_scale, alpha=mask_opacity)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    if show:
        plt.show()
    plt.close()
    

def get_patient_example_images(vol_path, nii_path, output_path, z_slice=None, color_scale=None, mask_opacity=0.5, window_width = 350, window_ceter=50, 
                           bb_cutoff_dicom=-500, show=False, bb_mask=False, bb_mask_padding=0, title="", mask_labels_to_ignore=None,
                           delte_px = None, flipr=False, flipud=False):
    
    output_path_with_mask = f"{output_path}_with_mask.png"
    overlay_dicom_nii_mask(vol_path, nii_path, output_path_with_mask, z_slice, color_scale, mask_opacity, window_width, window_ceter,
                            bb_cutoff_dicom, show, bb_mask, bb_mask_padding, title, mask_labels_to_ignore, delte_px, flipr, flipud)
    output_path_without_mask = f"{output_path}_without_mask.png"
    overlay_dicom_nii_mask(vol_path, nii_path, output_path_without_mask, z_slice, color_scale, 0, window_width, window_ceter,
                            bb_cutoff_dicom, show, bb_mask, bb_mask_padding, title, mask_labels_to_ignore, delte_px, flipr, flipud)