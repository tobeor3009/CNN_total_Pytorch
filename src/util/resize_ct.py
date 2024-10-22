import numpy as np
import SimpleITK as sitk
import os
import time
import uuid

def resize_dicom_series(image, resize_factor_list):

    dimension = image.GetDimension()

    reference_physical_size = np.zeros(image.GetDimension())
    reference_physical_size[:] = [(sz-1)*spc if sz*spc>mx  else mx for sz,spc,mx in zip(image.GetSize(), 
                                                                                        image.GetSpacing(), 
                                                                                        reference_physical_size)]
    reference_origin = image.GetOrigin()
    reference_direction = image.GetDirection()

    reference_size = [round(sz * resize_factor) for sz, resize_factor in zip(image.GetSize(), resize_factor_list)] 
    reference_spacing = [phys_sz/(sz-1) for sz,phys_sz in zip(reference_size, reference_physical_size)]

    reference_image = sitk.Image(reference_size, image.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))

    transform = sitk.AffineTransform(dimension)
#     transform.SetMatrix(image.GetDirection())
    transform.SetMatrix((1, 0, 0, 0, 1, 0, 0, 0, 1))
    transform.SetTranslation(np.array(image.GetOrigin()) - reference_origin)

    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(image.TransformContinuousIndexToPhysicalPoint(np.array(image.GetSize())/2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.Transform(transform)
    centered_transform = sitk.CompositeTransform([centered_transform, centering_transform])
    min_value = float(np.min(sitk.GetArrayFromImage(image)))
    
    # source_image, refrence, transform, interpolation method, default value
    new_img = sitk.Resample(image, reference_image, centered_transform, sitk.sitkBSpline, min_value)
    # Ensure the image is in grayscale (1-channel) if not already
    if new_img.GetNumberOfComponentsPerPixel() > 1:
        # Convert multi-channel image to grayscale by averaging channels
        array = sitk.GetArrayFromImage(new_img)
        grayscale_array = np.mean(array, axis=-1)  # Averaging the color channels
        new_img = sitk.GetImageFromArray(grayscale_array)
        new_img.CopyInformation(reference_image)  # Retain the spatial metadat
    new_img = sitk.Cast(new_img, sitk.sitkInt16)
    return new_img

def write_series_to_path(target_image, original_sample_path, target_path, slice_thickness):
    tags_to_copy = ["0010|0010", # Patient Name
                    "0010|0020", # Patient ID
                    "0010|0030", # Patient Birth Date
                    "0020|000D", # Study Instance UID, for machine consumption
                    "0020|0010", # Study ID, for human consumption
                    "0008|0020", # Study Date
                    "0008|0030", # Study Time
                    "0008|0050", # Accession Number
                    "0008|0060"  # Modality
    ]

    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")
    direction = target_image.GetDirection()
    
    try:
        series_tag_value = reader.GetMetaData(0,"0008|103e")
    except RuntimeError:
        series_tag_value = "tag_None"
    
    original_image = sitk.ReadImage(original_sample_path)
    original_key_tuple = original_image.GetMetaDataKeys()
    original_tag_values = [(tag, original_image.GetMetaData(tag)) for tag in original_key_tuple]
    series_tag_values = [(k, original_image.GetMetaData(k)) for k in tags_to_copy if original_image.HasMetaDataKey(k)] + \
                     [("0008|0031",modification_time), # Series Time
                      ("0008|0021",modification_date), # Series Date
                      #("0008|0008","DERIVED\\SECONDARY"), # Image Type
                      #("0020|000e", "1.2.826.0.1.3680043.2.1125."+modification_date+".1"+modification_time), # Series Instance UID
                      ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],# Image Orientation (Patient)
                                                        direction[1],direction[4],direction[7])))),
                      ("0008|103e", series_tag_value + " Processed-SimpleITK")]

    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    
    os.makedirs(target_path, exist_ok=True)
    target_image_depth = target_image.GetDepth()
    
    series_instance_uid = os.path.basename(target_path)
    
    
    for index in range(target_image_depth):
        image_slice = target_image[:, :, index]
        # Tags shared by the series.
        instance_number = index + 1
        sop_uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{series_instance_uid}.{instance_number}"))
        for tag, value in original_tag_values:
            try:
                image_slice.SetMetaData(tag, value)
            except:
                continue
        for tag, value in series_tag_values:
            image_slice.SetMetaData(tag, value)
        # Setting the type to CT preserves the slice location.
        image_slice.SetMetaData("0008|0060", "CT")  # set the type to CT so the thickness is carried over
        # Slice specific tags.
        image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d")) # Instance Creation Date
        image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S")) # Instance Creation Time
        image_slice.SetMetaData("0008|0018", sop_uid) # SOP UID
        
        image_slice.SetMetaData("0020|0032", '\\'.join(map(str, target_image.TransformIndexToPhysicalPoint((0,0,index))))) # Image Position (Patient)
        image_slice.SetMetaData("0020|0013", str(target_image_depth - index)) # Instance Number
        image_slice.SetMetaData("0018|0050", str(slice_thickness)) # set series slice thickness
        image_slice.SetMetaData("0020|000E", series_instance_uid)
        image_slice.SetMetaData("0020|000D", series_instance_uid)
        
        writer.SetFileName(f'{target_path}/{target_image_depth - index:04}.dcm')
        writer.Execute(image_slice)