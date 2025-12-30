import numpy as np
import SimpleITK as sitk
import os
import time
import uuid
import pydicom
from pydicom.dataelem import DataElement

def resize_dicom_series(image, resize_factor_list, interpolation_method="BSpline"):
    """
    Docstring for resize_dicom_series
    
    :param image: sitk Image Object with Patient CT info
    :param resize_factor_list: resize factor list for (X, Y, Z) axes, len 3 tuple or list
    0 < resize_factor <= 1.0 : downsampling
    resize_factor > 1.0 : upsampling
    :param interpolation_method: interpolation method for resampling, "nearest" or "BSpline"
    :return: new_img: sitk Image Object after resampling
    """
    interpolation_method_list = ["nearest", "BSpline"]
    assert interpolation_method in interpolation_method_list, f"Check interpolation_method. Available methods: {interpolation_method_list}"

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
    if interpolation_method == "nearest":
        new_img = sitk.Resample(image, reference_image, centered_transform, sitk.sitkNearestNeighbor, min_value)
    elif interpolation_method == "BSpline":
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
    reader = sitk.ImageSeriesReader()
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

############################ resize single dcm_file ############################
def read_dcm_path(dcm_path):
    return pydicom.dcmread(dcm_path, force=True)

def fix_dicom_spacing(dcm_path, fixed_dcm_path):
    # DICOM 파일 읽기
    dcm_obj = read_dcm_path(dcm_path)
    dcm_obj.SliceThickness = 1.0  # z축 spacing 값을 1.0으로 수정
    # Spacing Between Slices 를 1로 수정, 없으면 SimpleITK로 안 읽어짐
    dcm_obj[(0x0018, 0x0088)] = DataElement((0x0018, 0x0088), 'DS', '1')
    dcm_obj.save_as(fixed_dcm_path)

def _resize_dicom_slice(dcm_path, target_size, interpolation_method="spline"):
    
    interpolation_method_list = ["nearest", "spline"]
    assert interpolation_method in interpolation_method_list, f"Check interpolation_method. Available methods: {interpolation_method_list}"
    
    # SimpleITK의 Interpolator 설정
    if interpolation_method == "nearest":
        interpolation_method = sitk.sitkNearestNeighbor
    elif interpolation_method == "spline":
        interpolation_method = sitk.sitkBSpline
    
    # 이미지 읽기
    image = sitk.ReadImage(dcm_path)
    origin_size = np.array(image.GetSize())
    
    # 입력 이미지가 정사각형이 아닌 경우 거부
    assert target_size[0] == target_size[1], f"Input DICOM slice is not square. Size: {origin_size}"
    
    # 목표 크기 설정
    target_size = np.array(target_size)
    
    # 이미지의 최소값을 패딩 값으로 설정
    min_value = float(np.min(sitk.GetArrayFromImage(image)))
    
    # x, y 축에만 패딩 추가 (정사각형 유지)
    max_dim = max(origin_size[0], origin_size[1])
    x_padding = (max_dim - origin_size[0])  # x 축 패딩 필요 양
    y_padding = (max_dim - origin_size[1])  # y 축 패딩 필요 양

    half_x_padding = x_padding // 2
    hald_y_padding = y_padding // 2
    
    left_padding = (half_x_padding, hald_y_padding, 0)
    right_padding = (x_padding - half_x_padding, y_padding - hald_y_padding, 0) 
    
    left_padding = tuple(int(pad_num) for pad_num in left_padding)
    right_padding = tuple(int(pad_num) for pad_num in right_padding)
    padded_image = sitk.ConstantPad(image, left_padding, right_padding, min_value)

    # 패딩된 이미지 크기와 스페이싱 계산
    origin_size = np.array(padded_image.GetSize())
    resize_factor_list = tuple(target_size / origin_size)
    
    reference_physical_size = np.zeros(padded_image.GetDimension())
    reference_physical_size = [(sz - 1) * spc if sz * spc > mx else mx 
                                   for sz, spc, mx in zip(padded_image.GetSize(), 
                                                          padded_image.GetSpacing(), 
                                                          reference_physical_size)]
    reference_origin = padded_image.GetOrigin()
    reference_direction = padded_image.GetDirection()
    
    # 리사이즈된 이미지 크기와 스페이싱 계산
    reference_size = [round(sz * resize_factor) for sz, resize_factor in zip(padded_image.GetSize(), resize_factor_list)]
    reference_spacing = [
        phys_sz / (sz - 1) if sz > 1 else spc 
        for sz, phys_sz, spc in zip(reference_size, reference_physical_size, padded_image.GetSpacing())
    ]
    
    # 새로운 reference 이미지 생성
    reference_image = sitk.Image(reference_size, padded_image.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    # 중심점 계산
    reference_center = np.array(reference_image.GetSize()) / 2.0
    reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(reference_center))

    transform = sitk.AffineTransform(padded_image.GetDimension())
    transform.SetMatrix(reference_direction)
    transform.SetTranslation(np.array(padded_image.GetOrigin()) - reference_origin)

    centering_transform = sitk.TranslationTransform(padded_image.GetDimension())
    img_center = np.array(padded_image.TransformContinuousIndexToPhysicalPoint(np.array(padded_image.GetSize()) / 2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))

    # Transform을 합성
    centered_transform = sitk.Transform(transform)
    centered_transform = sitk.CompositeTransform([centered_transform, centering_transform])

    # 이미지의 최소값으로 기본 채우기 값 설정
    min_value = float(np.min(sitk.GetArrayFromImage(padded_image)))

    # Resample 함수 호출하여 이미지 리사이즈
    new_img = sitk.Resample(padded_image, reference_image, centered_transform, interpolation_method, min_value)

    # Grayscale 변환 (다중 채널일 경우만)
    if new_img.GetNumberOfComponentsPerPixel() > 1:
        array = sitk.GetArrayFromImage(new_img)
        grayscale_array = np.mean(array, axis=-1)
        new_img = sitk.GetImageFromArray(grayscale_array)
        new_img.CopyInformation(reference_image)

    # 이미지 타입 캐스팅
    new_img = sitk.Cast(new_img, sitk.sitkInt16)
    return new_img

def _write_slice_to_path(target_image, original_sample_path, target_path, slice_thickness=None):
    # 복사할 DICOM 태그 목록
    tags_to_copy = ["0010|0010",  # Patient Name
                    "0010|0020",  # Patient ID
                    "0010|0030",  # Patient Birth Date
                    "0020|000D",  # Study Instance UID, for machine consumption
                    "0020|0010",  # Study ID, for human consumption
                    "0008|0020",  # Study Date
                    "0008|0030",  # Study Time
                    "0008|0050",  # Accession Number
                    "0008|0060"   # Modality
    ]

    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")

    # 원본 이미지에서 메타데이터 읽기
    original_image = sitk.ReadImage(original_sample_path)
    original_key_tuple = original_image.GetMetaDataKeys()
    original_tag_values = [(tag, original_image.GetMetaData(tag)) for tag in original_key_tuple]

    # 시리즈 관련 태그 설정
    try:
        series_tag_value = original_image.GetMetaData("0008|103e")
    except RuntimeError:
        series_tag_value = "tag_None"

    series_tag_values = [(k, original_image.GetMetaData(k)) for k in tags_to_copy if original_image.HasMetaDataKey(k)] + \
                        [("0008|0031", modification_time),  # Series Time
                         ("0008|0021", modification_date),  # Series Date
                         ("0008|103e", series_tag_value + " Processed-SimpleITK")]

    # slice_thickness가 None인 경우, original_sample_path의 Slice Thickness를 참조
    if slice_thickness is None:
        if "0018|0050" in original_image.GetMetaDataKeys():  # Slice Thickness 태그 확인
            slice_thickness = original_image.GetMetaData("0018|0050")
        else:
            raise ValueError("Slice Thickness not found in the original DICOM file and not provided.")

    # 슬라이스를 저장할 경로 만들기 (디렉터리 생성)
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()

    # 3D 이미지 처리 - 각 슬라이스를 분리해서 저장
    for i in range(target_image.GetDepth()):
        slice_image = target_image[:, :, i]  # i번째 슬라이스 추출

        # 단일 슬라이스 저장
        instance_number = i + 1
        sop_uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{target_path}.{instance_number}"))

        # 원본 메타데이터 복사
        for tag, value in original_tag_values:
            try:
                slice_image.SetMetaData(tag, value)
            except:
                continue

        # 시리즈 메타데이터 추가
        for tag, value in series_tag_values:
            slice_image.SetMetaData(tag, value)

        # CT로 설정 (타입 지정)
        slice_image.SetMetaData("0008|0060", "CT")

        # 인스턴스 별 태그 추가
        slice_image.SetMetaData("0008|0012", time.strftime("%Y%m%d"))  # Instance Creation Date
        slice_image.SetMetaData("0008|0013", time.strftime("%H%M%S"))  # Instance Creation Time
        slice_image.SetMetaData("0008|0018", sop_uid)  # SOP Instance UID

        slice_position = target_image.TransformIndexToPhysicalPoint((0, 0, i))  # 3D 인덱스 사용
        slice_image.SetMetaData("0020|0032", '\\'.join(map(str, slice_position)))  # Image Position (Patient)
        slice_image.SetMetaData("0020|0013", str(instance_number))  # Instance Number
        slice_image.SetMetaData("0018|0050", str(slice_thickness))  # Slice Thickness
    writer.SetFileName(target_path)
    writer.Execute(target_image)

def resize_dicom_slice(image_path, target_size, write_path, interploation_method="spline"):
    resized_image_obj = _resize_dicom_slice(image_path, target_size,
                                            interpolation_method=interploation_method)
    _write_slice_to_path(resized_image_obj, image_path, target_path=write_path)