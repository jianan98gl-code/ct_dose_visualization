'''
最早写的一个轴面剂量叠加图，先保留在这里，可以当做可视化思路的模版，
后续可以改成更通用的函数，或者直接删除。
'''

import pydicom
import numpy as np
import matplotlib.pyplot as plt
import os
import SimpleITK as sitk
from typing import Dict, Tuple, Optional
from matplotlib.colors import LinearSegmentedColormap

def load_rtdose(dose_file):
    ds = pydicom.dcmread(dose_file)

    # 原始数据
    dose_grid_data = ds.pixel_array.astype(np.float32)
    # 缩放因子
    dose_scaling = float(ds.DoseGridScaling)
    # 实际剂量
    dose_array = dose_grid_data * dose_scaling

    dose_origin = tuple(map(float, ds.ImagePositionPatient))
    spacing_xy = list(map(float, ds.PixelSpacing))
    spacing_z = float(ds.GridFrameOffsetVector[1] - ds.GridFrameOffsetVector[0])

    dose_spacing = (spacing_xy[0], spacing_xy[1], spacing_z)

    dose_max = np.max(dose_array)

    return dose_array, dose_origin, dose_spacing, dose_max


def load_ct_series(file_folder):

    ct_files = [pydicom.dcmread(os.path.join(file_folder, f)) 
                for f in os.listdir(file_folder) if "Image" in f]
    ct_files = sorted(ct_files, key=lambda x: float(x.ImagePositionPatient[2]))

    # 先取原始像素，再根据 slope / intercept 转成 HU
    ct_array_raw = np.stack([f.pixel_array for f in ct_files]).astype(np.float32)
    ds_first = ct_files[0]
    slope = float(getattr(ds_first, "RescaleSlope", 1.0))
    intercept = float(getattr(ds_first, "RescaleIntercept", 0.0))
    ct_array = ct_array_raw * slope + intercept

    ds_first = ct_files[0]
    if 'WindowCenter' in ds_first and 'WindowWidth' in ds_first:
        window_center = float(ds_first.WindowCenter[0]) if isinstance(ds_first.WindowCenter, (list, tuple)) else float(ds_first.WindowCenter)
        window_width = float(ds_first.WindowWidth[0]) if isinstance(ds_first.WindowWidth, (list, tuple)) else float(ds_first.WindowWidth)
        vmin = window_center - window_width / 2
        vmax = window_center + window_width / 2
        print(f"从 DICOM 提取的推荐窗位：center={window_center}, width={window_width} → vmin={vmin}, vmax={vmax}")
    else:
        vmin, vmax = -150, 350
        print("DICOM 无窗位信息，使用默认软组织窗：vmin=-150, vmax=350")

    ct_origin = np.array(ct_files[0].ImagePositionPatient, dtype=float)
    ct_orientation = np.array(ct_files[0].ImageOrientationPatient, dtype=float)

    spacing_xy = np.array(ct_files[0].PixelSpacing, dtype=float)
    spacing_z = float(ct_files[1].ImagePositionPatient[2] - ct_files[0].ImagePositionPatient[2])

    ct_spacing = np.array([spacing_z, spacing_xy[0], spacing_xy[1]])

    return ct_array, ct_origin, ct_spacing, ct_orientation, vmin, vmax

def transtoImage(ct_array, ct_origin, ct_spacing, dose_array, dose_origin, dose_spacing):
    # 创建SimpleITK图像
    ct_image = sitk.GetImageFromArray(ct_array)
    ct_image.SetOrigin(ct_origin)
    ct_image.SetSpacing(ct_spacing[[2,1,0]]) 

    dose_image = sitk.GetImageFromArray(dose_array)
    dose_image.SetOrigin(dose_origin)
    dose_image.SetSpacing(dose_spacing)

    # 重采样剂量图像到CT图像的空间
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ct_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    dose_on_ct_sitk = resampler.Execute(dose_image)

    dose_on_ct = sitk.GetArrayFromImage(dose_on_ct_sitk)

    return ct_image, dose_image, dose_on_ct_sitk, dose_on_ct


def load_structures(rs_file):
    # 读取 RTSTRUCT 文件，提取所有 ROI 的信息。
    
    ds = pydicom.dcmread(rs_file)
    rois = {}
    
    # 遍历 ReferencedROISequence 获取 ROI 基本信息
    roi_numbers = {}
    if hasattr(ds, 'ReferencedROISequence'):
        for ref_roi in ds.ReferencedROISequence:
            roi_num = int(ref_roi.ReferencedROINumber)
            roi_name = ref_roi.ReferencedROIName
            roi_numbers[roi_num] = roi_name
    
    # 遍历 ROIContourSequence 获取轮廓数据和颜色
    if hasattr(ds, 'ROIContourSequence'):
        for roi_contour in ds.ROIContourSequence:
            roi_num = int(roi_contour.ReferencedROINumber)
            roi_name = roi_numbers.get(roi_num, f"ROI_{roi_num}")
            
            # 提取显示颜色
            display_color = None
            if hasattr(roi_contour, 'ROIDisplayColor'):
                try:
                    color_vals = roi_contour.ROIDisplayColor
                    if len(color_vals) >= 3:
                        # RGB 0-255 -> 0-1
                        display_color = tuple(float(c) / 255.0 for c in color_vals[:3])
                except Exception as e:
                    print(f"警告：无法解析 {roi_name} 的 ROIDisplayColor: {e}")
            
            # 提取轮廓数据
            contours = []
            if hasattr(roi_contour, 'ContourSequence'):
                for contour in roi_contour.ContourSequence:
                    if hasattr(contour, 'ContourData'):
                        # ContourData 是 [x1, y1, z1, x2, y2, z2, ...] 的平面数组
                        contour_data = contour.ContourData
                        # 重塑为 nx3 的点坐标数组
                        points = np.array(contour_data).reshape(-1, 3)
                        contours.append(points)
            
            rois[roi_num] = {
                'name': roi_name,
                'displayColor': display_color,
                'contours': contours
            }
    
    return {'rois': rois}


def patient_to_pixel_coords(
    patient_coords: np.ndarray,
    ct_origin: np.ndarray,
    ct_spacing: np.ndarray,
    ct_shape: Tuple[int, int, int],
    ct_orientation: Optional[np.ndarray]
):
    # 把患者坐标（mm）转换为 CT 图像的像素坐标。
    
    relative = patient_coords - ct_origin[np.newaxis, :]
    pixel_coords = np.column_stack([
        relative[:, 0] / ct_spacing[2], 
        relative[:, 1] / ct_spacing[1], 
        relative[:, 2] / ct_spacing[0], 
    ])
    
    # 转换为数组坐标顺序 [z_pix, y_pix, x_pix]
    pixel_coords = pixel_coords[:, [2, 1, 0]]
    return pixel_coords


# 关键的ROI映射与显示
roi_map = {
    45: "GTV70-4", 43: "GTVln66", 41: "CTV60", 40: "PTV6000",
    35: "BrainStem", 13: "SpinalCord", 20: "OpticChiasm",
    19: "OpticNrv_L", 18: "OpticNrv_R", 17: "Parotid_L", 16: "Parotid_R"
}
colors = {
    "GTV70-4": 'red', "GTVln66": 'magenta', "CTV60": 'limegreen', "PTV6000": 'cyan',
    "BrainStem": 'yellow', "SpinalCord": 'turquoise', "OpticChiasm": 'orange',
    "OpticNrv_L": 'blue', "OpticNrv_R": 'deepskyblue', "Parotid_L": 'purple', "Parotid_R": 'brown'
}
linewidths = {
    "GTV70-4": 2.8, "GTVln66": 2.5, "CTV60": 1.8, "PTV6000": 1.5,
    "BrainStem": 1.8, "SpinalCord": 1.8, "OpticChiasm": 1.5,
    "OpticNrv_L": 1.5, "OpticNrv_R": 1.5, "Parotid_L": 1.8, "Parotid_R": 1.8
}

def get_contours_on_slice(rois, slice_idx, ct_origin, ct_spacing, ct_shape, ct_orientation, slice_thickness=3.0):
    slice_z_mm = ct_origin[2] + slice_idx * ct_spacing[0]
    slice_contours = {}
    for roi_num in roi_map:
        if roi_num not in rois:
            continue
        name = roi_map[roi_num]
        contours = rois[roi_num]['contours']
        cs = []
        for contour in contours:
            z_vals = contour[:, 2]
            if np.min(z_vals) <= slice_z_mm + slice_thickness and np.max(z_vals) >= slice_z_mm - slice_thickness:
                pix = patient_to_pixel_coords(contour, ct_origin, ct_spacing, ct_shape, ct_orientation)
                pts = pix[:, [1, 2]]
                if len(pts) > 0:
                    cs.append(pts)
        if cs:
            slice_contours[roi_num] = {
                'name': name,
                'color': colors[name],
                'contours': cs,
                'linewidth': linewidths[name]
            }
    return slice_contours


def add_contours_to_plot(
    ax,
    slice_contours,
    close_contours
    ):
    # 把轮廓线绘到 matplotlib axes 上。
    
    legend_labels = []
    
    for roi_num, roi_data in slice_contours.items():
        roi_name = roi_data['name']
        color = roi_data['color']
        lw = roi_data['linewidth']
        contours = roi_data['contours']
        
        for contour_points in contours:
            if close_contours and len(contour_points) > 0:
                # 闭合轮廓
                contour_points = np.vstack([contour_points, contour_points[0]])
            
            ax.plot(contour_points[:, 1], contour_points[:, 0],
                   color=color, linewidth=lw, label=roi_name, alpha=0.95)
        
        legend_labels.append(roi_name)
    
    return legend_labels

def build_dose_overlay_cmap():
    # 色阶设计，高剂量红，低剂量蓝
    return LinearSegmentedColormap.from_list(
        "dose_overlay",
        [
            (0.0, 0.0, 1.0, 0.80), 
            (0.0, 1.0, 1.0, 0.75), 
            (0.0, 1.0, 0.0, 0.70),
            (1.0, 1.0, 0.0, 0.65), 
            (1.0, 0.5, 0.0, 0.60),
            (1.0, 0.0, 0.0, 0.55),
        ],
        N=256,
    )


def visualize_dose_overlay(
    ct_array,
    dose_on_ct,
    slice_idx,
    ct_vmin,
    ct_vmax,
    dose_max=None,
    dose_threshold_ratio=0.1,
    slice_contours: Optional[Dict] = None,
    ct_origin: Optional[np.ndarray] = None,
    ct_spacing: Optional[np.ndarray] = None,
    ):

    # 提取当前 slice 的 CT 和剂量数据

    ct_slice = np.clip(ct_array[slice_idx], ct_vmin, ct_vmax)
    dose_slice = dose_on_ct[slice_idx]

    if dose_max is None:
        dose_max = float(np.max(dose_on_ct))

    dose_threshold = dose_max * dose_threshold_ratio
    dose_masked = np.ma.masked_less(dose_slice, dose_threshold)

    fig, ax = plt.subplots(figsize=(11, 11), dpi=120)
    ax.imshow(ct_slice, cmap="gray", vmin=ct_vmin, vmax=ct_vmax, interpolation="none")
    cmap = build_dose_overlay_cmap()
    overlay = ax.imshow(
        dose_masked,
        cmap=cmap,
        vmin=dose_threshold,
        vmax=dose_max,
        interpolation="none",
        alpha=0.7
    )
    legend_handles = []
    legend_labels = []
    
    if slice_contours:
        added_to_legend = set()
        for roi_num, roi_data in slice_contours.items():
            name = roi_data['name']
            color = roi_data['color']
            lw = roi_data['linewidth']
            contours = roi_data['contours']
            for contour_points in contours:
                if len(contour_points) > 0:
                    contour_closed = np.vstack([contour_points, contour_points[0]])
                else:
                    contour_closed = contour_points
                if name not in added_to_legend:
                    ax.plot(contour_closed[:, 1], contour_closed[:, 0],
                            color=color, linewidth=lw, alpha=0.95, solid_capstyle='round', label=name)
                    added_to_legend.add(name)
                else:
                    ax.plot(contour_closed[:, 1], contour_closed[:, 0],
                            color=color, linewidth=lw, alpha=0.95, solid_capstyle='round')
        legend_handles = [plt.Line2D([0], [0], color=roi_data['color'], lw=2.0) for roi_num, roi_data in slice_contours.items() if roi_data['name'] in added_to_legend]
        legend_labels = [roi_data['name'] for roi_num, roi_data in slice_contours.items() if roi_data['name'] in added_to_legend]

    # 色条
    cbar = plt.colorbar(overlay, fraction=0.046, pad=0.04, ax=ax)
    cbar.set_label("Dose (Gy)", fontsize=11, fontweight='bold')

    # 图例
    if legend_labels:
        legend = ax.legend(
            legend_handles, legend_labels,
            loc='upper right',
            fontsize=9,
            framealpha=0.95,
            edgecolor='black',
            fancybox=True,
            shadow=True
        )
        # 加粗 legend 标题
        if legend.get_title():
            legend.get_title().set_fontweight('bold')

    ax.axis("off")
    title_str = f"Dose Distribution with Target & OAR Contours - Slice {slice_idx} (NPC IMRT Plan)"
    ax.set_title(title_str, fontsize=12, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    BASE_DIR = r"D:\python文件\rt_dose_view\NPC_401"
    dose_file = os.path.join(BASE_DIR, "RD.NPC_401.PlanOpt.dcm")
    rs_file   = os.path.join(BASE_DIR, "RS.NPC_401.AutoPlan.dcm")
    ct_folder = BASE_DIR
    
    # 加载 CT、Dose、Structures
    ct_array, ct_origin, ct_spacing, ct_orientation, default_vmin, default_vmax = load_ct_series(ct_folder)
    dose_array, dose_origin, dose_spacing, dose_max = load_rtdose(dose_file)
    ct_image, dose_image, dose_on_ct_sitk, dose_on_ct = transtoImage(
        ct_array, ct_origin, ct_spacing, dose_array, dose_origin, dose_spacing
    )
    
    structures = load_structures(rs_file)
    rois = structures['rois'] 
    print("="*50)
    
    print("\n" + "="*50)
    print("[CT 和 Dose 信息]")
    print("="*50)
    print(f"CT shape:            {ct_array.shape}")
    print(f"CT origin (x,y,z):   {ct_origin}")
    print(f"CT spacing [z,y,x]:  {ct_spacing}")
    print(f"CT HU window:        {default_vmin:.0f} ~ {default_vmax:.0f}")
    print(f"Dose resampled shape: {dose_on_ct.shape}")
    print(f"Dose max:            {dose_max:.2f} Gy (resampled max: {np.max(dose_on_ct):.2f} Gy)")

    # 选择要显示的平面
    slice_idx = int(np.argmax(np.max(dose_on_ct, axis=(1, 2))))  # 也可以手动选特定层
    slice_idx = 57
    print("\n" + "="*50)
    print(f"[选择 Slice {slice_idx} 进行可视化]")
    print("="*50)

    # 显示关键ROI
    slice_contours = get_contours_on_slice(
        rois, slice_idx, ct_origin, ct_spacing, ct_array.shape, ct_orientation, slice_thickness=3.0
    )

    print(f"\n当前 slice 上的关键 ROI（共 {len(slice_contours)} 个）：")
    print("-" * 50)
    for roi_num, roi_data in slice_contours.items():
        print(f"{roi_num}\t{roi_data['name']}")

    print("\n" + "="*50)
    print("[生成可视化图像]")
    print("="*50)
    visualize_dose_overlay(
        ct_array=ct_array,
        dose_on_ct=dose_on_ct,
        slice_idx=slice_idx,
        ct_vmin=default_vmin,
        ct_vmax=default_vmax,
        dose_max=dose_max,
        dose_threshold_ratio=0.1, # 可修改显示下阈值
        slice_contours=slice_contours,
        ct_origin=ct_origin,
        ct_spacing=ct_spacing,
    )
    print("Done! Image displayed.")

