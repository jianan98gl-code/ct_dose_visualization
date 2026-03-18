'''
一些函数包储存在这里，让主程序简洁一些，同时也方便后续维护和扩展。
'''

import numpy as np
import nibabel as nib
import SimpleITK as sitk
import pydicom
import os
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.ticker import MultipleLocator
import  ct_dose_origin as fc

DEFAULT_TITLE = "Head & Neck Cancer - Dose Distribution (CT+Dose)"

roi_map = {
    45: "GTV70-4",
    43: "GTVln66",
    41: "CTV60",
    40: "PTV6000",
    35: "BrainStem",
    13: "SpinalCord",
    20: "OpticChiasm",
    19: "OpticNrv_L",
    18: "OpticNrv_R",
    17: "Parotid_L",
    16: "Parotid_R",
}

colors = {
    "GTV70-4": "red",
    "GTVln66": "magenta",
    "CTV60": "limegreen",
    "PTV6000": "cyan",
    "BrainStem": "yellow",
    "SpinalCord": "turquoise",
    "OpticChiasm": "orange",
    "OpticNrv_L": "blue",
    "OpticNrv_R": "deepskyblue",
    "Parotid_L": "purple",
    "Parotid_R": "brown",
}

linewidths = {
    "GTV70-4": 2.8,
    "GTVln66": 2.5,
    "CTV60": 1.8,
    "PTV6000": 1.5,
    "BrainStem": 1.8,
    "SpinalCord": 1.8,
    "OpticChiasm": 1.5,
    "OpticNrv_L": 1.5,
    "OpticNrv_R": 1.5,
    "Parotid_L": 1.8,
    "Parotid_R": 1.8,
}
                    
def load_ct_nifti(ct_nii_file: str):
    img = nib.load(ct_nii_file)
    ct_array = img.get_fdata().astype(np.float32)
    affine = img.affine
    
    # 提取 spacing 和 origin (X,Y,Z坐标系)
    spacing = np.sqrt(np.sum(affine[:3, :3]**2, axis=0))
    origin = affine[:3, 3]
    
    # 数据转为(Z,Y,X)；origin保持(X,Y,Z)，spacing返回为[Z,Y,X]以兼容下游轮廓函数
    ct_array = np.transpose(ct_array, (2, 1, 0))
    origin_xyz = np.array([origin[0], origin[1], origin[2]])
    spacing_zyx = np.array([spacing[2], spacing[1], spacing[0]])
    
    vmin, vmax = -150, 350
    return ct_array, origin_xyz, spacing_zyx, vmin, vmax

def load_dose_nifti(dose_nii_file: str):
    img = nib.load(dose_nii_file)
    dose_array = img.get_fdata().astype(np.float32)
    affine = img.affine
    
    spacing = np.sqrt(np.sum(affine[:3, :3]**2, axis=0))
    origin = affine[:3, 3]

    # 剂量单位校正
    if dose_array.max() > 500:
        dose_array = dose_array / 10000.0
    
    # 数据转为(Z,Y,X)；origin保持(X,Y,Z)，spacing返回为[Z,Y,X]
    dose_array = np.transpose(dose_array, (2, 1, 0))
    origin_xyz = np.array([origin[0], origin[1], origin[2]])
    spacing_zyx = np.array([spacing[2], spacing[1], spacing[0]])
    
    return dose_array, origin_xyz, spacing_zyx

def resample(ct_array, ct_origin, ct_spacing, dose_array, dose_origin, dose_spacing, ct_affine=None, dose_affine=None):
    # 重采样
    
    # 创建SimpleITK图像
    ct_image = sitk.GetImageFromArray(ct_array)
    ct_image.SetOrigin(tuple(float(v) for v in ct_origin))
    # SimpleITK需要(X,Y,Z)顺序，这里把[Z,Y,X]转换为[X,Y,Z]
    ct_image.SetSpacing((float(ct_spacing[2]), float(ct_spacing[1]), float(ct_spacing[0])))
    
    # 从affine矩阵设置方向矩阵（处理坐标轴翻转）
    if ct_affine is not None:
        ct_rotation = ct_affine[:3, :3]
        ct_direction = []
        for i in range(3):
            col = ct_rotation[:, i]
            ct_direction.extend(col / np.linalg.norm(col))
        ct_image.SetDirection(ct_direction)
    
    dose_image = sitk.GetImageFromArray(dose_array)
    dose_image.SetOrigin(tuple(float(v) for v in dose_origin))
    dose_image.SetSpacing((float(dose_spacing[2]), float(dose_spacing[1]), float(dose_spacing[0])))
    
    if dose_affine is not None:
        dose_rotation = dose_affine[:3, :3]
        dose_direction = []
        for i in range(3):
            col = dose_rotation[:, i]
            dose_direction.extend(col / np.linalg.norm(col))
        dose_image.SetDirection(dose_direction)

    # 重采样到CT图像的空间
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ct_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    dose_on_ct_sitk = resampler.Execute(dose_image)
    dose_on_ct = sitk.GetArrayFromImage(dose_on_ct_sitk)

    return ct_image, dose_image, dose_on_ct_sitk, dose_on_ct

'''提取不同轴位数据和轮廓'''

def extract_slices(ct_array, dose_on_ct):
    """交互式选择三个轴位索引。"""
    z_default = ct_array.shape[0] // 2
    y_default = ct_array.shape[1] // 2
    x_default = ct_array.shape[2] // 2
    
    # 交互式输入
    print("\n" + "="*60)
    print("输入要提取的层号（按Enter使用默认值）")
    print("="*60)
    
    z_input = input(f"Z轴层号 [默认: {z_default}] (范围: 0-{ct_array.shape[0]-1}): ").strip()
    z_mid = int(z_input) if z_input else z_default
    
    y_input = input(f"Y轴层号 [默认: {y_default}] (范围: 0-{ct_array.shape[1]-1}): ").strip()
    y_mid = int(y_input) if y_input else y_default
    
    x_input = input(f"X轴层号 [默认: {x_default}] (范围: 0-{ct_array.shape[2]-1}): ").strip()
    x_mid = int(x_input) if x_input else x_default
    
    # 验证范围
    z_mid = max(0, min(z_mid, ct_array.shape[0]-1))
    y_mid = max(0, min(y_mid, ct_array.shape[1]-1))
    x_mid = max(0, min(x_mid, ct_array.shape[2]-1))

    print(f"✓ Axial (Z={z_mid}): ct {ct_array[z_mid].shape}, dose {dose_on_ct[z_mid].shape}")
    print(f"✓ Coronal (Y={y_mid}): ct {ct_array[:, y_mid, :].shape}, dose {dose_on_ct[:, y_mid, :].shape}")
    print(f"✓ Sagittal (X={x_mid}): ct {ct_array[:, :, x_mid].shape}, dose {dose_on_ct[:, :, x_mid].shape}")

    return z_mid, y_mid, x_mid


def load_roi_reference_geometry(ct_folder: str):
    """读取RTSTRUCT所对应DICOM CT的几何参数"""
    ct_files = [os.path.join(ct_folder, f) for f in os.listdir(ct_folder) if "CT." in f and f.endswith(".dcm")]
    if not ct_files:
        raise FileNotFoundError(f"未在 {ct_folder} 找到CT DICOM文件")

    dsets = [pydicom.dcmread(fp, stop_before_pixels=True) for fp in ct_files]
    dsets = sorted(dsets, key=lambda ds: float(ds.ImagePositionPatient[2]))

    ds0 = dsets[0]
    ct_origin = np.array(ds0.ImagePositionPatient, dtype=float)  # [x, y, z]
    ct_orientation = np.array(ds0.ImageOrientationPatient, dtype=float)
    spacing_xy = np.array(ds0.PixelSpacing, dtype=float)

    if len(dsets) > 1:
        spacing_z = float(dsets[1].ImagePositionPatient[2] - dsets[0].ImagePositionPatient[2])
    else:
        spacing_z = float(getattr(ds0, "SliceThickness", 1.0))

    # 与view_rt_dcm_files一致，使用[z, y, x]
    ct_spacing = np.array([spacing_z, spacing_xy[0], spacing_xy[1]], dtype=float)
    return ct_origin, ct_spacing, ct_orientation


def build_roi_masks(rois, ct_shape, ct_origin, ct_spacing, ct_orientation):
    """将RTSTRUCT轮廓栅格化为3D掩膜，便于任意轴位切片显示真实轮廓。"""
    roi_masks = {}
    for roi_num in roi_map:
        if roi_num not in rois:
            continue

        mask_3d = np.zeros(ct_shape, dtype=bool)
        for contour in rois[roi_num]['contours']:
            if len(contour) < 3:
                continue

            pix = fc.patient_to_pixel_coords(contour, ct_origin, ct_spacing, ct_shape, ct_orientation)
            z_idx = int(round(float(np.mean(pix[:, 0]))))
            if z_idx < 0 or z_idx >= ct_shape[0]:
                continue

            poly_yx = pix[:, [1, 2]]
            y_min = max(int(np.floor(np.min(poly_yx[:, 0]))), 0)
            y_max = min(int(np.ceil(np.max(poly_yx[:, 0]))), ct_shape[1] - 1)
            x_min = max(int(np.floor(np.min(poly_yx[:, 1]))), 0)
            x_max = min(int(np.ceil(np.max(poly_yx[:, 1]))), ct_shape[2] - 1)
            if y_min > y_max or x_min > x_max:
                continue

            yy, xx = np.mgrid[y_min:y_max + 1, x_min:x_max + 1]
            points_xy = np.column_stack((xx.ravel(), yy.ravel()))
            poly_xy = np.column_stack((poly_yx[:, 1], poly_yx[:, 0]))
            inside = Path(poly_xy).contains_points(points_xy).reshape(yy.shape)
            mask_3d[z_idx, y_min:y_max + 1, x_min:x_max + 1] |= inside

        if np.any(mask_3d):
            roi_masks[roi_num] = mask_3d

    return roi_masks


def draw_mask_contours(ax, plane_masks, show_legend=False, origin_mode='upper', extent=None):
    """在单个子图上绘制2D掩膜轮廓。"""
    if not plane_masks:
        return

    added_to_legend = set()
    for roi_num in roi_map:
        if roi_num not in plane_masks:
            continue

        mask_2d = plane_masks[roi_num]
        if not np.any(mask_2d):
            continue

        name = roi_map[roi_num]
        color = colors[name]
        lw = linewidths[name]
        contour_kwargs = {
            'levels': [0.5],
            'colors': [color],
            'linewidths': lw,
            'origin': origin_mode,
        }
        if extent is not None:
            contour_kwargs['extent'] = extent
        ax.contour(mask_2d.astype(np.float32), **contour_kwargs)
        if show_legend and name not in added_to_legend:
            added_to_legend.add(name)

    if show_legend and added_to_legend:
        handles = [plt.Line2D([0], [0], color=colors[name], lw=2.0) for name in added_to_legend]
        labels = list(added_to_legend)
        ax.legend(handles, labels, loc='upper right', fontsize=9, framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)


def _pad_center_2d(arr, target_h, target_w, fill_value):
    """中心填充数组到目标尺寸。"""
    h, w = arr.shape
    target_h = max(target_h, h)
    target_w = max(target_w, w)
    pad_h = target_h - h
    pad_w = target_w - w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return np.pad(arr, ((top, bottom), (left, right)), mode='constant', constant_values=fill_value)


def _build_canvas_slice(ct_slice, dose_slice, plane_masks, canvas_h_mm, canvas_w_mm, row_mm, col_mm, vmin):
    """为单个切面构建 canvas，padding 到统一尺寸。"""
    target_h = int(np.round(canvas_h_mm / row_mm))
    target_w = int(np.round(canvas_w_mm / col_mm))

    ct_padded = _pad_center_2d(ct_slice, target_h, target_w, vmin)
    dose_padded = _pad_center_2d(dose_slice, target_h, target_w, 0.0)
    masks_padded = {
        roi_num: _pad_center_2d(mask_2d.astype(np.uint8), target_h, target_w, 0).astype(bool)
        for roi_num, mask_2d in plane_masks.items()
    }
    return ct_padded, dose_padded, masks_padded


def _prepare_plane_slices(ct_array, dose_on_ct, roi_masks, z_idx, y_idx, x_idx, vmin, vmax):
    """提取 4 个切面的 CT、剂量和 ROI 掩膜数据。"""
    axial_masks = {roi_num: mask[z_idx] for roi_num, mask in roi_masks.items()}
    coronal_masks = {roi_num: mask[:, y_idx, :] for roi_num, mask in roi_masks.items()}
    sagittal_masks = {roi_num: mask[:, :, x_idx] for roi_num, mask in roi_masks.items()}
    mip_masks = {roi_num: np.max(mask, axis=0) for roi_num, mask in roi_masks.items()}

    ct_mip = np.clip(np.max(ct_array, axis=0), vmin, vmax)
    dose_mip = np.max(dose_on_ct, axis=0)

    return {
        'axial': (np.clip(ct_array[z_idx], vmin, vmax), dose_on_ct[z_idx], axial_masks),
        'coronal': (np.clip(ct_array[:, y_idx, :], vmin, vmax), dose_on_ct[:, y_idx, :], coronal_masks),
        'sagittal': (np.clip(ct_array[:, :, x_idx], vmin, vmax), dose_on_ct[:, :, x_idx], sagittal_masks),
        'mip': (ct_mip, dose_mip, mip_masks),
    }


def _compute_canvas_geometry(ct_array, ct_spacing, z_idx, y_idx, x_idx):
    """计算 canvas 尺寸（mm）和纵横比。"""
    sz, sy, sx = ct_spacing
    
    axial_h_mm = ct_array[z_idx].shape[0] * sy
    axial_w_mm = ct_array[z_idx].shape[1] * sx
    coronal_h_mm = ct_array[:, y_idx, :].shape[0] * sz
    coronal_w_mm = ct_array[:, y_idx, :].shape[1] * sx
    sagittal_h_mm = ct_array[:, :, x_idx].shape[0] * sz
    sagittal_w_mm = ct_array[:, :, x_idx].shape[1] * sy
    
    ct_mip = np.max(ct_array, axis=0)
    mip_h_mm = ct_mip.shape[0] * sy
    mip_w_mm = ct_mip.shape[1] * sx

    canvas_w_mm = max(axial_w_mm, coronal_w_mm, sagittal_w_mm, mip_w_mm)
    canvas_h_mm = max(axial_h_mm, coronal_h_mm, sagittal_h_mm, mip_h_mm)
    canvas_box_aspect = canvas_h_mm / canvas_w_mm if canvas_w_mm else 1.0

    return canvas_w_mm, canvas_h_mm, canvas_box_aspect


def _create_figure_layout():
    """创建 figure 和 gridspec 布局。"""
    fig = plt.figure(figsize=(14.4, 11.4), dpi=110, facecolor='white')
    grid = fig.add_gridspec(
        nrows=2, ncols=3,
        width_ratios=[1.0, 1.0, 0.18],
        left=0.045, right=0.958, top=0.895, bottom=0.055,
        wspace=0.02, hspace=0.03,
    )
    axes = np.array([
        [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])],
        [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])],
    ])
    side_grid = grid[:, 2].subgridspec(2, 1, height_ratios=[0.50, 0.50], hspace=0.03)
    cax = fig.add_subplot(side_grid[0, 0])
    legend_ax = fig.add_subplot(side_grid[1, 0])
    
    return fig, axes, cax, legend_ax


def _draw_single_panel(ax, ct_plot, dose_plot, masks_plot, panel_title, 
                       canvas_w_mm, canvas_h_mm, canvas_box_aspect, 
                       dose_masked, cmap, dose_threshold, dose_max, 
                       vmin, vmax, dose_interp, origin_mode, full_extent):
    """绘制单个面板（CT + 剂量叠加 + ROI 轮廓）。"""
    ax.set_box_aspect(canvas_box_aspect)
    ax.set_anchor('C')
    ax.set_facecolor('black')
    
    # 绘制 CT
    ax.imshow(ct_plot, cmap='gray', vmin=vmin, vmax=vmax, interpolation='none',
              aspect='equal', origin=origin_mode, extent=full_extent)
    
    # 绘制剂量叠加
    overlay_obj = ax.imshow(dose_masked, cmap=cmap, vmin=dose_threshold, vmax=dose_max,
                            interpolation=dose_interp, alpha=0.7, aspect='equal',
                            origin=origin_mode, extent=full_extent)
    
    # 绘制 ROI 轮廓
    draw_mask_contours(ax, masks_plot, show_legend=False, origin_mode=origin_mode, extent=full_extent)
    
    # 设置坐标轴
    ax.set_xlim(0, canvas_w_mm)
    ax.set_ylim(0, canvas_h_mm)
    ax.axis('off')
    
    # 添加标题标签
    ax.text(0.02, 0.98, panel_title, transform=ax.transAxes,
            ha='left', va='top', fontsize=10.5, fontweight='bold', color='white',
            bbox=dict(facecolor='black', alpha=0.42, edgecolor='none', pad=2.0))
    
    return overlay_obj


def _add_legend_to_axes(legend_ax, present_roi_names):
    """在指定 axes 上添加 ROI 图例。"""
    legend_pos = legend_ax.get_position()
    legend_ax.set_position([
        legend_pos.x0 + legend_pos.width * 0.02,
        legend_pos.y0 - legend_pos.height * 0.03,
        legend_pos.width * 0.96,
        legend_pos.height * 1.08,
    ])
    legend_ax.axis('off')
    
    if present_roi_names:
        handles = [plt.Line2D([0], [0], color=colors[name], lw=2.2) for name in present_roi_names]
        legend = legend_ax.legend(
            handles, present_roi_names,
            loc='lower right', bbox_to_anchor=(0.98, 0.02), ncol=1, fontsize=9.0,
            frameon=False, prop={'family': 'Arial', 'size': 9.0},
            columnspacing=0.6, handlelength=2.4, borderpad=0.1, labelspacing=0.44,
        )
        legend.set_title("ROI", prop={'family': 'Arial', 'size': 10.2, 'weight': 'bold'})


def _add_colorbar_to_axes(fig, cax, overlay_obj, dose_max, dose_threshold):
    """在指定 axes 上添加色条。"""
    cax_pos = cax.get_position()
    cax.set_position([
        cax_pos.x0 + cax_pos.width * 0.18,
        cax_pos.y0 + cax_pos.height * 0.015,
        cax_pos.width * 0.24,
        cax_pos.height * 0.97,
    ])

    cbar = fig.colorbar(overlay_obj, cax=cax)
    cbar.set_label('Dose (Gy)', fontsize=12, fontweight='bold', fontname='Arial')

    # 计算色条刻度
    tick_candidates = np.arange(10, int(np.ceil(dose_max / 10.0)) * 10 + 1, 10, dtype=int)
    major_ticks = [int(t) for t in tick_candidates if dose_threshold <= t <= dose_max]
    if not major_ticks:
        major_ticks = [int(np.round(dose_threshold)), int(np.round(dose_max))]

    cbar.set_ticks(major_ticks)
    cbar.ax.set_yticklabels([str(t) for t in major_ticks], fontname='Arial', fontsize=9)
    cbar.ax.yaxis.set_minor_locator(MultipleLocator(5))
    cbar.ax.tick_params(which='major', length=5, width=1.0)
    cbar.ax.tick_params(which='minor', length=3, width=0.8)

# 关键输出函数
def visualize_three_planes_overlay(
    ct_array,
    dose_on_ct,
    roi_masks,
    ct_spacing,
    z_idx,
    y_idx,
    x_idx,
    vmin,
    vmax,
    dose_max,
    dose_threshold_ratio=0.1,
    title="Head & Neck Cancer - Dose Distribution (CT+Dose)"
):
    """
    终：可视化 CT、剂量分布和 ROI 轮廓的四面板视图。
    一些关键参数说明
        ct_array: CT 图像数据
        dose_on_ct: 重采样到 CT 空间的剂量数据
        roi_masks: ROI 掩膜字典
        ct_spacing: CT 间距 [z, y, x]
        z_idx, y_idx, x_idx: 切面索引
        dose_threshold_ratio: 剂量阈值比例
    """
    # 初始化参数
    dose_threshold = dose_max * dose_threshold_ratio
    cmap = fc.build_dose_overlay_cmap()
    sz, sy, sx = ct_spacing
    
    # 获取存在的 ROI 名称
    present_roi_names = [
        roi_map[roi_num] for roi_num in roi_map
        if roi_num in roi_masks and np.any(roi_masks[roi_num])
    ]
    
    # 创建 figure 和布局
    fig, axes, cax, legend_ax = _create_figure_layout()
    
    # 计算几何参数
    canvas_w_mm, canvas_h_mm, canvas_box_aspect = _compute_canvas_geometry(
        ct_array, ct_spacing, z_idx, y_idx, x_idx
    )
    full_extent = [0, canvas_w_mm, 0, canvas_h_mm]
    
    # 准备切面数据
    plane_slices = _prepare_plane_slices(ct_array, dose_on_ct, roi_masks, z_idx, y_idx, x_idx, vmin, vmax)
    
    # 定义 4 个面板的绘制配置
    plane_config = [
        (axes[0, 0], 'axial', sy, sx, 'none', 'lower', f"Axial (Z={z_idx})"),
        (axes[0, 1], 'coronal', sz, sx, 'bilinear', 'lower', f"Coronal (Y={y_idx})"),
        (axes[1, 0], 'sagittal', sz, sy, 'bilinear', 'lower', f"Sagittal (X={x_idx})"),
        (axes[1, 1], 'mip', sy, sx, 'bilinear', 'lower', "Dose MIP (Axial Projection)"),
    ]
    
    overlay_obj = None
    for ax, plane_name, row_mm, col_mm, dose_interp, origin_mode, panel_title in plane_config:
        ct_slice, dose_slice, masks = plane_slices[plane_name]
        
        # 构建 canvas
        ct_plot, dose_plot, masks_plot = _build_canvas_slice(
            ct_slice, dose_slice, masks, canvas_h_mm, canvas_w_mm, row_mm, col_mm, vmin
        )
        
        # 准备剂量叠加
        dose_masked = np.ma.masked_less(dose_plot, dose_threshold)
        
        # 绘制面板
        overlay_obj = _draw_single_panel(
            ax, ct_plot, dose_plot, masks_plot, panel_title,
            canvas_w_mm, canvas_h_mm, canvas_box_aspect,
            dose_masked, cmap, dose_threshold, dose_max,
            vmin, vmax, dose_interp, origin_mode, full_extent
        )
    
    # 添加图例和色条
    _add_legend_to_axes(legend_ax, present_roi_names)
    _add_colorbar_to_axes(fig, cax, overlay_obj, dose_max, dose_threshold)
    
    # 设置标题
    fig.suptitle(title, fontsize=17, fontweight='bold', y=0.978)
    
    return fig
