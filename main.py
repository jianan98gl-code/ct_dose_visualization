'''
主程序，功能：
1. 读取CT和剂量NIfTI文件，重采样剂量到CT空间，
2. 提取用户指定的轴位切片，加载RTSTRUCT轮廓并映射到CT空间，
3. 最后在三个轴位切片上叠加剂量分布和轮廓进行可视化展示。
'''

import os
import matplotlib.pyplot as plt
import nibabel as nib
import ct_dose_origin as fc
from utils import (
    DEFAULT_TITLE,
    build_roi_masks,
    extract_slices,
    load_ct_nifti,
    load_dose_nifti,
    load_roi_reference_geometry,
    resample,
    visualize_three_planes_overlay,
)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    volume_dir = os.path.join(base_dir, "Volume data")

    ct_nii_file = os.path.join(volume_dir, "201 Extended FOV iDose (3).nii")
    dose_nii_file = os.path.join(volume_dir, "205 Eclipse Doses.nii")
    rs_dcm_file = os.path.join(base_dir, "NPC_401", "RS.NPC_401.AutoPlan.dcm")

    ct_img_nib = nib.load(ct_nii_file)
    dose_img_nib = nib.load(dose_nii_file)

    ct_array, ct_origin, ct_spacing, vmin, vmax = load_ct_nifti(ct_nii_file)
    dose_array, dose_origin, dose_spacing = load_dose_nifti(dose_nii_file)

    _, _, _, dose_on_ct = resample(
        ct_array,
        ct_origin,
        ct_spacing,
        dose_array,
        dose_origin,
        dose_spacing,
        ct_affine=ct_img_nib.affine,
        dose_affine=dose_img_nib.affine,
    )

    z_mid, y_mid, x_mid = extract_slices(ct_array, dose_on_ct)

    structures = fc.load_structures(rs_dcm_file)
    rois = structures["rois"]

    roi_ct_origin, roi_ct_spacing, roi_ct_orientation = load_roi_reference_geometry(
        os.path.join(base_dir, "NPC_401")
    )
    roi_masks = build_roi_masks(rois, ct_array.shape, roi_ct_origin, roi_ct_spacing, roi_ct_orientation)

    print("重采样完成")
    print(f"  CT shape:              {ct_array.shape}")
    print(f"  Dose resampled shape:  {dose_on_ct.shape}")
    print(f"  Dose max:              {dose_on_ct.max():.2f} Gy")

    fig = visualize_three_planes_overlay(
        ct_array=ct_array,
        dose_on_ct=dose_on_ct,
        roi_masks=roi_masks,
        ct_spacing=ct_spacing,
        z_idx=z_mid,
        y_idx=y_mid,
        x_idx=x_mid,
        vmin=vmin,
        vmax=vmax,
        dose_max=dose_on_ct.max(),
        dose_threshold_ratio=0.1,
        title=DEFAULT_TITLE,
    )

    output_file = os.path.join(base_dir, "ct_dose_roi_four_panel.png")
    fig.savefig(output_file, dpi=150, facecolor="white")
    print(f"已保存: {output_file}")
    plt.show()
    print("Done! Image displayed.")


if __name__ == "__main__":
    main()
