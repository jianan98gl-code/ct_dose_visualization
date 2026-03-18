# ct_dose_visualization

## Project Overview

A medical imaging processing tool for **preprocessing, registering, and visualizing CT scans with radiotherapy dose distributions**. Supports RTSTRUCT contour mapping and display, used for head and neck cancer radiotherapy plan assessment.

## Project Structure

ct_dose_view/
├── main.py                              # Main program entry point
├── utils.py                             # Utility function library (resampling, processing, visualization)
├── ct_dose_origin.py                    # Original dose processing algorithm (retained as reference)
├── ct_dose_view_whole.py                # Full program version
├── NPC_401/                             # Sample data directory
│   ├── CT.NPC_401.Image*.dcm            # CT slice DICOM files
│   └── RS.NPC_401.AutoPlan.dcm          # RTSTRUCT contour file
├── Volume data/                         # Data storage directory
│   ├── 201 Extended FOV iDose (3).nii   # CT NIfTI data
│   └── 205 Eclipse Doses.nii            # Dose NIfTI data
└── ct_dose_roi_four_panel.png           # Output image example

## Quick Start

### Environment Setup

**Required Libraries:**
bash
pip install nibabel==4.0.2       # NIfTI file handling
pip install pydicom              # DICOM file processing
pip install SimpleITK            # Medical image registration and resampling
pip install numpy                # Numerical computing
pip install matplotlib           # Plot generation

### Basic Usage

1. **Prepare Data**:
   Place the following in `Volume data/` directory:
   - CT.nii` - CT scan NIfTI file
   - Dose.nii` - Radiotherapy plan dose distribution NIfTI file
   - Place `RS.*.dcm` RTSTRUCT file in `NPC_401/` directory

3. **Run the Program**:
bash
python main.py

3. **View Results**: Output image is saved to `ct_dose_roi_four_panel.png`

## Workflow

Input Files
  ⬇
CT NIfTI Load ──┐
               ├─→ Spatial Alignment & Resampling ──→ Resample Dose to CT Space
Dose NIfTI Load ┘

RTSTRUCT File
  ⬇
ROI Load ──→ ROI Mask Generation ──→ Map to CT Space

Fused Data
  ⬇
Slice Extraction (Coronal/Sagittal/Axial)
  ⬇
Visualization Overlay (CT + Dose + ROI)
  ⬇
Output High-Resolution Image

## Core Module Documentation

### main.py
Main program executing the complete data processing pipeline
1.Loads CT and dose NIfTI files
2.Resamples dose data
3.Extracts key axial slices
4.Invokes visualization functions

### utils.py
Function utility library containing:
1.load_ct_nifti()` - Load CT data and associated parameters
2.load_dose_nifti()` - Load dose data
3.resample()` - Spatial resampling engine
4.build_roi_masks()` - ROI mask generation
5.visualize_three_planes_overlay()` - Three-plane visualization
6.extract_slices()` - Key slice extraction

### ct_dose_origin.py
Original algorithm version (reference use), primarily containing:
1.Direct DICOM processing functions
2.Early visualization schemes
3.Can serve as template for feature extensions

## Clinical Application Case

This project is applied to **nasopharyngeal carcinoma (NPC) radiotherapy plan verification**, including the following ROIs:

**GTV (Gross Tumor Volume)**: GTV70-4, GTVln66
**Clinical Target Volume**: CTV60, PTV6000
**Organ Constraints**: Brain Stem, Spinal Cord, Optic Chiasm, Optic Nerves, Parotids

## Output Examples

Generates four-panel medical imaging figure:
**Coronal View** - Top left
**Sagittal View** - Top right
**Axial View** - Bottom left
**Info Panel** - Bottom right

Each panel includes:
- Grayscale CT background
- Color-coded dose isodose lines
- Colored ROI contours (customizable)
- Coordinate scales and measurement information

## Data Format Specifications

### NIfTI File Format
1.Supports both compressed (.nii.gz) and uncompressed (.nii) formats
2.Metadata includes: affine transformation matrix, voxel spacing, origin coordinates

### RTSTRUCT File Format
1.DICOM Structured Report containing hand-drawn or automatically segmented ROI contours
2.Supports multiple contours and hole structures

### Coordinate Systems
- **DICOM Coordinate System**: Patient coordinate system (LPS)
- **NIfTI Coordinate System**: Usually RAS (determined by affine matrix)
- Program automatically handles coordinate transformation

## Configuration Parameters

Customizable in 'main.py':
python
dose_threshold_ratio=0.1      # Dose display threshold (relative to maximum dose)
z_idx, y_idx, x_idx           # Slice positions (auto-calculated or manual)
vmin, vmax                    # CT window center and width
dpi=150                       # Output resolution

Customizable in 'utils.py':
python
roi_map                       # ROI ID to name mapping
colors                        # ROI contour colors
linewidths                    # ROI contour line widths

## Known Issues & Future Improvements

- [ ] Support for direct DICOM format CT/Dose input (currently NIfTI only)
- [ ] Interactive slice selection UI
- [ ] Dose Volume Histogram (DVH) generation
- [ ] Batch processing for multiple patient datasets
- [ ] 3D visualization preview capability

## License
 
This project is intended for academic and research purposes.

**Development Tip**: Modify the `roi_map` and `colors` dictionaries in `utils.py` to adapt to different clinical scenarios (e.g., lung cancer, breast cancer, etc.).



