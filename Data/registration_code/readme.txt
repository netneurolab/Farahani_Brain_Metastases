





antsRegistrationSyNQuick.sh -d 3 -f T1w.nii.gz \
  -m /media/afarahani/Expansion1/Cancer_data/UCSF_brain_Glioblastoma/100013/100013_time1_t1.nii.gz -o output_ -x /media/afarahani/Expansion1/Cancer_data/UCSF_brain_Glioblastoma/100013/100013_time1_seg.nii.gz


# Weight image for FLIRT: 1 outside lesion, 0 inside
fslmaths /media/afarahani/Expansion1/Cancer_data/UCSF_brain_Glioblastoma/100013/100013_time1_seg.nii.gz -sub /media/afarahani/Expansion1/Cancer_data/UCSF_brain_Glioblastoma/100013/100013_time1_t1.nii.gz -thr 0 inweight.nii

flirt -in T1_brain.nii -ref T1w.nii.gz \
      -omat affine.mat -out T1_aff.nii -dof 12 -bins 256 -cost mutualinfo \
      -inweight inweight.nii

# Nonlinear with masks (FNIRT):
fnirt --in=T1.nii --aff=affine.mat --ref=MNI152_T1_1mm.nii.gz \
      --inmask=brain_mask.nii --refmask=MNI152_brain_mask.nii.gz \
      --config=T1_2_MNI152_2mm --iout=T1_fnirt.nii --cout=warpcoef.nii

