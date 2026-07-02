#------------------------------------------------------------------------------
# Numeric constants
#------------------------------------------------------------------------------

num_subcort_voxels = 31870
num_cort_vertices_noMW = 59412
num_vertices_voxels = 91282
num_cort_vertices_withMW = 64984
num_vertices_gifti = 32492

nnodes_Schaefer = 400
nnodes_Schaefer_S1 = 416
nnodes_Schaefer_S4 = 454

nnodes_S1 = 16
nnodes_S4 = 54
nnodes_half_subcortex_S4 = 27

#------------------------------------------------------------------------------
# Paths needed for the analysis
#------------------------------------------------------------------------------

# Path for HCP-wb_command
path_wb_command = '/home/afarahani/Downloads/workbench/bin_linux64/'

# Paths where results, data, and figures are stored for this project
my_pc = '/home/afarahani/Desktop/met_data/'

path_results     = my_pc + 'git_results/'
path_data        = my_pc + 'Data/'

path_reg         = path_data + 'registration_code/'
path_MNI09a      = path_data + 'mni_icbm152_nlin_sym_09a/'
path_MNI09c      = path_data + 'mni_icbm152_nlin_sym_09c/'
path_nifti       = path_data + 'BM_files/'
path_diedrichsen = path_data + 'Diedrichsen_2009/'

path_perfusion   = '/home/afarahani/Desktop/blood_annotation/Farahani_blood_perfusion/results/'
path_FC          = my_pc + 'FC/'

path_medialwall  = path_data + 'medialwall/' # Mask for CIFTI medial wall vertices
path_templates   = path_data + 'templates/'  # Path to template CIFTI, I load this and save my data on this template map
path_surface     = path_data + 'GA_surface_files/' # Path for HCP-surface files - these are used in wb_command when visualizing surface maps for figures.
path_MNI06       = path_data + 'MNI152NLin6Asym/'  # Template of interest
path_file        = path_data + 'files/'
path_atlas       = path_data + 'schaefer_tian/'
path_network     = path_data + 'cortical_networks/'
path_validation  = path_data + 'replication_data/'
path_csv         = path_data + 'csv/'

#------------------------------------------------------------------------------
# END
