from scipy.io import loadmat
from pathlib import Path

#car models
this_folder = Path(__file__).resolve().parent
recon_folder = this_folder / "06222308_8711joint_0-l0_500.0-l2_10.0-l3_1.0-l4_0.001-model_car-trcet_0.75" / "recon40000"
SPVAE_mat_file =  recon_folder / "recover_sym.mat"
body_mat_file = recon_folder / "body" / "recover.mat"

# Load the mat file
SPVAE_data = loadmat(SPVAE_mat_file)
body_data = loadmat(body_mat_file)

# print(data)

SPVAE_keys = SPVAE_data.keys()
print(SPVAE_keys)

part_keys = body_data.keys()
print(part_keys)

name_list = [element for element in SPVAE_data['model_names']]
print(name_list)
print(len(name_list))

SPVAE_list = [vector for vector in SPVAE_data['emb']]
print(len(SPVAE_list[0]))

body_list = [vector for vector in body_data['emb']]
print(len(body_list[0]))

# plane models
import mat73
plane_feature_file = this_folder / "plane_vaefeature.mat"
plane_feature_data = mat73.loadmat(plane_feature_file)
plane_feature_keys = plane_feature_data.keys()
print(plane_feature_keys)
model_list = [name for name in plane_feature_data['modelname']]
print(len(model_list))

# car_feature_file = this_folder / "car_vaefeature.mat"
# car_feature_data = mat73.loadmat(car_feature_file)
# car_feature_keys = car_feature_data.keys()
# print(car_feature_keys)