from scipy.io import loadmat
from pathlib import Path
import pandas as pd

def extract_vectors(vector_file, output_file, model_drag_file):
    # Load the vector file
    vector_data = loadmat(vector_file)
    keys = vector_data.keys()
    print(keys) #dict_keys(['__header__', '__version__', '__globals__', 'model_names', 'emb'])
    name_list = [element for element in vector_data['model_names']]
    vector_list = [vector for vector in vector_data['emb']]

    # Load the model drag file in csv format
    model_drag_data = pd.read_csv(model_drag_file)
    #show the header 
    print(model_drag_data.head())
    #access the column of the model name
    model_name_list = model_drag_data['model_name']
    drag_list = model_drag_data['drag_coefficient']
    
    #store i, name, vector, and drag in a csv file where vector is np.array and drag is float  
    #create a new csv file
    csv_file = output_file
    csv_file.touch()
    #write the header
    with open(csv_file, 'w') as f:  
        #write the data   
        i = 0
        f.write('i,name,')
        for j in range(len(vector_list[0])):
                f.write(f'dim_{j+1},')
        f.write('drag\n')
        
        for name in model_name_list:
            if name in name_list: 
                index = name_list.index(name)    
                vector = vector_list[index]
                drag = drag_list[i]
                f.write(f'{i},{name},')
                #vector is an np array and needs to be stored one value by one value
                for j in range(len(vector)):
                    f.write(f'{vector[j]},')
                f.write(f'{round(drag, 3)}\n')
                i += 1
    
# body_mat_file = recon_folder / "body" / "recover.mat"
# body_data = loadmat(body_mat_file)
# part_keys = body_data.keys()
# print(part_keys) #dict_keys(['__header__', '__version__', '__globals__', 'model_names', 'emb'])
# body_list = [vector for vector in body_data['emb']]
# print(len(body_list[0]))

# # plane models
# import mat73
# plane_feature_file = this_folder / "plane_vaefeature.mat"
# plane_feature_data = mat73.loadmat(plane_feature_file)
# plane_feature_keys = plane_feature_data.keys()
# print(plane_feature_keys)
# model_list = [name for name in plane_feature_data['modelname']]
# print(len(model_list))


if __name__ == "__main__":
    #car models
    this_folder = Path(__file__).resolve().parent
    recon_folder = this_folder / "carVAE_64_128"
    SPVAE_mat_file =  recon_folder / "recover_sym.mat"
    models_with_drags_file = this_folder / "overlapped_names_drags.csv"
    spvae_file = this_folder / "spvae_vectors_drags.csv"
    extract_vectors(SPVAE_mat_file, spvae_file, models_with_drags_file)
    