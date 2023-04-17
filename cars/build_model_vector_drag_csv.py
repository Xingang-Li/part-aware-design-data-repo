from scipy.io import loadmat
from pathlib import Path
import pandas as pd
import os

'''
Form csv files: model names, vectors (e.g., SPVAE vectors, part-VAE vectors, etc.), drags 

Output examples: spvae_vectors_drags.csv
'''

#extract part vectors from mat files and concatenate them to store them in one csv file
def extract_part_vectors(folder, output_file=None, category='car'):
    if category == 'car':
        parts = ['body', 'left_mirror', 'right_mirror', 'left_front_wheel', 
                 'right_front_wheel', 'left_back_wheel',  'right_back_wheel']
    k = 0
    for part in parts[:1]:
        subfolder_path = os.path.join(folder, part)
        for file in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file)
            if file_path.endswith(".mat"):
                mat_data = loadmat(file_path)
                # keys = mat_data.keys()
                name_list = [element.strip() for element in mat_data['model_names']] #model names in my training set
                vector_list = [vector for vector in mat_data['emb']]

                #store i, name, vector, and drag in a csv file where vector is np.array and drag is float  
                #create a new csv file
                csv_file = output_file
                csv_file.touch()
                #write the header
                with open(csv_file, 'a') as f:  
                    #write the data   
                    i = 0
                    f.write('i,name,')
                    for j in range(len(vector_list[0])):
                            f.write(f'dim_{j+k*len(vector_list[0])},')
                    
                    for name in name_list:
                        index = name_list.index(name)    
                        vector = vector_list[index]
                        f.write(f'{i},{name},')
                        #vector is an np array and needs to be stored one value by one value
                        for l in range(len(vector)):
                            f.write(f'{vector[l]},')
                        i += 1
        k += 1



def form_model_vector_drag(vector_file, model_drag_file, output_file):
    # Load the vector file
    vector_data = loadmat(vector_file)
    keys = vector_data.keys()
    print(keys) #dict_keys(['__header__', '__version__', '__globals__', 'model_names', 'emb'])
    name_list = [element.strip() for element in vector_data['model_names']] #model names in my training set
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
    

if __name__ == "__main__":

    #car models
    this_folder = Path(__file__).resolve().parent
    recon_folder = this_folder / "carVAE_64_128"
    models_with_drags_file = this_folder / "overlapped_names_drags.csv"

    # #SPVAE vectors
    # SPVAE_mat_file =  recon_folder / "recover_sym.mat"
    # spvae_file = this_folder / "model_vector_drag" / "spvae_vectors_drags.csv"
    # form_model_vector_drag(vector_file=SPVAE_mat_file, model_drag_file=models_with_drags_file, output_file=spvae_file)

    # #part-VAE vectors
    # body_mat_file = recon_folder / "body" / "recover.mat"
    # body_file = this_folder / "model_vector_drag" / "body_vectors_drags.csv"
    # form_model_vector_drag(vector_file=body_mat_file, model_drag_file=models_with_drags_file, output_file=body_file)

    #
    parts_vectors_csv = this_folder / "model_vector_drag" / "all_parts_vectors_drags.csv"
    extract_part_vectors(recon_folder, parts_vectors_csv, category='car')

    