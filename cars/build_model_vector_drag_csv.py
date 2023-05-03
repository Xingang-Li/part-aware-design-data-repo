from scipy.io import loadmat
from pathlib import Path
import pandas as pd
import os
import csv
import numpy as np
from tqdm import tqdm


'''
Form csv files: model names, vectors (e.g., SPVAE vectors, part-VAE vectors, etc.), drags 

Output examples: spvae_vectors_drags.csv
'''

#extract all part vectors from mat files and concatenate them to store them in one csv file
#csv file: name, vector (e.g., dim 64*7=448)
def extract_part_vectors(folder, output_file=None, category='car'):
    if category == 'car':
        parts = ['body', 'left_mirror', 'right_mirror', 'left_front_wheel', 
                 'right_front_wheel', 'left_back_wheel',  'right_back_wheel']
        
    all_vector_list = []
    for part in parts[:]:
        subfolder_path = os.path.join(folder, part)
        for file in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file)
            if file_path.endswith(".mat"):
                mat_data = loadmat(file_path)
                # keys = mat_data.keys()
                name_list = [element.strip() for element in mat_data['model_names']] #model names in my training set
                vector_list = [vector for vector in mat_data['emb']]
                all_vector_list.append(vector_list)
    print(len(all_vector_list))
    for i, item in enumerate(all_vector_list):
        if i == 0:
            vector_array = np.array(item)
        else:
            vector_array = np.concatenate((vector_array, item), axis=1)
    print(vector_array.shape)

    #create a new csv file
    output_file.touch()
    #write the header
    with open(output_file, 'w') as f:  
        #write the data   
        for name in name_list:
            index = name_list.index(name)    
            vector = vector_array[index][:]
            f.write(f'{name},')
            #vector is an np array and needs to be stored one value by one value
            for i in range(len(vector) - 1):
                f.write(f'{vector[i]},')
            f.write(f'{vector[-1]}\n')


#form a csv file: i, name, vector, drag
def form_model_vector_drag(vector_file, model_drag_file, output_file):
    # Load the vector file
    if str(vector_file).endswith(".mat"):
        vector_data = loadmat(vector_file)
        keys = vector_data.keys()
        print(keys) #dict_keys(['__header__', '__version__', '__globals__', 'model_names', 'emb'])
        name_list = [element.strip() for element in vector_data['model_names']] #model names in my training set
        vector_list = [vector for vector in vector_data['emb']]
    elif str(vector_file).endswith(".csv"):
        name_list = []
        vector_list = []
        with open(vector_file, newline='') as csvfile:
            # Create a CSV reader object
            reader = csv.reader(csvfile)
            # Iterate over the rows
            for row in reader:
                # Output the second column
                name = row[0].strip()
                name_list.append(name)
                vector = np.array([float(value) for value in row[1:]])
                vector_list.append(vector)
                
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

    # #obtain all part vectors from mat files and concatenate them to store them in one csv file
    # parts_vectors_csv = this_folder / "all_parts_vectors.csv"
    # extract_part_vectors(recon_folder, parts_vectors_csv, category='car')

    # #SPVAE vectors
    # SPVAE_mat_file =  recon_folder / "recover_sym.mat"
    # spvae_file = this_folder / "model_vector_drag" / "spvae_vectors_drags.csv"
    # form_model_vector_drag(vector_file=SPVAE_mat_file, model_drag_file=models_with_drags_file, output_file=spvae_file)

    # #part-VAE vectors
    # #body
    # body_mat_file = recon_folder / "body" / "recover.mat"
    # body_file = this_folder / "model_vector_drag" / "body_vectors_drags.csv"
    # form_model_vector_drag(vector_file=body_mat_file, model_drag_file=models_with_drags_file, output_file=body_file)

    # #all parts
    # all_part_vector_file = this_folder / "all_parts_vectors.csv"
    # all_part_output_file = this_folder / "model_vector_drag" / "all_parts_vectors_drags.csv"
    # form_model_vector_drag(vector_file=all_part_vector_file, model_drag_file=models_with_drags_file, output_file=all_part_output_file)
    
    #3DPG vectors, three configurations
    dimension_list = [5040, 10240, 20000]
    for dimension in tqdm(dimension_list):
        vector_file = this_folder / f"{dimension}_vectors_sdf_low_res50200.csv"
        output_file = this_folder / "model_vector_drag" / f"{dimension}_vectors_drags_sdf_low_res50200.csv"
        form_model_vector_drag(vector_file=vector_file, model_drag_file=models_with_drags_file, output_file=output_file)

    