from scipy.io import loadmat
from pathlib import Path
import csv

#car models
this_folder = Path(__file__).resolve().parent
recon_folder = this_folder / "cars" / "carVAE_64_128"
SPVAE_mat_file =  recon_folder / "recover_sym.mat"
drag_file = this_folder / "cars" / "car_drags.csv"

# Load the mat file
SPVAE_data = loadmat(SPVAE_mat_file)

# print(data)

SPVAE_keys = SPVAE_data.keys()
print(SPVAE_keys)

name_list1 = [element.strip() for element in SPVAE_data['model_names']]
# print(name_list1)
# print(len(name_list)) #1162

name_list2 = []
# Open the CSV file
with open(str(drag_file), newline='') as csvfile:
    # Create a CSV reader object
    reader = csv.reader(csvfile)
    
    # Iterate over the rows
    for row in reader:
        # Output the second column
        content = row[1].strip()
        if '_aug' not in content and '_flip' not in content:
            name_list2.append(content)
# print(name_list2)

overlapped_name_list = []
for name in name_list1:
    if name in name_list2:
        overlapped_name_list.append(name)
    
print(len(overlapped_name_list))

