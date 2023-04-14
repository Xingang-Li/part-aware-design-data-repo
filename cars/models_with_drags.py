from scipy.io import loadmat
from pathlib import Path
import csv

#car models
this_folder = Path(__file__).resolve().parent
recon_folder = this_folder / "carVAE_64_128"
SPVAE_mat_file =  recon_folder / "recover_sym.mat"
drag_file = this_folder / "car_drags.csv"

# Load the mat file
SPVAE_data = loadmat(SPVAE_mat_file)
# print(data)

SPVAE_keys = SPVAE_data.keys()
print(SPVAE_keys)
name_list1 = [element.strip() for element in SPVAE_data['model_names']]
# print(name_list1)
# print(len(name_list)) #1162

name_list2 = []
drag_list = []
# Open the CSV file
with open(str(drag_file), newline='') as csvfile:
    # Create a CSV reader object
    reader = csv.reader(csvfile)
    # Iterate over the rows
    for row in reader:
        # Output the second column
        name = row[1].strip()
        drag = row[2].strip()
        if '_aug' not in name and '_flip' not in name:
            name_list2.append(name)
            drag_list.append(drag)

overlapped_name_list = []
overlapped_drag_list = []
for name in name_list1:
    if name in name_list2:
        overlapped_name_list.append(name)
        #index of name in name_list2
        index = name_list2.index(name)
        drag = drag_list[index]
        overlapped_drag_list.append(drag)

print(len(overlapped_drag_list))

#store the names of overlapped models
with open (this_folder / "overlapped_name_list.txt", "w") as f:
    for name in overlapped_name_list:
        f.write(name + "\n")  

#csv file contains model names and corresponding drag coefficients
with open(this_folder / "overlapped_names_drags.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["number", "model_name", "drag_coefficient"])
    for i in range(len(overlapped_name_list)):
        writer.writerow([i, overlapped_name_list[i], overlapped_drag_list[i]])




