from pathlib import Path
import csv
import random

'''
all 1597 models have drags 
but the computer cannot calculate the feature file for 1597 models
random select 1000 models
'''

this_folder = Path(__file__).resolve().parent
my_model_file = this_folder / "plane_models_1597.txt"
drag_model_file = this_folder / "plane_drags.csv"

#open txt file and read lines to a list
with open(my_model_file) as f:
    lines = f.readlines()
    #strip whitespace and newlines
    my_model_list = [line.strip()[:-9] for line in lines]
    # print(my_model_list)

#open a csv file and read the column "Airplane ID"
drag_model_list = []
with open(drag_model_file) as f:
    # Create a CSV reader object
    reader = csv.reader(f)
    # Iterate over the rows
    for row in reader:
        # Output the second column
        name = row[0].strip()
        drag_model_list.append(name)

i = 1 
for model in my_model_list:
    if model in drag_model_list:
        # print(model)
        # print(i)
        i += 1

#randomly select non-repeating models from the list
random.seed(1)
n = 1200
random_list = random.sample(my_model_list, n)
print(random_list)

#store random_list in a txt file
with open(this_folder / f"random_models_{n}.txt", "w") as f:
    for model in random_list:
        f.write(model + "\n")


