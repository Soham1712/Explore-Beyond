import os

image_class_mapping = {
        # 0: 'other',
        1: 'crater', # 6034
        2: 'dark_dune', # 1232
        3: 'slope_streak', # 2366
        4: 'bright_dune', # 1769
        5: 'impact_ejecta', # 518
        # 6: 'swiss_cheese',
        # 7: 'spider'
}

path = r"./Datasets/training-data/"

for file in os.listdir(path):
    dir_path = os.path.join(path, file)
    count = 0
    for i in os.listdir(dir_path):
        count += 1
    print(f"{file} =  {count}")