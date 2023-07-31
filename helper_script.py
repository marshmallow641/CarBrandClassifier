from PIL import Image
import os
import pandas as pd
import torch
import csv

def resize_training_images(img_dir="", new_dir="resized images", annotation_file="", width=255, height=255):
    annotation_file = pd.read_csv(annotation_file)

    parent_dir = new_dir

    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)

    directories: list = []

    for i in os.listdir(img_dir):
        if os.path.isdir(i):
            print(i)
            os.makedirs(f".\\{parent_dir}\\{i}")

            # directories.append(i)
        else:
            continue

    for idx in range(len(annotation_file)):
        img_path_1 = os.path.join(str(img_dir), str(annotation_file.iloc[idx, 2]))
        img_path_2 = os.path.join(str(img_path_1), str(annotation_file.iloc[idx, 0]))

        altered_image_path_1 = os.path.join(parent_dir, str(annotation_file.iloc[idx, 2]))
        altered_image_path_2 = os.path.join(str(altered_image_path_1), str(annotation_file.iloc[idx, 0]))

        img = Image.open(img_path_2)
        img = img.resize((width, height))

        img.save(altered_image_path_2, "PNG")

def update_console():
    pass

def dataset_split(dir="", train_size=0.7, dataset_in: list=[]):
    test_size = 1 - train_size
    dataset:list = []

    test_dir = f"{dir}_test"

    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    train_dir:list = []

    '''if dataset_in:
        dataset = dataset_in

        training_data = dataset[int(train_size * len(dataset))]
        testing_data = dataset[int(test_size * len(dataset))]

        return training_data, testing_data

    elif not dataset_in:'''
    for i in os.listdir(dir):
        if os.path.isdir(i):
            for j in os.listdir(i):
                if os.path.isfile(j):
                    dataset.append(os.path.join(i, j))
        else:
            dataset.append(os.path.join(dir, i))

    training_data = dataset[0:int(train_size * len(dataset))]
    #training_data = torch.tensor(data=training_data)

    testing_data = dataset[int(train_size * len(dataset)):int(test_size * len(dataset))]
    #testing_data = torch.tensor(data=testing_data)

    for i in os.listdir(dir):
        if os.path.isdir(i):
            for j in os.listdir(i):
                #if os.path.isfile(j):
                train_dir.append(i)

                for k in training_data:
                    Image.open(k).save(os.path.join(i, j), "PNG")



        for i in train_dir:
            os.mkdir(f"{test_dir}\\{i}")

        for i in os.listdir(test_dir):
            if os.path.isdir(i):
                for j in os.listdir(i):
                    #if os.path.isfile(j):

                    for k in testing_data:
                        k.save(os.path.join(i, j), "PNG")

        #return training_data, testing_data

def get_data_file_paths(directory_name="", path_code: int=0):
    filenames_list: list = []
    relative_path_list: list = []
    absolute_path_list: list = []

    # Relative and absolute paths for the images
    try:
        absolute_file_path = f"{os.getcwd()}\\{directory_name}"

        directory_contents: list = os.listdir(directory_name)

        for contents in directory_contents:
            filenames_list.append(contents)
            relative_path_list.append(f"{directory_name}\\{contents}")
            absolute_path_list.append(f"{absolute_file_path}\\{contents}")

        if path_code == 0:
            return relative_path_list
        elif path_code == 1:
            return absolute_path_list
        elif path_code == 2:
            return filenames_list

    except FileExistsError as fe:
        print(f"{fe.__str__()}")

def generate_annotations_csv(filename, dir, headers: list=[], path_code: int=0):
    with open(filename, "w") as f:
        class_id = 0
        counter = 0
        directories = get_data_file_paths(dir)

        writer = csv.writer(f)

        writer.writerow(headers)

        for i in directories:

            current_files = get_data_file_paths(i, path_code)
            for file in current_files:
                counter += 1
                category = i[len(dir) + 1:]
                im = Image.open(os.path.join(i, file))

                row = [file, class_id, category, i, im.size] # Creates rows with the file's name, the class id, the category (directory name), and the filee's relative path
                writer.writerow(row)

            class_id += 1

        f.close()
        print(f"Found {counter} files")
