import numpy as np
import pandas as pd

def split_dataset(train_size, csv_path_in, csv_path_out_train, csv_path_out_test):
    df = pd.read_csv(csv_path_in, sep=";")
    N = len(df)
    N_blood = 0
    unique_ids_dict = {}
    blood_ids = set()
    for (filename, label_str) in zip(df["filename"], df["finding_class"]):
        unique_id = filename.split("_")[0]
        label = 0
        if label_str.startswith('Blood'):
            N_blood += 1
            label = 1
            blood_ids.add(unique_id)  
        if(unique_id not in unique_ids_dict):
            unique_ids_dict[unique_id] = [0, 0]
        unique_ids_dict[unique_id][label] += 1
    sorted_blood_ids = sorted(list(blood_ids), key=lambda x: unique_ids_dict[x][1], reverse=True)
    N_normal = N-N_blood

    train_IDs = []
    test_IDs = []
    N_blood_train = 0
    N_normal_train = 0

    for ID in sorted_blood_ids:
        if(N_blood_train < int(N_blood*train_size)):
            train_IDs.append(ID)
            N_blood_train += unique_ids_dict[ID][1]
            N_normal_train += unique_ids_dict[ID][0]
        else:
            test_IDs.append(ID)

    for ID, sample_numbers in unique_ids_dict.items():
        if(ID in sorted_blood_ids):
            continue
        if(N_normal_train < int(N_normal*train_size)):
            train_IDs.append(ID)
            N_normal_train += sample_numbers[0]
        else:
            test_IDs.append(ID)
    
    train_dict = {
        "filename" : [],
        "Label" : []
    }

    test_dict = {
        "filename" : [],
        "Label" : []
    }

    for (filename, label_str) in zip(df["filename"], df["finding_class"]):
        unique_id = filename.split("_")[0]
        if(unique_id in train_IDs):
            train_dict["filename"].append(filename)
            train_dict["Label"].append(1 if label_str.startswith('Blood') else 0)
        else:
            test_dict["filename"].append(filename)
            test_dict["Label"].append(1 if label_str.startswith('Blood') else 0)#
    
    df_train = pd.DataFrame(train_dict)
    df_test = pd.DataFrame(test_dict)

    df_train.to_csv(csv_path_out_train, index=False)
    df_test.to_csv(csv_path_out_test, index=False)

if __name__ == "__main__":
    csv_path_in = r"C:\Users\JadHa\Desktop\Uni\DLMI-Project\kvasir-capsule-labeled-images\labelled_images\metadata.csv"
    csv_path_out_train =r"C:\Users\JadHa\Desktop\Uni\DLMI-Project\kvasir-capsule-labeled-images\dataset_train.csv"
    csv_path_out_test =r"C:\Users\JadHa\Desktop\Uni\DLMI-Project\kvasir-capsule-labeled-images\dataset_test.csv"
    split_dataset(train_size=0.8, csv_path_in=csv_path_in, csv_path_out_train=csv_path_out_train, csv_path_out_test=csv_path_out_test)