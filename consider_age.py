import json
import pandas as pd
import os
import matplotlib.pyplot as plt


def get_id_age(file_folder):
    case_age = []
    for filename in os.listdir(file_folder):
        file_path = f"/home/bnt4me/MasterTh/gdc_data/info/{filename}"

        f = open(file_path)
        data = json.load(f)
        for i in data['hits']:
            try:
                age = i["diagnoses"][0]["age_at_diagnosis"] / 365
            except Exception as err:
                print(err, i["diagnoses"][0]["age_at_diagnosis"])
                age = 0
            case_id = i["fpkm_files"][0]["file_id"]+ ".tsv"
            case_age.append((case_id, age))
        print(len(case_age))
    return case_age


case_age = get_id_age('/home/bnt4me/MasterTh/gdc_data/info')
kk = []
# print(case_age)
for d in case_age:
   kk.append(d[1])

def sort_data(max_age):
    full_data = pd.read_csv("./gdc_data/last_file.csv", delimiter="\t", index_col=0, low_memory=False)
    print(len(full_data))
    for d in case_age:
        if d[1] < max_age:
            # full_data.drop("9c2ffccc-c4c8-4f87-bcd8-8ce099e31b1b.tsv")
            try:
                full_data = full_data.drop(f"{d[0]}")
            except:
                print("No data found")

    print(len(full_data))
    print("_______________")
    full_data.to_csv(f"./gdc_data/last_file_age{max_age}.csv", sep='\t')

#sort_data(50)

def create_freq_plot(x, main_title, file_path):
    num_bins = 50
    fig, ax = plt.subplots()

    n, bins, patches = ax.hist(x, num_bins,
                               # density=True,
                               color='blue',
                               alpha=0.8)

    ax.set_xlabel('Age')
    ax.set_ylabel('Frequency')
    ax.set_title(main_title)

    # plt.show()
    plt.savefig(file_path, dpi=500)
    print("Fig was saved..")


create_freq_plot(kk,"","./file_p.svg")


# full_data = pd.read_csv("./gdc_data/last_file.csv", delimiter="\t", index_col=0, low_memory=False)
# print(full_data.index)