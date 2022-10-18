import json
import pandas as pd
import os

pd.set_option('display.max_columns', None)

def get_gen_list():
    gene_list = list(pd.read_csv("../../../../gdc_data/gene_breast.tsv", delimiter="\t", low_memory=False)["ENSG"])
    gene_list.append("tumor_stage")
    return gene_list

def get_data_df():
    return pd.read_csv("../../../../gdc_data/last_file.csv", delimiter="\t", index_col=0, low_memory=False)

def filter_by_col(pd_df, col_list):
    pd_df1 = pd_df.copy()
    col_names = list(pd_df1.columns)
    new_col_names = {}
    for k in col_names:
        new_col_names[k] = k[0:15]
    # pd_df.rename(columns=new_col_names)
    pd_df1.columns = pd_df1.columns.to_series().map(new_col_names)

    pd_df1 = pd_df1.filter(items=col_list)
    return pd_df1

def save_data(pd_dataframe):
    pd_dataframe.to_csv(f"./gdc_data/full_data_genes.csv", sep='\t')




def main():
    data = get_data_df()
    col_list = get_gen_list()
    print(col_list)
    new_data = filter_by_col(data, col_list)
    print(new_data)
    save_data(new_data)
    # print(new_data)


if __name__ == "__main__":
    main()