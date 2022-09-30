# Author: Xinwen Xu
# Date: 2022/8/1

from operator import index
import pandas as pd
import numpy as np

def read_data():
    # load all the corpora
    data1 = pd.read_excel("./data/Corpus_traduction_indiv/traduction_seuils_persos_17_12_21.xlsx")
    data1.insert(data1.shape[1], "type", "traduction")

    data2 = pd.read_excel("./data/Corpus_experimental_adultes_indiv/formulation_seuils_persos_07_12_21.xlsx")
    data2.insert(data2.shape[1], "type", "formulation")

    data3 = pd.read_excel("./data/Corpus_experimental_adultes_indiv/planification_seuils_persos_07_12_21.xlsx")
    data3.insert(data3.shape[1], "type", "planification")

    data4 = pd.read_excel("./data/Corpus_experimental_adultes_indiv/revision_seuils_persos_07_12_21.xlsx")
    data4.insert(data4.shape[1], "type", "resision")

    data5 = pd.read_excel("./data/Corpus_enfants_seuils_indiv/enfants_seuils_persos_07_12_21.xlsx")
    data5.insert(data5.shape[1], "type", "enfants")

    data6 = pd.read_excel("./data/Corpus_dossiers_academiques_indiv/corpus_académique_seuils_persos_07_12_21.xlsx")
    data6.insert(data6.shape[1], "type", "academiques")

    data7 = pd.read_excel("./data/Corpus_Rapports_sociaux_indiv/rapports_seuils_persos_07_12_21_original.xlsx")
    data7.insert(data7.shape[1], "type", "rapports")

    # Pick out the columns needed
    cols = ["ID","burst","burst_len","raw_burst","n_chars","pause","pct_pause","type"]
    data = pd.concat([data1[cols],data2[cols],data3[cols],data4[cols],data5[cols],data6[cols],data7[cols]],ignore_index=True)
    len_ori = len(data)             # original number of data
    return data, len_ori

def processing(data):
    # Drop null raw bursts
    data.dropna(subset=["raw_burst","pause","pct_pause"],inplace=True)

    # Pad null bursts which are not null raw bursts
    for i,row in data.iterrows():
        if len(str(row["burst"]).split()) == 0:
            data.drop(i, inplace=True)

        if pd.isnull(row["burst"]):
            length = row["burst_len"]
            if length<=0:
                data.loc[i,"burst"] = "<delete>"
            elif length>0:
                data.loc[i,"burst"] = "<blank>"
        
        # "¦" means skipping to another location and entering, so we replace it by blank
        if "¦" in str(row["burst"]):
            data.loc[i,"burst"] = row["burst"].replace("¦","<jump> ")

    data.reset_index(drop=True, inplace=True)
    len_use = len(data)             # data to be used
    return data, len_use


if __name__ == "__main__":
    data_ori, len_ori = read_data()
    data_use, len_use = processing(data_ori)
    data_use.to_excel("data.xls", engine='openpyxl')
    print(f'There are {len_ori} data originally, after processing, {len_use} are kept after processing.')
    print(data_use.info())