import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import os

def seq2kmer(seq, k):
    """
    Convert original sequence to kmers
    
    """
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers

def encode_seq(target,off_target):

    encode_dict ={
        ("A","A"):"A",
        ("A","C"):"Z",
        ("A","G"):"Y",
        ("A","T"):"X",
        ("C","C"):"C",
        ("C","A"):"W",
        ("C","G"):"V",
        ("C","T"):"U",
        ("G","G"):"G",
        ("G","A"):"S",
        ("G","C"):"R",
        ("G","T"):"L",
        ("T","T"):"T",
        ("T","A"):"Q",
        ("T","C"):"P",
        ("T","G"):"O",
    }
    
    new_seq = ""
    for char in zip(target,off_target):
        new_seq += encode_dict[char]
   
    return(new_seq)

 

def gen_random_split(seed,k):
    
    df1 = pd.read_csv("DeepCRISPR_dataset/k562.epiotrt",sep="\t",header=None)

    df1.columns = ["Id","Target Seq","Target CTCF","Target Dnase","Target H3K4me3","Target RRBS","Off-target Seq","Off-target CTCF","Off-target Dnase","Off-target H3K4me3","Off-target RRBS","Label"]

    k562_full = df1[["Target Seq","Off-target Seq","Label"]]
    k562_full["comb"]  = k562_full.apply(lambda x: encode_seq(x["Target Seq"],x["Off-target Seq"]),axis=1)

    print(k562_full.shape)
    k562_full = k562_full.drop_duplicates()
    print(k562_full.shape)


    k562_full["comb"] = k562_full.apply(lambda x: seq2kmer(x["comb"],k),axis=1)
    k562_full = k562_full.drop(columns=['Target Seq', 'Off-target Seq'])
    k562_full = k562_full[["comb","Label"]]

    df2 = pd.read_csv("DeepCRISPR_dataset/hek293t.epiotrt",sep="\t",header=None)

    df2.columns = ["Id","Target Seq","Target CTCF","Target Dnase","Target H3K4me3","Target RRBS","Off-target Seq","Off-target CTCF","Off-target Dnase","Off-target H3K4me3","Off-target RRBS","Label"]
    
    hek293t_full = df2[["Target Seq","Off-target Seq","Label"]]
    hek293t_full["comb"]  = hek293t_full.apply(lambda x: encode_seq(x["Target Seq"],x["Off-target Seq"]),axis=1)


    hek293t_full = hek293t_full.drop_duplicates()

    hek293t_full["comb"] = hek293t_full.apply(lambda x: seq2kmer(x["comb"],k),axis=1)
    hek293t_full = hek293t_full.drop(columns=['Target Seq', 'Off-target Seq'])
    hek293t_full = hek293t_full[["comb","Label"]]


    k562_trainval, k562_test = train_test_split(k562_full, test_size=0.2, random_state=seed, stratify=k562_full["Label"])
    hek293t_trainval, hek293t_test = train_test_split(hek293t_full, test_size=0.2, random_state=seed, stratify=hek293t_full["Label"])

    k562_train, k562_val = train_test_split(k562_trainval, test_size=0.1, random_state=seed, stratify=k562_trainval["Label"])
    hek293t_train, hek293t_val = train_test_split(hek293t_trainval, test_size=0.1, random_state=seed, stratify=hek293t_trainval["Label"])


    print("---------k562---------")
    print(k562_full.shape)
    print(k562_train.shape,k562_test.shape,k562_val.shape)
    print(len(k562_train[k562_train["Label"]==1]),len(k562_test[k562_test["Label"]==1]),len(k562_val[k562_val["Label"]==1]))

    print("--------hek293t-------")
    print(hek293t_full.shape)
    print(hek293t_train.shape,hek293t_test.shape,hek293t_val.shape)
    print(len(hek293t_train[hek293t_train["Label"]==1]),len(hek293t_test[hek293t_test["Label"]==1]),len(hek293t_val[hek293t_val["Label"]==1]))

    combined_train = pd.concat([k562_train, hek293t_train])
    combined_val = pd.concat([k562_val, hek293t_val])

    print(combined_train.shape)
    print(combined_val.shape)
    print(combined_train)

   
    if not os.path.exists("data/data_newsplit3_" + str(seed) + "/hek293t_test"):
        os.makedirs("data/data_newsplit3_" + str(seed) + "/hek293t_test", exist_ok=True)
    if not os.path.exists("data/data_newsplit3_" + str(seed) + "/k562_test"):
        os.makedirs("data/data_newsplit3_" + str(seed) + "/k562_test", exist_ok=True)    


    hek293t_test.to_csv("data/data_newsplit3_" + str(seed) + "/hek293t_test/dev.tsv",index=False, sep="\t")
    k562_test.to_csv("data/data_newsplit3_" + str(seed) + "/k562_test/dev.tsv",index=False, sep="\t")

    combined_train.to_csv("data/data_newsplit3_" + str(seed) + "/train.tsv",index=False, sep="\t")
    combined_val.to_csv("data/data_newsplit3_" + str(seed) + "/dev.tsv",index=False, sep="\t")

    return

gen_random_split(42,3)