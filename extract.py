from lingfeat import extractor
import pandas as pd
from collections import defaultdict
import os
import tqdm
import csv
import logging
import math
import spacy
from alive_progress import alive_bar

dir_path = os.path.dirname(os.path.realpath(__file__))
target_datas = [
#"Research_Data/onestop.0.test",
#"Research_Data/onestop.1.test",
#"Research_Data/onestop.2.test",
#"Research_Data/onestop.3.test",
#"Research_Data/onestop.4.test",
#"Research_Data/onestop.0.train",
#"Research_Data/onestop.1.train",
#"Research_Data/onestop.2.train",
#"Research_Data/onestop.3.train",
#"Research_Data/onestop.4.train",
#"Research_Data/weebit.0.test",
#"Research_Data/weebit.1.test",
#"Research_Data/weebit.2.test",
#"Research_Data/weebit.3.test",
#"Research_Data/weebit.4.test",
#"Research_Data/weebit.0.train",
#"Research_Data/weebit.1.train",
#"Research_Data/weebit.2.train",
#"Research_Data/weebit.3.train",
#"Research_Data/weebit.4.train",
#"Research_Data/cambridge.0.test",
#"Research_Data/cambridge.1.test",
#"Research_Data/cambridge.2.test",
#"Research_Data/cambridge.3.test",
#"Research_Data/cambridge.4.test",
#"Research_Data/cambridge.0.train",
#"Research_Data/cambridge.1.train",
#"Research_Data/cambridge.2.train",
#"Research_Data/cambridge.3.train",
#"Research_Data/cambridge.4.train",
#"Research_Data/weebit20p.0.test",
#"Research_Data/weebit20p.1.test",
#"Research_Data/weebit20p.2.test",
#"Research_Data/weebit20p.3.test",
#"Research_Data/weebit20p.4.test",
#"Research_Data/weebit20p.0.train",
#"Research_Data/weebit20p.1.train",
#"Research_Data/weebit20p.2.train",
#"Research_Data/weebit20p.3.train",
#"Research_Data/weebit20p.4.train",
#"Research_Data/weebit40p.0.test",
#"Research_Data/weebit40p.1.test",
#"Research_Data/weebit40p.2.test",
#"Research_Data/weebit40p.3.test",
#"Research_Data/weebit40p.4.test",
#"Research_Data/weebit40p.0.train",
#"Research_Data/weebit40p.1.train",
#"Research_Data/weebit40p.2.train",
#"Research_Data/weebit40p.3.train",
#"Research_Data/weebit40p.4.train",
#"Research_Data/weebit60p.0.test",
#"Research_Data/weebit60p.1.test",
#"Research_Data/weebit60p.2.test",
#"Research_Data/weebit60p.3.test",
#"Research_Data/weebit60p.4.test",
#"Research_Data/weebit60p.0.train",
#"Research_Data/weebit60p.1.train",
#"Research_Data/weebit60p.2.train",
#"Research_Data/weebit60p.3.train",
#"Research_Data/weebit60p.4.train",
#"Research_Data/weebit80p.0.test",
#"Research_Data/weebit80p.1.test",
#"Research_Data/weebit80p.2.test",
#"Research_Data/weebit80p.3.test",
#"Research_Data/weebit80p.4.test",
#"Research_Data/weebit80p.0.train",
#"Research_Data/weebit80p.1.train",
#"Research_Data/weebit80p.2.train",
#"Research_Data/weebit80p.3.train",
#"Research_Data/weebit80p.4.train",
"Research_Data/weebit10.0.train",
"Research_Data/weebit20.0.train",
"Research_Data/weebit30.0.train",
"Research_Data/weebit40.0.train",
"Research_Data/weebit50.0.train",
"Research_Data/weebit60.0.train",
"Research_Data/weebit70.0.train",
"Research_Data/weebit80.0.train",
"Research_Data/weebit90.0.train",
"Research_Data/weebit100.0.train",
"Research_Data/weebit110.0.train",
"Research_Data/weebit120.0.train",
"Research_Data/weebit130.0.train",
"Research_Data/weebit140.0.train",
"Research_Data/weebit150.0.train",
#"Research_Data/weebit160.0.train",
#"Research_Data/weebit170.0.train",
#"Research_Data/weebit180.0.train",
#"Research_Data/weebit190.0.train",
"Research_Data/weebit10.0.test",
"Research_Data/weebit20.0.test",
"Research_Data/weebit30.0.test",
"Research_Data/weebit40.0.test",
"Research_Data/weebit50.0.test",
"Research_Data/weebit60.0.test",
"Research_Data/weebit70.0.test",
"Research_Data/weebit80.0.test",
"Research_Data/weebit90.0.test",
"Research_Data/weebit100.0.test",
"Research_Data/weebit110.0.test",
"Research_Data/weebit120.0.test",
"Research_Data/weebit130.0.test",
"Research_Data/weebit140.0.test",
"Research_Data/weebit150.0.test",
"Research_Data/weebit160.0.test",
"Research_Data/weebit170.0.test",
"Research_Data/weebit180.0.test",
"Research_Data/weebit190.0.test",
]

def extract(text):
    LingFeat = extractor.pass_text(text)
    LingFeat.preprocess()
    #1
    WoKF = LingFeat.WoKF_()
    #2
    WBKF = LingFeat.WBKF_()
    #3
    OSKF = LingFeat.OSKF_()
    #4
    EnDF = LingFeat.EnDF_()
    #5
    EnGF = LingFeat.EnGF_()
    #6
    PhrF = LingFeat.PhrF_()
    #7
    TrSF = LingFeat.TrSF_()
    #8
    POSF = LingFeat.POSF_()
    #9
    TTRF = LingFeat.TTRF_()
    #10
    VarF = LingFeat.VarF_()
    #11
    PsyF = LingFeat.PsyF_() 
    #12
    WoLF = LingFeat.WorF_()
    #13
    ShaF = LingFeat.ShaF_()
    #14
    TraF = LingFeat.TraF_()
    result = {**WoKF,**WBKF,**OSKF,**EnDF,**EnGF,**PhrF,**TrSF,**POSF,**TTRF,**VarF,**PsyF,**WoLF,**ShaF,**TraF}
    return result

def iterator(input_df):
    result = defaultdict(list)
    with alive_bar(len(input_df['Text'])) as bar:
        for i, text in enumerate(input_df['Text']):
            extracted_new = extract(text)
            for key in extracted_new:
                result[key].append(extracted_new[key])
            bar()   
    return result

def make_feature_table(target_data):
    input_df = pd.read_csv(target_data+".csv")
    input_dict = input_df.reset_index().to_dict(orient='list')

    # feature dict
    print("extracting handcrafting features...")
    feature_dict = iterator(input_df)
    feature_df = pd.DataFrame.from_dict(feature_dict)
    feature_df.to_csv(target_data+".feature.csv", index=False)
    return feature_df

def make_combined_table(target_data):
    input_df = pd.read_csv(target_data+".csv")
    input_dict = input_df.reset_index().to_dict(orient='list')
    feature_df = pd.read_csv(target_data+".feature.csv")
    feature_dict = feature_df.reset_index().to_dict(orient='list')

    # combined
    combined_dict = {**input_dict, **feature_dict}
    combined_df = pd.DataFrame.from_dict(combined_dict)
    combined_df.to_csv(target_data+".combined.csv", index=False)
    return combined_df

def check_correlation(corpus):
    corr_list = []
    input_df = pd.read_csv(dir_path+"/Data/extracted_"+corpus+".csv")
    for feature in input_df:
        this_corr = input_df['Grade'].corr(input_df[feature])
        if math.isnan(this_corr):
            corr_list.append((feature,0))
        else:
            corr_list.append((feature,abs(this_corr)))
    corr_list.sort(key=lambda tup: tup[1],reverse=True)
    return corr_list

if __name__ == '__main__':
    for target_data in target_datas:
        #make_feature_table(target_data)
        make_combined_table(target_data)