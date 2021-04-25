WoKF_features_ = [
    #WoKF
    "WRich05_S","WClar05_S","WNois05_S","WTopc05_S","WRich10_S","WClar10_S","WNois10_S","WTopc10_S","WRich15_S","WClar15_S","WNois15_S","WTopc15_S","WRich20_S","WClar20_S","WNois20_S","WTopc20_S",
    ]

WBKF_features_ = [
    #WBKF
    "BRich05_S","BClar05_S","BNois05_S","BTopc05_S","BRich10_S","BClar10_S","BNois10_S","BTopc10_S","BRich15_S","BClar15_S","BNois15_S","BTopc15_S","BRich20_S","BClar20_S","BNois20_S","BTopc20_S",
    ]

OSKF_features_ = [
    #OSKF
    #"ORich05_S","OClar05_S","ONois05_S","OTopc05_S","ORich10_S","OClar10_S","ONois10_S","OTopc10_S","ORich15_S","OClar15_S","ONois15_S","OTopc15_S","ORich20_S","OClar20_S","ONois20_S","OTopc20_S",
    ]

EnDF_features_ = [
    #EnDF
    "to_EntiM_C","as_EntiM_C","at_EntiM_C","to_UEnti_C","as_UEnti_C","at_UEnti_C",
    ]

EnGF_features_ = [
    #EnGF
    "ra_SSToT_C","ra_SOToT_C","ra_SXToT_C","ra_SNToT_C","ra_OSToT_C","ra_OOToT_C","ra_OXToT_C","ra_ONToT_C","ra_XSToT_C","ra_XOToT_C","ra_XXToT_C","ra_XNToT_C","ra_NSToT_C","ra_NOToT_C","ra_NXToT_C","ra_NNToT_C","LoCohPA_S","LoCohPW_S","LoCohPU_S","LoCoDPA_S","LoCoDPW_S","LoCoDPU_S",]

PhrF_features_ = [
    #PhrF
    "to_NoPhr_C","as_NoPhr_C","at_NoPhr_C","ra_NoVeP_C","ra_NoSuP_C","ra_NoPrP_C","ra_NoAjP_C","ra_NoAvP_C","to_VePhr_C","as_VePhr_C","at_VePhr_C","ra_VeNoP_C","ra_VeSuP_C","ra_VePrP_C","ra_VeAjP_C","ra_VeAvP_C","to_SuPhr_C","as_SuPhr_C","at_SuPhr_C","ra_SuNoP_C","ra_SuVeP_C","ra_SuPrP_C","ra_SuAjP_C","ra_SuAvP_C","to_PrPhr_C","as_PrPhr_C","at_PrPhr_C","ra_PrNoP_C","ra_PrVeP_C","ra_PrSuP_C","ra_PrAjP_C","ra_PrAvP_C","to_AjPhr_C","as_AjPhr_C","at_AjPhr_C","ra_AjNoP_C","ra_AjVeP_C","ra_AjSuP_C","ra_AjPrP_C","ra_AjAvP_C","to_AvPhr_C","as_AvPhr_C","at_AvPhr_C","ra_AvNoP_C","ra_AvVeP_C","ra_AvSuP_C","ra_AvPrP_C","ra_AvAjP_C",
    ]

to_PhrF_features_ = [
    #PhrF
    "to_NoPhr_C","to_VePhr_C","to_PrPhr_C","to_AvPhr_C"
    ]

as_PhrF_features_ = [
    #PhrF
    "as_NoPhr_C","as_SuPhr_C","as_PrPhr_C","as_AvPhr_C",
    ]

ra_PhrF_features_ = [
    #PhrF
    "ra_NoVeP_C","ra_NoSuP_C","ra_NoPrP_C","ra_NoAjP_C","ra_NoAvP_C","ra_VeNoP_C","ra_VeSuP_C","ra_VePrP_C","ra_VeAjP_C","ra_VeAvP_C","ra_SuNoP_C","ra_SuVeP_C","ra_SuPrP_C","ra_SuAjP_C","ra_SuAvP_C","ra_PrNoP_C","ra_PrVeP_C","ra_PrSuP_C","ra_PrAjP_C","ra_PrAvP_C","ra_AjNoP_C","ra_AjVeP_C","ra_AjSuP_C","ra_AjPrP_C","ra_AjAvP_C","ra_AvNoP_C","ra_AvVeP_C","ra_AvSuP_C","ra_AvPrP_C","ra_AvAjP_C",
    ]

TrSF_features_ = [
    #TrSF
    "to_TreeH_C","as_TreeH_C","at_TreeH_C","to_FTree_C","as_FTree_C","at_FTree_C",
    ]

POSF_features_ = [
    #POSF
    "to_NoTag_C","as_NoTag_C","at_NoTag_C","ra_NoAjT_C","ra_NoVeT_C","ra_NoAvT_C","ra_NoSuT_C","ra_NoCoT_C","to_VeTag_C","as_VeTag_C","at_VeTag_C","ra_VeAjT_C","ra_VeNoT_C","ra_VeAvT_C","ra_VeSuT_C","ra_VeCoT_C","to_AjTag_C","as_AjTag_C","at_AjTag_C","ra_AjNoT_C","ra_AjVeT_C","ra_AjAvT_C","ra_AjSuT_C","ra_AjCoT_C","to_AvTag_C","as_AvTag_C","at_AvTag_C","ra_AvAjT_C","ra_AvNoT_C","ra_AvVeT_C","ra_AvSuT_C","ra_AvCoT_C","to_SuTag_C","as_SuTag_C","at_SuTag_C","ra_SuAjT_C","ra_SuNoT_C","ra_SuVeT_C","ra_SuAvT_C","ra_SuCoT_C","to_CoTag_C","as_CoTag_C","at_CoTag_C","ra_CoAjT_C","ra_CoNoT_C","ra_CoVeT_C","ra_CoAvT_C","ra_CoSuT_C","to_ContW_C","as_ContW_C","at_ContW_C","to_FuncW_C","as_FuncW_C","at_FuncW_C","ra_CoFuW_C",
    ]

at_POSF_features_ = [
    #POSF
    "at_NoTag_C","at_VeTag_C","at_AjTag_C","at_AvTag_C","at_SuTag_C","at_CoTag_C","at_ContW_C","at_FuncW_C",
    ]

to_POSF_features_ = [
    #POSF
    "to_NoTag_C","to_VeTag_C","to_AjTag_C","to_AvTag_C","to_CoTag_C","to_ContW_C","to_FuncW_C",
    ]


TTRF_features_ = [
    #TTRF
    "SimpTTR_S","CorrTTR_S","BiLoTTR_S","UberTTR_S","MTLDTTR_S",
    ]

VarF_features_ = [
    #VarF
    "SimpNoV_S","SquaNoV_S","CorrNoV_S","SimpVeV_S","SquaVeV_S","CorrVeV_S","SimpAjV_S","SquaAjV_S","CorrAjV_S","SimpAvV_S","SquaAvV_S","CorrAvV_S",
    ]

PsyF_features_ = [
    #PsyF
    "to_AAKuW_C","as_AAKuW_C","at_AAKuW_C","to_AAKuL_C","as_AAKuL_C","at_AAKuL_C","to_AABiL_C","as_AABiL_C","at_AABiL_C","to_AABrL_C","as_AABrL_C","at_AABrL_C","to_AACoL_C","as_AACoL_C","at_AACoL_C",
    ]

to_PsyF_features_ = [
    #PsyF
    "to_AAKuW_C","to_AAKuL_C","to_AABiL_C","to_AABrL_C","to_AACoL_C",
    ]

at_PsyF_features_ = [
    #PsyF
    "at_AAKuW_C","at_AAKuL_C","at_AABiL_C","at_AABrL_C","at_AACoL_C",
    ]

as_PsyF_features_ = [
    #PsyF
    "as_AAKuW_C","as_AAKuL_C","as_AABiL_C","as_AABrL_C","as_AACoL_C",
    ]

WorF_features_ = [
    #WorF
    "to_SbFrQ_C","as_SbFrQ_C","at_SbFrQ_C","to_SbCDC_C","as_SbCDC_C","at_SbCDC_C","to_SbFrL_C","as_SbFrL_C","at_SbFrL_C","to_SbCDL_C","as_SbCDL_C","at_SbCDL_C","to_SbSBW_C","as_SbSBW_C","at_SbSBW_C","to_SbL1W_C","as_SbL1W_C","at_SbL1W_C","to_SbSBC_C","as_SbSBC_C","at_SbSBC_C","to_SbL1C_C","as_SbL1C_C","at_SbL1C_C",
    ]

ShaF_features_ = [
    #ShaF
    "TokSenM_S","TokSenS_S","TokSenL_S","as_Token_C","as_Sylla_C","at_Sylla_C","as_Chara_C","at_Chara_C",
    ]

TraF_features_ = [
    #TraF
    "SmogInd_S","ColeLia_S","Gunning_S","AutoRea_S","FleschG_S","LinseaW_S",
    ]

# all handcrafted features
FeatureSet_Total_HF_ = WoKF_features_+WBKF_features_+OSKF_features_+EnDF_features_+EnGF_features_+PhrF_features_+TrSF_features_+POSF_features_+TTRF_features_+VarF_features_+PsyF_features_+WorF_features_+ShaF_features_+TraF_features_

# Advanced Semantic features
FeatureSet_AdSem_HF_ = WoKF_features_+WBKF_features_+OSKF_features_

# Discourse features
FeatureSet_Disco_HF_ = EnDF_features_+EnGF_features_

# Syntactic features
FeatureSet_Synta_HF_ = PhrF_features_+TrSF_features_+POSF_features_

# Lexico-Semantic features
FeatureSet_LxSem_HF_ = TTRF_features_+VarF_features_+PsyF_features_+WorF_features_

# Shallow Traditional features
FeatureSet_ShTra_HF_ = ShaF_features_+TraF_features_

def FeatureSet_Total_HF():
    return FeatureSet_Total_HF_

def FeatureSet_AdSem_HF():
    return FeatureSet_AdSem_HF_

def FeatureSet_Disco_HF():
    return FeatureSet_Disco_HF_

def FeatureSet_Synta_HF():
    return FeatureSet_Synta_HF_

def FeatureSet_LxSem_HF():
    return FeatureSet_LxSem_HF_

def FeatureSet_ShTra_HF():
    return FeatureSet_ShTra_HF_

def FeatureSet_T_1_HF():
    return FeatureSet_Total_HF_

def FeatureSet_T_2_HF():
    return FeatureSet_Disco_HF_ + FeatureSet_Synta_HF_ + FeatureSet_LxSem_HF_ + FeatureSet_ShTra_HF_

def FeatureSet_T_3_HF():
    return FeatureSet_AdSem_HF_ + FeatureSet_Synta_HF_ + FeatureSet_LxSem_HF_ + FeatureSet_ShTra_HF_

def FeatureSet_H_1_HF():
    return FeatureSet_AdSem_HF_ + FeatureSet_Disco_HF_

def FeatureSet_L_1_HF():
    return FeatureSet_Synta_HF_ + FeatureSet_LxSem_HF_

def FeatureSet_L_2_HF():
    return TrSF_features_ + POSF_features_ + FeatureSet_LxSem_HF_

def FeatureSet_L_3_HF():
    return TrSF_features_ + POSF_features_ + TTRF_features_ + PsyF_features_ + WorF_features_

def FeatureSet_L_4_HF():
    return PhrF_features_ + TrSF_features_ + FeatureSet_LxSem_HF_

def FeatureSet_E_1_HF():
    return FeatureSet_AdSem_HF_ + PsyF_features_ + WorF_features_ + TraF_features_

def FeatureSet_E_2_HF():
    return FeatureSet_AdSem_HF_ + PsyF_features_ + WorF_features_

def FeatureSet_E_3_HF():
    return PsyF_features_ + WorF_features_ + TraF_features_

def FeatureSet_P_1_HF():
    return EnDF_features_ + ShaF_features_ + TrSF_features_ + POSF_features_ + WorF_features_ + PsyF_features_

def FeatureSet_P_2_HF():
    return EnDF_features_ + ShaF_features_ + TrSF_features_ + POSF_features_ + WorF_features_ + PsyF_features_ + TraF_features_

def FeatureSet_P_3_HF():
    return EnDF_features_ + ShaF_features_ + TrSF_features_ + POSF_features_ + WorF_features_ + PsyF_features_ + TraF_features_ + VarF_features_

def FeatureSet_G_3_HF():
    return EnDF_features_ + ShaF_features_ + TrSF_features_ + POSF_features_ + WorF_features_ + PsyF_features_ + TraF_features_ + VarF_features_ + EnGF_features_

def FeatureSet_C_1_HF():
    return EnDF_features_ + ShaF_features_ + TrSF_features_ + to_POSF_features_ + WorF_features_ + at_PsyF_features_ + TraF_features_ + VarF_features_ 

def WoKF_features():
    return WoKF_features_ 

def WBKF_features():
    return WBKF_features_

def OSKF_features():
    return OSKF_features_

def EnDF_features():
    return EnDF_features_

def EnGF_features():
    return EnGF_features_

def PhrF_features():
    return PhrF_features_

def TrSF_features():
    return TrSF_features_

def POSF_features():
    return POSF_features_

def TTRF_features():
    return TTRF_features_

def VarF_features():
    return VarF_features_

def PsyF_features():
    return PsyF_features_

def WorF_features():
    return WorF_features_

def ShaF_features():
    return ShaF_features_

def TraF_features():
    return TraF_features_

def FeatureSet_all_as_list_HF():
    return [WoKF_features_,#1
            WBKF_features_,
            EnDF_features_,#3
            EnGF_features_,
            PhrF_features_,#5
            TrSF_features_,
            POSF_features_,#7
            TTRF_features_,
            VarF_features_,#9
            PsyF_features_,
            WorF_features_,#11
            ShaF_features_,
            TraF_features_]