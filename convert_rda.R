# conda install -c conda-forge r-base r-essentials r-tidyverse r-writexl r-devtools
# Rscript --vanilla -e "devtools::install_github('krumsieklab/maplet', subdir='maplet')"
library(dplyr)
library(SummarizedExperiment)
library(maplet)
library(magrittr)

file.path <- "data/replicate_Willer_diet_health_v3.rda"

load(file = file.path)

# Randomly insert NA
RANDOM_NA <- FALSE

# Based on https://github.com/krumsieklab/ad-adni-Xplatforms/blob/main/nightingale_lipoproteins_preprocessing.R
# and https://github.com/krumsieklab/ad-brain-landscape/blob/c39bf85fb16fb75238e62be4c325fe4b9b37a4c8/1_metabolomics_preprocessing.R
D2 <- D %>%
  mt_pre_zero_to_na() %>%
  mt_pre_filter_missingness(feat_max = 0.4) %>%
  mt_pre_filter_missingness(samp_max = 0.4) %>%
  mt_anno_missingness(anno_type = "features", out_col = "missing") %>%
  mt_pre_trans_log() %>%
  mt_pre_impute_min() %>% #mt_pre_impute_knn() %>%
  # Combine duplicates
  #mt_modify_avg_samples("eid") %>%  # Note this function fails since there are no duplicate samples
  # Outlier
  # Sample outlier
  mt_pre_outlier_lof(seq_k = c(5, 10, 20, 30, 40, 50)) %>%
  # Metabolomic outlier
  mt_pre_outlier_to_na(use_quant=TRUE, quant_thresh =0.025) %>%
  mt_pre_impute_min() #mt_pre_impute_knn()

mt_write_se_xls(D2, "data/BioBank_export.xlsx")

# Cross experiment integration
# Based on mtVAE: https://www.nature.com/articles/s42003-022-03579-3#Sec10
adni <- mt_load_se_xls(file = "data/Neuro-Datasets/ADNI_Nightingale_Baseline_preprocessed.xlsx")
tulsa <- mt_load_se_xls(file = "data/Neuro-Datasets/Tulsa_Nightingale_Preprocessed.xlsx")

# biobank minimum age is 40
uniform_biobank <- D2 %>%
  mt_modify_filter_samples(Age.at.recruitment <= 60) %>%
  mt_modify_filter_samples(Age.at.recruitment >= 40) %>%
  mt_modify_filter_samples(Sex == "Male")
colData(D2)$is_ref <- colData(D2)$eid %in% colData(uniform_biobank)$eid


# Adni minimum age is 54
colData(adni)$SC_Age[grepl(">89", colData(adni)$SC_Age, fixed=TRUE)] <- "90"
colData(adni)$SC_Age <- as.numeric(colData(adni)$SC_Age)
uniform_adni <- adni %>%
    mt_modify_filter_samples(SC_Age <= 70) %>%
    mt_modify_filter_samples(SC_Age >= 50) %>%
    mt_modify_filter_samples(PTGENDER == 1)  # 1 is male
colData(adni)$is_ref <- colData(adni)$Sample_id %in% colData(uniform_adni)$Sample_id

# Tulsa minimum age is 18
uniform_tulsa <- tulsa %>%
    mt_modify_filter_samples(Age <= 40) %>%
    mt_modify_filter_samples(Age >= 20) %>%
    mt_modify_filter_samples(Gender == "Male")
colData(tulsa)$is_ref <- colData(tulsa)$Sample_id %in% colData(uniform_tulsa)$Sample_id

D2 %>%
  mt_pre_trans_scale(ref_samples=is_ref) %>%
  mt_write_se_xls("data/BioBank_norm.xlsx")

adni %>%
  mt_pre_trans_scale(ref_samples=is_ref) %>%
  mt_write_se_xls("data/ADNI_norm.xlsx")

tulsa %>%
  mt_pre_trans_scale(ref_samples=is_ref) %>%
  mt_write_se_xls("data/Tulsa_norm.xlsx")
