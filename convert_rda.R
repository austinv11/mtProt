# conda install -c conda-forge r-base r-essentials r-tidyverse r-writexl r-devtools
# Rscript --vanilla -e "devtools::install_github('krumsieklab/maplet', subdir='maplet')"
library(dplyr)
library(writexl)
library(SummarizedExperiment)
library(maplet)
library(magrittr)

file.path <- "data/replicate_Willer_diet_health_v3.rda"

load(file = file.path)

# Randomly insert NA
RANDOM_NA <- FALSE

# TODO: run on server
D2 <- D %>%
  mt_pre_zero_to_na() %>%
  mt_pre_filter_missingness(feat_max = 0.4, samp_max = 0.4) %>%
  mt_pre_trans_log() %>%
  #mt_pre_impute_knn() %>%
  mt_pre_impute_min() %>%
  mt_pre_outlier_lof() %>%
  mt_pre_outlier_to_na() %>%
  #mt_pre_impute_knn()
  mt_pre_impute_min()

mt_write_se_xls(D2, "data/BioBank_export.xlsx")

info <- elementMetadata(D2)
fields <- info$title  # field_id2
names <- rownames(info)
group <- info$Group
subgroup <- info$Subgroup

metaDf <- data.frame(
  COMP_IDstr = fields,
  BIOCHEMICAL = names,
  SUPER_PATHWAY = group,
  SUB_PATHWAY = subgroup
)

patientInfo <- colData(D2)

# 85/15 train/test split
data.matrix <- as.data.frame(t(assay(D2)))

colnames(data.matrix) <- fields
data.matrix$eid <- patientInfo$eid

training.set <- sample_frac(data.matrix, 0.85)
testing.set <- data.matrix[which(!(rownames(data.matrix) %in% rownames(training.set))),]

eids <- c(training.set$eid, testing.set$eid)
training.set <- select(training.set, -eid)
testing.set <- select(testing.set, -eid)

training.set.nona <- data.frame(training.set)
testing.set.nona <- data.frame(testing.set)

if (RANDOM_NA) {
  for (column in 1:ncol(data.matrix)) {
    training.set[sample(1:nrow(training.set), as.integer(.15*nrow(training.set))),column] <- NA
    testing.set[sample(1:nrow(testing.set), as.integer(.15*nrow(testing.set))),column] <- NA
  }
  training.col.remove <- which(colMeans(is.na(training.set)) > .4)
  testing.col.remove <- which(colMeans(is.na(testing.set)) > .4)

  if (length(training.col.remove) > 0) {
    for (column in training.col.remove) {
      na.rows <- which(is.na(training.set[,column]))
      to.replace <- sample(na.rows, as.integer(0.6*length(na.rows)))
      training.set[to.replace,column] <- training.set.nona[to.replace,column]
    }
  }

  if (length(testing.col.remove) > 0) {
    for (column in testing.col.remove) {
      na.rows <- which(is.na(testing.set[,column]))
      to.replace <- sample(na.rows, as.integer(0.6*length(na.rows)))
      testing.set[to.replace,column] <- testing.set.nona[to.replace,column]
    }
  }

  training.remove <- which(rowMeans(is.na(training.set)) > .4)
  testing.remove <- which(rowMeans(is.na(training.set)) > .4)

  if (length(training.remove) > 0) {
    for (row in training.remove) {
      na.cols <- which(is.na(training.set[row,]))
      to.replace <- sample(na.cols, as.integer(0.6*length(na.cols)))
      training.set[row,to.replace] <- training.set.nona[row,to.replace]
    }
  }

  if (length(testing.remove) > 0) {
    for (row in testing.remove) {
      na.cols <- which(is.na(testing.set[row,]))
      to.replace <- sample(na.cols, as.integer(0.6*length(na.cols)))
      testing.set[row,to.replace] <- testing.set.nona[row,to.replace]
    }
  }
}

write_xlsx(
  x = list(`Training Set`=training.set,
           `Testing Set`=testing.set,
           `Training Set No NA`=training.set.nona,
           `Testing Set No NA`=testing.set.nona,
           `Metabolite Annotations`=metaDf,
           `Patient Information`=inner_join(data.frame(eid=eids), as.data.frame(patientInfo), by = "eid") %>% as.data.frame()),
  path = if (RANDOM_NA) { "data/BioBank_NA.xlsx" } else { "data/BioBank.xlsx" },
  use_zip64 = TRUE
)