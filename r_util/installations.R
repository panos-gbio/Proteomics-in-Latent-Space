# Install packages for the project
install.packages(c("openxlsx", "igraph", "ggplot2", "caret", "ROCR", "rgexf", "Hmisc", "patchwork", 
                   "Hmisc","VennDiagram","dplyr","HGNChelper","tidyr","tidyverse"))
install.packages("RColorBrewer")
# Bioconductor Packages 
if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
install.packages("glmnet")
BiocManager::install("biomaRt")



getwd()

