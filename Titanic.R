#########################
## TITANIC KAGGLE IN R ##
#########################

## SET UP ##

rm(list=ls()); gc()     # clear the workspace
setwd("/Users/Zelda/Data Science/GA/Homework/Titanic/")  # Set working directory
train <- read.csv("train.csv", stringsAsFactors=FALSE)  # Load train dataset
str(train)  # look at data frame

