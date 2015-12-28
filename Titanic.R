#########################
## TITANIC KAGGLE IN R ##
#########################

## SET UP ##

rm(list=ls()); gc()     # clear the workspace
set.seed(21)  # To make results repicable
setwd("/Users/Zelda/Data Science/GA/Homework/Titanic/")  # Set working directory

library(rpart)
library(randomForest)

train <- read.csv("train.csv")  # Load train dataset
test <- read.csv("test.csv")  # Load test dataset
test$Survived <- NA
combo <- rbind(train,test) # Create large dataset with both train/test

str(train)  # look at train data frame

#######################
## LOOK AT VARIABLES ##
#######################
summary(train$Sex)  # Gender
prop.table(table(train$Sex, train$Survived), 1)  # Gender vs survival
# LN: Females were more likely to survive

summary(train$Age)  # Age

train$Child<- 0  # Create new variable for child
train$Child[train$Age<18]<- 1

# Total number of survivors
aggregate(Survived ~ Child + Sex, data=train, FUN=sum)
# Total number in each group
aggregate(Survived ~ Child + Sex, data=train, FUN=length)
# Proportions
aggregate(Survived ~ Child + Sex, data=train, FUN=function(x) {sum(x)/length(x)})

train$Faregrp <- '30+'
train$Faregrp[train$Fare < 30 & train$Fare >= 20] <- '20-30'
train$Faregrp[train$Fare < 20 & train$Fare >= 10] <- '10-20'
train$Faregrp[train$Fare < 10] <- '10'

aggregate(Survived ~ Faregrp + Pclass + Sex, data=train, FUN=function(x) {sum(x)/length(x)})

#############################
## IMPUTING MISSING VALUES ##
############################
summary(combo)

#-- Age --#
summary(combo$Age)  # 263 missing values
# Use decision tree to impute missing values
Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize,data=combo[!is.na(combo$Age),], method="anova")
combo$Age[is.na(combo$Age)] <- predict(Agefit, combo[is.na(combo$Age),])

#-- Embarked --#
combo$Embarked[combo$Embarked==''] = 'S'
summary(combo$Embarked)
combo$Embarked <- factor(combo$Embarked)

#-- Fare --#
summary(combo$Fare)
combo$Fare[is.na(combo$Fare)] <- median(combo$Fare, na.rm=TRUE) # Replace missing with median
summary(combo$Fare)  # Check it worked

#########################
## FEATURE ENGINEERING ##
#########################

combo$Name <- as.character(combo$Name)  # Change string to character from factor
str(combo)  # Check it worked

# Create variable for title
combo$Title <- sapply(combo$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
combo$Title <- sub(' ', '', combo$Title)

table(combo$Title)

# Change uncommon titles to more common ones
combo$Title[combo$Title=='Mme'] <- 'Mrs'
combo$Title[combo$Title %in% c('Mlle','Ms')] <- 'Miss'
combo$Title[combo$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
combo$Title[combo$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'

# Change Title variable into a factor
combo$Title <- factor(combo$Title)


# Family size
combo$FamilySize <- combo$SibSp + combo$Parch + 1

# Surname
combo$Surname <- sapply(combo$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})

# Family ID - family size + surname
combo$FamilyID <- paste(as.character(combo$FamilySize), combo$Surname, sep="")
combo$FamilyID[combo$FamilySize <= 3] <- 'Small'
# Small families are less than 2 people

table(combo$FamilyID)

# Clean up family ID variable
famIDs <- data.frame(table(combo$FamilyID))
famIDs <- famIDs[famIDs$Freq <= 3,]
combo$FamilyID[combo$FamilyID %in% famIDs$Var1] <- 'Small'
combo$FamilyID <- factor(combo$FamilyID)
summary(combo$FamilyID)

######################
## MAKE PREDICTIONS ##
######################

# Split train/test data frame
train <- combo[1:891,]
test <- combo[892:1309,]

fit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID, data=train, importance=TRUE, ntree=2000)

Prediction <- predict(fit, test)
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "firstforestR.csv", row.names = FALSE)

