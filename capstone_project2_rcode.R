#############################################################
# Data Science Capstone - Choose Your Own!
#############################################################

rm(list=ls())

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org"); library(tidyverse)
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org"); library(caret)
if(!require("ROCR")) install.packages("ROCR", repos = "http://cran.us.r-project.org"); library(ROCR)

#### Load data ####

# Set working directory in which data is stored here
setwd('')

data <- read.csv('heart.csv')
names(data)[1] <- 'age'
str(data)

# Converting to factor
data$sex <- as.factor(data$sex)
data$cp <- as.factor(data$cp)
data$fbs <- as.factor(data$fbs)
data$restecg <- as.factor(data$restecg)
data$exang <- as.factor(data$exang)
data$slope <- as.factor(data$slope)
data$ca <- as.factor(data$ca)
data$thal <- as.factor(data$thal)
data$target <- as.factor(data$target)

#### Exploratory Analyses ####

# Check if there are any blank values in the dataset
sapply(data, function(x) sum(is.na(x))) # No NA values

# View summary of data 
summary(data)

# Bivariate analysis between independent variables and target
# This is used to assess whether the variable can be a good predictor of target

barplot(table(data$target, data$sex), 
        main = 'Split of target by sex buckets', 
        xlab = 'sex', ylab = 'Count',
        legend.text = c('Target 0', 'Target 1'), args.legend = c(x=1,y=200))

barplot(table(data$target, data$cp), 
        main = 'Split of target by cp buckets', 
        xlab = 'cp', ylab = 'Count',
        legend.text = c('Target 0', 'Target 1'), args.legend = c(x=5,y=150))

barplot(table(data$target, data$fbs), 
        main = 'Split of target by fbs buckets', 
        xlab = 'fbs', ylab = 'Count',
        legend.text = c('Target 0', 'Target 1'), args.legend = c(x=2.2,y=200))

barplot(table(data$target, data$restecg), 
        main = 'Split of target by restecg buckets', 
        xlab = 'restecg', ylab = 'Count',
        legend.text = c('Target 0', 'Target 1'), args.legend = c(x=4,y=150))

barplot(table(data$target, data$exang), 
        main = 'Split of target by exang buckets', 
        xlab = 'exang', ylab = 'Count',
        legend.text = c('Target 0', 'Target 1'), args.legend = c(x=2.2,y=200))

barplot(table(data$target, data$slope), 
        main = 'Split of target by slope buckets', 
        xlab = 'slope', ylab = 'Count',
        legend.text = c('Target 0', 'Target 1'), args.legend = c(x=1.25,y=120))

barplot(table(data$target, data$ca), 
        main = 'Split of target by ca buckets', 
        xlab = 'ca', ylab = 'Count',
        legend.text = c('Target 0', 'Target 1'))

barplot(table(data$target, data$thal), 
        main = 'Split of target by thal buckets', 
        xlab = 'thal', ylab = 'Count', 
        legend.text = c('Target 0', 'Target 1'), args.legend = c(x=2,y=150))

p1 <- hist(data$age[data$target==0]) 
p2 <- hist(data$age[data$target==1])
plot(p1, col=rgb(0,0,1,1/4), xlim=c(0,100), main='Distribution of age by target buckets', xlab='Age')
plot(p2, col=rgb(0,1,0,1/4), xlim=c(0,100), add=T)

p1 <- hist(data$trestbps[data$target==0]) 
p2 <- hist(data$trestbps[data$target==1])
plot(p2, col=rgb(0,1,0,1/4), xlim=c(80,200), main='Distribution of trestbps by target buckets', xlab='trestbps')
plot(p1, col=rgb(0,0,1,1/4), xlim=c(80,200), add=T)

p1 <- hist(data$chol[data$target==0]) 
p2 <- hist(data$chol[data$target==1])
plot(p2, col=rgb(0,1,0,1/4), xlim=c(100,600), main='Distribution of chol by target buckets', xlab='chol')
plot(p1, col=rgb(0,0,1,1/4), xlim=c(100,600), add=T)

p1 <- hist(data$thalach[data$target==0]) 
p2 <- hist(data$thalach[data$target==1])
plot(p2, col=rgb(0,1,0,1/4), xlim=c(50,250), main='Distribution of thalach by target buckets', xlab='thalach')
plot(p1, col=rgb(0,0,1,1/4), xlim=c(50,250), add=T)

p1 <- hist(data$oldpeak[data$target==0]) 
p2 <- hist(data$oldpeak[data$target==1])
plot(p2, col=rgb(0,1,0,1/4), xlim=c(0,8), main='Distribution of oldpeak by target buckets', xlab='oldpeak')
plot(p1, col=rgb(0,0,1,1/4), xlim=c(0,8), add=T)


#### Creating train and test datasets ####
set.seed(1)
test_index <- createDataPartition(y = data$target, times = 1, p = 0.2, list = FALSE)
train_data <- data[-test_index,]
test_data <- data[test_index,]

#### Building logistic regression model #### 
model <- glm(target~., data = train_data, family = binomial(link = 'logit'))
summary(model)

# Using stepwise backward elimination to select variables and improve model
select_vars_model <- step(model)
summary(select_vars_model)

#### Making predictions on test set and evaluating model performance ####

test_predicted <- predict(select_vars_model, test_data, type='response')

ROCRpred = prediction(test_predicted,test_data$target)
ROCRperf = performance(ROCRpred, "tpr", "fpr")

# Plot ROC curve and get auc
plot(ROCRperf, main='ROC curve')
auc <- attributes(performance(ROCRpred, 'auc'))$y.values[[1]]

# Confusion Matrix
predClass <- as.factor(ifelse(test_predicted>=0.5,1,0))
confusionMatrix(test_data$target, predClass)
