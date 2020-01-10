#Prostate Cancer Prediction Project
#Harvardx: PH125.9x Data Science
#Frederico de Almeida Meirelles Palma

## Dataset

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(funModeling)) install.packages("funModeling", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("recorrplotadr", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(nnet)) install.packages("nnet", repos = "http://cran.us.r-project.org")

# Prostate Cancer dataset:
# https://www.kaggle.com/sajidsaifi/prostate-cancer #source
# https://github.com/fredpalma/machine-learning/blob/master/prostateCancer/Prostate_Cancer.csv 
# The file will be loaded from my github account

# The data 
data <- read.csv("https://raw.githubusercontent.com/fredpalma/machine-learning/master/prostateCancer/Prostate_Cancer.csv")

library("ggplot2")
library("dplyr")
library("corrplot")
library("funModeling")

# Data Analysis


## Exploring the dataset

# number of rows and columns
dim(data)


#head
head(data)



#data summary
summary(data)


#map data
map(data, function(.x) sum(is.na(.x)))

#proportion
prop.table(table(data$diagnosis_result))



#diagnosis result distribution
options(repr.plot.width=5, repr.plot.height=5)
ggplot(data, aes(x=diagnosis_result))+geom_bar(fill="blue",alpha=0.5)+theme_bw()+labs(title="Diagnosis Result Distribution")


#show all histograms
plot_num(data %>% select(-id), bins=10)


#Remove id variable
data2 <- data %>%select(-id)
ncol(data2)


# Model Development

## PCA and LDA

### Principal Component Analysis (PCA).

#Principal Component Analysis
pca_data <- prcomp(data2[,2:ncol(data2)], center = TRUE, scale = TRUE)
plot(pca_data, type="l")


#PCA data Summary
summary(pca_data)


#PCA data2 
pca_data2 <- prcomp(data2[,3:ncol(data2)], center = TRUE, scale = TRUE)
plot(pca_data2, type="l")

#PCA data2 Summary
summary(pca_data2)


### Linear Discriminant Analysis (LDA)

#Linear Discriminant Analysis
lda_data <- MASS::lda(diagnosis_result~., data = data, center = TRUE, scale = TRUE) 
lda_data

lda_predict <- predict(lda_data, data)$x %>% as.data.frame() %>% cbind(diagnosis_result=data$diagnosis_result)

ggplot(lda_predict, aes(x=LD1, fill=diagnosis_result)) + geom_density(alpha=0.5)



## Creating training and testing datasets 

#Training and Testing 
#set.seed(3) # if using R 3.5 or earlier
set.seed(3, sample.kind = "Rounding")    # if using R 3.6 or later
data3 <- cbind (diagnosis_result=data$diagnosis_result, data2)
data_sampling_index <- createDataPartition(data$diagnosis, times=1, p=0.8, list = FALSE)
training <- data3[data_sampling_index, ]
testing <- data3[-data_sampling_index, ]


fitControl <- trainControl(method="cv",
                           number = 15,    #Either the number of folds or number of resampling iterations
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)


## Naive Bayes Model

# Naive Bayes Model

naive <- train(diagnosis_result~.,
                      training,
                      method="nb",
                      metric="ROC",
                      preProcess=c('center', 'scale'), #in order to normalize the data
                      trace=FALSE,
                      trControl=fitControl)

naive_pred <- predict(naive, testing)
confusionmatrix_naive <- confusionMatrix(naive_pred, testing$diagnosis_result, positive = "M")
confusionmatrix_naive



#Naive Bayes plot
plot(varImp(naive), top=5, main="Naive Bayes - Top 5 variables")


## Logistic Regression Model 

# Logistic Regression Model 

logreg<- train(diagnosis_result ~., 
               data = training, method = "glm",
                     metric = "ROC",
                     preProcess = c("scale", "center"),  # in order to normalize the data
                     trControl= fitControl)
logreg_pred<- predict(logreg, testing)

# Checking results
confusionmatrix_logreg <- confusionMatrix(logreg_pred, testing$diagnosis_result, positive = "M")
confusionmatrix_logreg


#Logistic Regression Top 5 variables Visualization

plot(varImp(logreg), top=5, main=" Logistic  Regression - Top 5 variables")




## Random Forest Model

#Random Forest Model 

randomforest <- train(diagnosis_result~.,
                            training,
                            method="rf",  
                            metric="ROC",
                            tuneLength=10,
                            tuneGrid = expand.grid(mtry = c(2, 3, 6)),
                            preProcess = c('center', 'scale'),
                            trControl=fitControl)

randomforest_pred <- predict(randomforest, testing)

confusionmatrix_randomforest <- confusionMatrix(randomforest_pred, testing$diagnosis_result, positive = "M")
confusionmatrix_randomforest

#Random Forest - Top 5 variables 

plot(varImp(randomforest), top=5, main="Random Forest - Top 5 variables")


## K Nearest Neighbors (KNN) Model

#K Nearest Neighbors

knn <- train(diagnosis_result~.,
                   training,
                   method="knn",
                   metric="ROC",
                   preProcess = c('center', 'scale'),
                   tuneLength=10,
                   trControl=fitControl)

knn_pred <- predict(knn, testing)
confusionmatrix_knn <- confusionMatrix(knn_pred, testing$diagnosis_result, positive = "M")
confusionmatrix_knn


#KNN - Top 5 variables}

plot(varImp(knn), top=5, main="KNN - Top 5 variables")



## Neural Network with PCA Model

#Neural Network with PCA Model

nn_pca <- train(diagnosis_result~.,
                        training,
                        method="nnet",
                        metric="ROC",
                        preProcess=c('center', 'scale', 'pca'),
                        tuneLength=10,
                        trace=FALSE,
                        trControl=fitControl)

nn_pca_pred <- predict(nn_pca, testing)
confusionmatrix_nn_pca <- confusionMatrix(nn_pca_pred, testing$diagnosis_result, positive = "M")
confusionmatrix_nn_pca


# Neural Network with PCA - Top 5 variables

plot(varImp(nn_pca), top=5, main="Neural Network with PCA - Top 5 variables")



## Neural Network with LDA Model

#LDA training and testing sets creation

training_lda <- lda_predict[data_sampling_index, ]
testing_lda <- lda_predict[-data_sampling_index, ]


#Neural Network with LDA Model

nn_lda <- train(diagnosis_result~.,
                        training_lda,
                        method="nnet",
                        metric="ROC",
                        preProcess=c('center', 'scale'),
                        tuneLength=10,
                        trace=FALSE,
                        trControl=fitControl)

nn_lda_pred <- predict(nn_lda, testing_lda)
confusionmatrix_nn_lda <- confusionMatrix(nn_lda_pred, testing_lda$diagnosis_result, positive = "M")
confusionmatrix_nn_lda

# Results

#Summary Models Results

model_list <- list(Naive_Bayes=naive,
                    Logistic_Regression=logreg,
                    Random_Forest=randomforest,
                    KNN=knn,
                    Neural_PCA=nn_pca,
                    Neural_LDA=nn_lda)                                     
model_results <- resamples(model_list)

summary(model_results)


#Plotting the results 

bwplot(model_results, metric="ROC")


#Confusion Matrix Results

confusionmatrix_list <- list(
  Naive_Bayes=confusionmatrix_naive,
  Logistic_Regression=confusionmatrix_logreg,
  Random_Forest=confusionmatrix_randomforest,
  KNN=confusionmatrix_knn,
  Neural_PCA=confusionmatrix_nn_pca,
  Neural_LDA=confusionmatrix_nn_lda)   
confusionmatrix_list_results <- sapply(confusionmatrix_list, function(x) x$byClass)
confusionmatrix_list_results %>% knitr::kable()



#Final Results

confusionmatrix_results_max <- apply(confusionmatrix_list_results, 1, which.is.max)

metrics_report <- data.frame(metric=names(confusionmatrix_results_max), 
                            best_model=colnames(confusionmatrix_list_results)[confusionmatrix_results_max],
                            value=mapply(function(x,y) {confusionmatrix_list_results[x,y]}, 
                                         names(confusionmatrix_results_max), 
                                         confusionmatrix_results_max))
rownames(metrics_report) <- NULL
metrics_report


# System Information
print("Operating System and R:")
version
