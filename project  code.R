install.packages("lattice")
library(ranger)
library(caret)
library(data.table)
creditcard <- read.csv("C:/Users/PAWAN/Downloads/Credit-Card-Dataset/creditcard.csv")
View(creditcard)

dim(creditcard)
head(creditcard,6)
tail(creditcard,6)
table(creditcard$Class)
summary(creditcard$Amount)
names(creditcard)
var(creditcard$Amount)
sd(creditcard$Amount)

#Data Manipulation
head(creditcard)
creditcard$Amount=scale(creditcard$Amount)
NewData=creditcard[,-c(1)]
head(NewData)

# data modeling
library(caTools)
set.seed(123)
data_sample = sample.split(NewData$Class,SplitRatio=0.80)
train_data = subset(NewData,data_sample==TRUE)
test_data = subset(NewData,data_sample==FALSE)
dim(train_data)
dim(test_data)
Logistic_Model=glm(Class~.,test_data,family=binomial())
summary(Logistic_Model)
plot(Logistic_Model)
library(pROC)
lr.predict <- predict(Logistic_Model,train_data, probability = TRUE)


library(rpart)
library(rpart.plot)
decisionTree_model <- rpart(Class ~ . , creditcard, method = 'class')
predicted_val <- predict(decisionTree_model, creditcard, type = 'class')
probability <- predict(decisionTree_model, creditcard, type = 'prob')
rpart.plot(decisionTree_model)
library(neuralnet)
ANN_model =neuralnet (Class~.,train_data,linear.output=FALSE)
plot(ANN_model)

predANN=compute(ANN_model,test_data)
resultANN=predANN$net.result
resultANN=ifelse(resultANN>0.5,1,0)
library(gbm, quietly=TRUE)

# Get the time to train the GBM model
system.time(
  model_gbm <- gbm(Class ~ .
                   , distribution = "bernoulli"
                   , data = rbind(train_data, test_data)
                   , n.trees = 500
                   , interaction.depth = 3
                   , n.minobsinnode = 100
                   , shrinkage = 0.01
                   , bag.fraction = 0.5
                   , train.fraction = nrow(train_data) / (nrow(train_data) + nrow(test_data))
  )
)
# Determine best iteration based on test data
gbm.iter = gbm.perf(model_gbm, method = "test")
