getwd()
setwd("c:/Users/Atul/Desktop/")

### Reading the Data and Test files.====

oridata <- read.csv("train.csv")
tdata <- oridata

test <- read.csv("test.csv") # Reading test

### Data Exploration & Typecasting====

dim(tdata)
head(tdata)
tail(tdata)

str(tdata)

summary(tdata)


min(tdata$rally)
levels(oridata$outcome)

unique(oridata$gender)

tdata$ID <- NULL

test$ID -> ID
test$ID <- NULL # Dropping 

tdata$serve <- as.factor(tdata$serve) # Typecasting
test$serve <- as.factor(test$serve)

tdata[,c(8,9,21,24)] <- lapply(tdata[,c(8,9,21,24)],as.factor)

test[,c(8,9,21,24)] <- lapply(test[,c(8,9,21,24)],as.factor)


# Splitting into Train and Validation====

library(caret)

set.seed(1103)

train_rows <- createDataPartition(tdata$outcome, p = 0.75, list = F)
train <- tdata[train_rows, ]
val<-tdata[-train_rows, ]

# SCaling====

IDV <- setdiff(names(tdata), c("outcome"))  # Seperating the DV

std_model <- preProcess(train[, IDV], method = c("center","scale"))

train[, IDV] <- predict(object = std_model, newdata = train[, IDV])
val[, IDV] <- predict(object = std_model, newdata = val[,IDV])

test <- predict(object = std_model, newdata = test)

### Model1 - Multinomial Logistic Regression====

library(nnet)

model1 <- multinom(formula = train$outcome ~ ., data = train)
summary(model1)

# Train Predictions
model1preds <- predict(model1, newdata = train)
confusionMatrix(model1preds, train$outcome)
# 82.62

# Validation Predictions
model1valpreds <- predict(model1, newdata = val)
confusionMatrix(model1valpreds, val$outcome)
# 80.34

# Test Predictions
model1testpreds <- predict(model1, newdata = test)
# 0.82

# Writing to csv for submission try
try1 <- cbind(ID, model1testpreds)
write.csv(x = try1, file = "try2.csv", row.names = F)

### Model 2 - Naive-Bayes====

###create dummies for factor varibales 
dummies <- dummyVars(outcome~., data = train)


x.train=predict(dummies, newdata = train)
y.train=train$outcome
x.val = predict(dummies, newdata = val)
y.val = val$outcome

dummies1 <- dummyVars(~., data = test)
x.test <- predict(dummies1, newdata = test)

# install.packages("e1071")
library(e1071)

model2 <- naiveBayes(x = x.train, y = y.train)
model2

# Train Predictions
model2preds  =  predict(model2, x.train) # x is all the input variables
confusionMatrix(model2preds,y.train)
# 67.36

# Validation Predictions
model2valpreds=predict(model2,x.val)
confusionMatrix(model2valpreds,y.val)
# 67.33

# Test Predictions
model2testpreds <- predict(model2, newdata = test)
# 0.67

# Writing to csv for submission try
try2 <- cbind(ID, model2testpreds)
write.csv(x = try2, file = "try3.csv", row.names = F)

### Model 3 - Decision Trees====

install.packages("C50")
library(C50)

model3 <- C5.0(outcome~.,data=train,rules=TRUE)
summary(model3)
write(capture.output(summary(model3)), "c50model1.txt")

# Train Predictions
model3preds <- predict(model3,newdata=train, type="class")
confusionMatrix(model3preds, train$outcome)
# 90.19

# Validation Predictions
model3valpreds <- predict(model3, newdata=val, type="class")
confusionMatrix(model3valpreds, val$outcome)
# 82.74

# Test Predictions
model3testpreds <- predict(model3, newdata = test)
# 0.83

# Writing to csv for submission try
try3 <- cbind(ID, model3testpreds)
write.csv(x = try3, file = "try4.csv", row.names = F)

#c. Check variable importance
C5imp(model3, pct=TRUE)


### Model 4 - KNN====

library(class)

#Deciding k value for k-NN

#Experiment with various odd values of k; k={1,3,5,7,..}

# k = 1
noOfNeigh <- 1
pred = knn(x.train, x.val,y.train, k = noOfNeigh)
a = confusionMatrix(pred,y.val)
a

#k = 3

noOfNeigh <- 3
pred = knn(x.train, x.val,y.train, k = noOfNeigh)
a = confusionMatrix(pred,y.val)
a

# k = 5

noOfNeigh <- 5
pred = knn(x.train, x.val,y.train, k = noOfNeigh)
a = confusionMatrix(pred,y.val)
a

noOfNeigh <- 11
pred = knn(x.train, x.val,y.train, k = noOfNeigh)
a = confusionMatrix(pred,y.val)
a

 noOfNeigh <- 19
 pred = knn(x.train, x.val,y.train, k = noOfNeigh)
 a = confusionMatrix(pred,y.val)
 a

# noOfNeigh <- 23
# pred = knn(x.train, x.val,y.train, k = noOfNeigh)
# a = confusionMatrix(pred,y.val)
# a
 
# Validation Predictions 
model4valpreds <- knn(x.train, x.val, y.train, k = noOfNeigh) 
confusionMatrix(model4valpreds, y.val)
# 77.54

# Test Predictions
model4testpreds <- knn(x.train, x.test,y.train, k = noOfNeigh)
# 0.77

# Writing to csv for submission try
try4 <- cbind(ID, model4testpreds)
write.csv(x = try4, file = "try5.csv", row.names = F)


### Model 5 - SVM====
?svm
model5  =  svm(x = x.train, y = y.train, type = "C-classification", kernel = "linear", cost = 10)
summary(model5)

# Train Predictions
model5preds  =  predict(model5, x.train) # x is all the input variables
confusionMatrix(model5preds,y.train)
# 83.37

# Validation Predictions
model5valpreds=predict(model5,x.val)
confusionMatrix(model5valpreds,y.val)
# 81.09

# Test Predictions
model5testpreds <- predict(model5, x.test)
# 0.83

# Writing to csv for submission try
try5 <- cbind(ID, model5testpreds)
write.csv(x = try5, file = "try6.csv", row.names = F)

#### Model 6 - Svm with cost 20====

model6  =  svm(x = x.train, y = y.train, type = "C-classification", kernel = "linear", cost = 20)
summary(model6)

# Train Predictions
model6preds  =  predict(model6, x.train) # x is all the input variables
confusionMatrix(model6preds,y.train)
# 83.42

# Validation Predictions
model6valpreds=predict(model6,x.val)
confusionMatrix(model6valpreds,y.val)
# 81.09

# Test Predictions
model6testpreds <- predict(model6, x.test)
# 0.83

# Writing to csv for submission try
try6 <- cbind(ID, model6testpreds)
write.csv(x = try6, file = "try7.csv", row.names = F)


### Model 7 - SVM with kernel-radial====

model7  =  svm(x = x.train, y = y.train, type = "C-classification", kernel = "radial", cost = 20)
summary(model7)

# Train Predictions
model7preds  =  predict(model7, x.train) # x is all the input variables
confusionMatrix(model7preds,y.train)
# 98.1

# Validation Predictions
model7valpreds=predict(model7,x.val)
confusionMatrix(model7valpreds,y.val)
# 82.24

# Test Predictions
model7testpreds <- predict(model7, x.test)
# 0.85

# Writing to csv for submission try
try7 <- cbind(ID, model7testpreds)
write.csv(x = try7, file = "try8.csv", row.names = F)


### Model 8 - SVM with kernel-polynomial====

model8  =  svm(x = x.train, y = y.train, type = "C-classification", kernel = "polynomial", cost = 20)
summary(model8)

# Train Predictions
model8preds  =  predict(model8, x.train) # x is all the input variables
confusionMatrix(model8preds,y.train)
# 97.07

# Validation Predictions
model8valpreds=predict(model8,x.val)
confusionMatrix(model8valpreds,y.val)
# 79.69

# Test Predictions
model8testpreds <- predict(model8, x.test)
# 0.82

# Writing to csv for submission try
try8 <- cbind(ID, model8testpreds)
write.csv(x = try8, file = "try9.csv", row.names = F)


### Model 9 - Random Forests====

rfctrl <- trainControl(method="cv", number=10)
set.seed(1235869)
rf_grid1 <- expand.grid(mtry=c(1:15))

model9 <- train(outcome ~ ., data=train, method = "rf",trControl=rfctrl, tuneGrid = rf_grid1)
model9$finalModel
model9$bestTune


# Train Predictions
model9preds <- predict(model9)
confusionMatrix(model9preds, train$outcome)
# 100

# Validation Predictions
model9valpreds <- predict(model9, val)
confusionMatrix(model9valpreds, val$outcome)
# 86.49

# Test Predictions
model9testpreds <- predict(model9, test)
# 0.88

# Writing to csv for submission try
try9 <- cbind(ID, model9testpreds)
write.csv(x = try9, file = "try10.csv", row.names = F)

### Model 10 - XG Boost====

xgb_grid_1 = expand.grid(
  nrounds = 1000,
  eta = c(0.01, 0.001, 0.0001),
  max_depth = c(2, 4, 6, 8, 10),
  gamma = 1,
  colsample_bytree=0.7,
  min_child_weight=2,
  subsample=0.9
)

# pack the training control parameters
xgb_trcontrol_1 = trainControl(
  method = "cv",
  number = 5
)

# train the model for each parameter combination in the grid, 
#   using CV to evaluate
model10 = train(outcome~., data=train,
                    trControl = xgb_trcontrol_1,
                    tuneGrid = xgb_grid_1,
                    method = "xgbTree"
)

# Train Predictions
model10preds <- predict(model10, train)
confusionMatrix(model10preds, train$outcome)
# 97.05

# Validation Predictions
model10valpreds <- predict(model10, val)
confusionMatrix(model10valpreds, val$outcome)
# 87.14

# Test Predictions
model10testpreds <- predict(model10, test)
# 0.89

# Writing to csv for submission try
try10 <- cbind(ID, model10testpreds)
write.csv(x = try10, file = "try11.csv", row.names = F)

### Model 11 - GBM====

# Using caret
gbm_grid1 <- expand.grid(interaction.depth=c(1, 3, 5), n.trees = (0:50)*50,
                         shrinkage=c(0.01, 0.001),
                         n.minobsinnode=10)
gbmctrl <- trainControl(method="cv", number=10)

set.seed(99)
model11 <- train(outcome~., data = train, method ="gbm",trControl = gbmctrl, tuneGrid = gbm_grid1)   

# Train Predictions
model11preds <- predict(model11, train)
confusionMatrix(model11preds, train$outcome)
# 93.77

# Validation Predictions
model11valpreds <- predict(model11, val)
confusionMatrix(model11valpreds, val$outcome)
# 86.54

# Test Predictions
model11testpreds <- predict(model11, test)
# 0.88

# Writing to csv for submission try
try11 <- cbind(ID, model11testpreds)
write.csv(x = try11, file = "try12.csv", row.names = F)

model11$finalModel

### Model12 - ADA Boost====

install.packages("adabag")
library(adabag)

adactrl <- trainControl(method = "cv", number = 5)

ada_grid1 <- expand.grid(iter = c(50,100,150,200),maxdepth=7:10,nu=c(0.1,0.5,0.9))
set.seed(3233)

model12 <- boosting.cv(formula = outcome~., data = train, v = 5, boos = T, coeflearn = "Zhu", mfinal = 100, control = rpart.control(c(7:10)))
#model12 <- train(x.train, y.train, method ="adaboost", trControl = adactrl, tuneGrid = ada_grid1)

# Train Predictions
model12preds <- predict(model12, train)
confusionMatrix(model12preds, train$outcome)
# 

# Validation Predictions
model12valpreds <- predict(model12, val)
confusionMatrix(model12valpreds, val$outcome)
# 

# Test Predictions
model12testpreds <- predict(model12, test)
# 

# Writing to csv for submission try
try12 <- cbind(ID, model12testpreds)
write.csv(x = try12, file = "try13.csv", row.names = F)
# 

### PCA====

pcavars <- princomp(x.train)

plot(pcavars)
pcavars$loadings
names(pcavars)


summary(pcavars)

comp_Names = c("Comp.1","Comp.2","Comp.3","Comp.4","Comp.5","Comp.6","Comp.7","Comp.8","Comp.9","Comp.10","Comp.11","Comp.12","Comp.13","Comp.14","Comp.15","Comp.16")

# transformed data in new components
train_pca<-pcavars$scores

# get only highest variation components and bind target
train_pca<-data.frame(train_pca[,comp_Names],"outcome"=train$outcome)

# Applying same transformation on validation
val_pca<-predict(pcavars,x.val)
val_pca<-data.frame(val_pca[,comp_Names],"outcome"=val$outcome)

# Applying the same transformation on Test

test_pca<-predict(pcavars,x.test)
test_pca<-data.frame(test_pca[,comp_Names])

### Model13 - Logistic with PCA====

model13 <- multinom(formula = train_pca$outcome ~ ., data = train_pca)
summary(model13)

# Train Predictions
model13preds <- predict(model13, newdata = train_pca)
confusionMatrix(model13preds, train_pca$outcome)
# 68.39

# Validation Predictions
model13valpreds <- predict(model13, newdata = val_pca)
confusionMatrix(model13valpreds, val_pca$outcome)
# 67.13

# Test Predictions
model13testpreds <- predict(model13, newdata = test_pca)
# 0.68

# Writing to csv for submission try
try13 <- cbind(ID, model13testpreds)
write.csv(x = try13, file = "try14.csv", row.names = F)


### Model 14 - Naive-Bayes with PCA====

# install.packages("e1071")
library(e1071)

model14 <- naiveBayes(x = train_pca[,setdiff(names(train_pca), c("outcome"))], y = y.train)
model14

# Train Predictions
model14preds  =  predict(model14, train_pca)
confusionMatrix(model14preds,y.train)
# 63.56

# Validation Predictions
model14valpreds=predict(model14,val_pca)
confusionMatrix(model14valpreds,y.val)
# 63.58

# Test Predictions
model14testpreds <- predict(model14, newdata = test_pca)

# Writing to csv for submission try
try14 <- cbind(ID, model14testpreds)
write.csv(x = try14, file = "try15.csv", row.names = F)
# 1.0

### Model 15 - Decision Trees with PCA====

#install.packages("C50")
library(C50)

#a. Build model
model15 <- C5.0(outcome~.,data=train_pca,rules=TRUE)
summary(model15)
write(capture.output(summary(model15)), "c50model15.txt")

# Train Predictions
model15preds <- predict(model15,newdata=train_pca, type="class")
confusionMatrix(model15preds, train_pca$outcome)
# 80.06

# Validation Predictions
model15valpreds <- predict(model15, newdata=val_pca, type="class")
confusionMatrix(model15valpreds, val_pca$outcome)
# 65.88

# Test Predictions
model15testpreds <- predict(model15, newdata = test_pca)
# 66

# Writing to csv for submission try
try15 <- cbind(ID, model15testpreds)
write.csv(x = try15, file = "try16.csv", row.names = F)

# Checking variable importance
C5imp(model15, pct=TRUE)

### Stacking====

stackvars <- cbind(model1valpreds, model2valpreds, model3valpreds, model4valpreds, model5valpreds, model6valpreds, model7valpreds, model8valpreds, model9valpreds, model10valpreds, model11valpreds, y.val)
stackvars <- as.data.frame(stackvars)
write.csv(x = stackvars, file = "stackvars.csv", row.names = F)
#stackvars[,c(1:12)] <- as.factor(stackvars[,c(1:12)])
stcakdata <- read.csv("stackvars.csv")

model16 <- multinom(formula = stcakdata$y.val ~ ., data = stcakdata)
summary(model16)

# Train Predictions
model16preds <- predict(model16, newdata = stcakdata)
confusionMatrix(model16preds, stcakdata$y.val)
# 87.54

# Test Predictions
model16testpreds <- predict(model16, newdata = test)
# 0.82

# Writing to csv for submission try
try16 <- cbind(ID, model16testpreds)
write.csv(x = try16, file = "try17.csv", row.names = F)


###

model17 <- svm(formula = stcakdata$y.val ~ ., data = stcakdata)
summary(model17)

# Train Predictions
model17preds <- predict(model17, newdata = stcakdata)
confusionMatrix(model17preds, stcakdata$y.val)
# 87.54

# Test Predictions
model17testpreds <- predict(model17, newdata = test)
# 0.37

# Writing to csv for submission try
try17 <- cbind(ID, model17testpreds)
write.csv(x = try17, file = "try18.csv", row.names = F)




###############################################################

?cor()
install.packages("corrplot")
library(corrplot)

cor_matrix <- cor(tdata[,c("previous.speed", "previous.time.to.net", "opponent.depth")])
par(mfrow = c(1,1))
corrplot(cor_matrix, method = "number")



votevars <- cbind(model1testpreds, model2testpreds, model3testpreds, model4testpreds, model5testpreds, model6testpreds, model7testpreds, model8testpreds, model9testpreds, model10testpreds, model11testpreds)
votevars <- as.data.frame(votevars)
write.csv(x = votevars, file = "votevars.csv", row.names = F)
