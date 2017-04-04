library(caTools)
library(rpart)
library(rpart.plot)
library(mice)
library(DMwR)
library(car)
library(data.table)
library(ROCR)
library(randomForest)
library(FSelector)
library(xgboost)
library(MASS)
library(e1071)

#reading files .

test = read.csv("F:/dataset/titanic/test.csv")
train = read.csv("F:/dataset/titanic/train.csv")

#processing the data .

train$SibSp = as.numeric(train$SibSp)
train$Parch = as.numeric(train$Parch)
train$Ticket = as.numeric(train$Ticket)
train$Cabin = as.numeric(train$Cabin)
train$Embarked = as.numeric(train$Embarked)
train$Sex = as.numeric(train$Sex)
train$Name = as.numeric(train$Name)

#imputing missing values with KNN.

new_data = knnImputation( train , k=10 )

#spliting data .

split = sample.split( new_data$Survived , SplitRatio = 0.75)
mtrain = subset(new_data , split==TRUE)
mtest = subset(new_data, split==FALSE)

mtrain = as.data.frame(mtrain)
mtest = as.data.frame(mtest)

#baseline model with decision tree .

model = rpart(Survived ~ .  , data = mtrain , method = "class" )
guess = predict( model , newdata = mtest , type = "prob")
pred = prediction(guess[,2] , mtest$Survived )
pref = performance( pred , "tpr" , "fpr" )
plot(pref)
auc.tmp = performance(pred,"auc")
auc = as.numeric(auc.tmp@y.values)
table( mtest$Survived , guess[,2]>0.5)
mtrain=as.matrix(mtrain)
mtest = as.matrix(mtest)



#applying XGboost.

bst = xgboost(data = mtrain[,-2] , label = mtrain[,2] , nrounds = 200, objective = "binary:logistic")
pred = predict( bst , mtest[,-2] , type="prob")
predi = prediction( pred , mtest[,2])
bst_pref = performance( predi , "tpr" , "fpr")
plot(bst_pref)
table(mtest[,2] , pred>0.5)
auc.tmp = performance(predi,"auc")
auc = as.numeric(auc.tmp@y.values)

#gives test error and train error of every round ,this is used to find the optimum number of roundes .

dtrain <- xgb.DMatrix(data = mtrain[,-2], label=mtrain[,2])
dtest <- xgb.DMatrix(data = mtest[,-2], label=mtest[,2])
watchlist = list(train=dtrain, test=dtest)
bstt = xgb.train(data=dtrain , nrounds=300,watchlist=watchlist, objective = "binary:logistic")



#applying PCA .

prin_comp = prcomp(mtrain, scale. = T)
biplot(prin_comp, scale = 0)
std_dev = prin_comp$sdev
pr_var = std_dev^2
prop_varex = pr_var/sum(pr_var)

#Scee plots to find the optimum number of components .

plot(prop_varex, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b")
plot(cumsum(prop_varex), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")

#predictive modeling on the PCAed data .

train.data = data.frame(sur = mtrain[,2], prin_comp$x)#adding a training set with the above PCA's
train.data = train.data[,1:10]
rpart.model = rpart(sur ~ . ,data = train.data, method = "anova")
test.data = predict(prin_comp, newdata = mtest)
test.data = as.data.frame(test.data)
test.data = test.data[,1:10]
rpart.prediction = predict(rpart.model, test.data )
rpart.predi = prediction( rpart.prediction , mtest[,2])
curve = performance(rpart.predi  , "tpr" , "fpr") 
plot(curve)
auc.tmp = performance(rpart.predi,"auc")
auc = as.numeric(auc.tmp@y.values)
table(mtest[,2] , rpart.prediction>0.5)


#applying SVM

msvm = svm(Survived ~ . , data = mtrain)
predsvm = predict(msvm , data = mtest)


#parameter tuning.

svm_tune <- tune(svm, train.x=mtrain[,-2] , train.y = mtrain[,2] ,kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)))
print(svm_tune)

#creating model with tuned parameters .

svm_model_after_tune = svm(Survived ~ ., data=mtrain, kernel="radial", cost=1, gamma=0.5)
summary(svm_model_after_tune)
svmpred = predict( svm_model_after_tune ,  newdata = mtest , type = "prob")
svmpredi = prediction( svmpred , mtest[,2])
svmcurve = performance(svmpredi  , "tpr" , "fpr") 
plot(svmcurve)
auc.tmp = performance(svmpredi,"auc")
auc = as.numeric(auc.tmp@y.values)
table(mtest[,2] , svmpred>0.5)

#predictive modeling on the PCAed data using SVM .
train.data = data.frame(sur = mtrain[,2], prin_comp$x)#adding a training set with the above PCA's
train.data = train.data[,1:10]
svm.model = svm(sur ~ . ,data = train.data, method = "anova")
test.data = predict(prin_comp, newdata = mtest)
test.data = as.data.frame(test.data)
test.data = test.data[,1:10]
svm.prediction = predict(svm.model, test.data )
svm.predi = prediction( svm.prediction , mtest[,2])
svm.curve = performance(svm.predi , "tpr" , "fpr") 
plot(svm.curve)
auc.tmp = performance(svm.predi,"auc")
auc = as.numeric(auc.tmp@y.values)
table(mtest[,2] , svm.prediction>0.5)














