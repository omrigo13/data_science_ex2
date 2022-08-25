library(arules) # discretize
library(caret) # splitting to train and test sets
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)

#### ---1. data preparation---####


fill_NA_values = function(dataset){
  for (col in colnames(dataset)){
    if(is.numeric(dataset[,col])){ #check if numeric-fill value with mean
      dataset[is.na(dataset[,col]), col] = round(mean(dataset[,col], na.rm=TRUE))
    }
    else{# else is categorized- fill value with most common value
      dataset[is.na(dataset[,col]), col] = names(sort(table(dataset[,col]), decreasing = TRUE))[1] 
    }
  }
  return(dataset)
}

#discretize spouse income by fixed to 3 bins
discretize_spouse_income=function(dataset){
  meanIncome = mean(dataset$Spouse_Income, na.rm=TRUE) # calculate the mean of spuse income
  dataset$Spouse_Income<-discretize(dataset$Spouse_Income, method = "fixed", breaks = c(0, 0.1, meanIncome, Inf), 
                               labels = c("zero","lessEqualAVG" ,"higherThenAVG"))
  return(dataset)
}



## 1.1 read csv file 
# allow user to choose a file from library  
print ("please choose the requested file in the pop up window")
file_path = file.choose() #choosing input file
# read the csv file and convert empty values as NA
DF_LoanDataset <- read.csv(file_path,na.strings=c("","NA")) 

## 1.2 fill NA values
# fill numeric values with average
#fill Categorical values with the most common value
DF_LoanDataset<-fill_NA_values(dataset = DF_LoanDataset)

##1.3 Discretization 

# Monthly_Profit
DF_LoanDataset$Monthly_Profit<-discretize(DF_LoanDataset$Monthly_Profit, method="frequency", breaks = 5, labels = c("Low", "LowerThenAVG", "AVG", "HigherThenAVG", "High"))

#Spouse_Income
DF_LoanDataset<-discretize_spouse_income(dataset = DF_LoanDataset)

#Loan_Amount
DF_LoanDataset$Loan_Amount<-discretize(DF_LoanDataset$Loan_Amount, method="frequency", breaks = 5, labels = c("Low", "LowerThenAVG", "AVG", "HigherThenAVG", "High"))

##1.4 Factor - change categorical to factors
DF_LoanDataset$Spouse_Income<-factor(DF_LoanDataset$Spouse_Income)

DF_LoanDataset$Request_Approved<-factor(DF_LoanDataset$Request_Approved)

## 1.5 training and testing set 
splitInTrain<-createDataPartition(y=DF_LoanDataset$Request_Approved, p=0.7, list = FALSE)
trainingSet<-DF_LoanDataset[splitInTrain,]
testingSet<-DF_LoanDataset[-splitInTrain,]

trainingSet$Request_Approved<- as.factor(trainingSet$Request_Approved)
testingSet$Request_Approved<- as.factor(testingSet$Request_Approved)



####----2. building model---####
#2.1 
# will remove Request_Number column
trainingSet<- subset(trainingSet, select= -c(Request_Number))
testingSet<- subset(testingSet, select= -c(Request_Number))

#split tree by gini 
tree_gini_default = rpart(data = trainingSet, trainingSet$Request_Approved~.,
                  parms=list(split="gini"), cp = 0.0045)
fancyRpartPlot(tree_gini_default) # plot the tree 
#calculate the tree accuracy 
tree_accuracy_gini_default = 1 -mean(predict(tree_gini_default, type="class") != trainingSet$Request_Approved)

#split tree by information gain
tree_information_gain_default = rpart(data = trainingSet, trainingSet$Request_Approved~.,
                     parms=list(split="information"), cp = 0.0045)
fancyRpartPlot(tree_information_gain_default)#plot the tree
#calculate the tree accuracy 
tree_accuracy_gain_information1 = 1 -mean(predict(tree_information_gain_default, type="class") != trainingSet$Request_Approved)

#2.2.1 comparing minsplit- 2,50

#gini minsplit=2
tree_gini_minsplit_2 = rpart(data = trainingSet, trainingSet$Request_Approved~.,
                             parms=list(split="gini"), cp = 0.007, minsplit = 2)
tree_accuracy_gini2 = 1 -mean(predict(tree_gini_minsplit_2, type="class") != trainingSet$Request_Approved)
fancyRpartPlot(tree_gini_minsplit_2)

#gini minslpit=50
tree_gini_minsplit_50 = rpart(data = trainingSet, trainingSet$Request_Approved~.,
                             parms=list(split="gini"), cp = 0.007, minsplit = 50)
tree_accuracy_gini50 = 1 -mean(predict(tree_gini_minsplit_50, type="class") != trainingSet$Request_Approved)
fancyRpartPlot(tree_gini_minsplit_50)


#2.1.4 tree complexity 
# check tree with cp=0.1 

#cp=0.1, minsplit=20
tree_gini_comp1 = rpart(data = trainingSet, trainingSet$Request_Approved~.,
                    parms=list(split="gini"), cp = 0.1)
tree_accuracy_gini_comp1 = 1 -mean(predict(tree_gini_comp1, type="class") != trainingSet$Request_Approved)

tree_gini_comp01 = rpart(data = trainingSet, trainingSet$Request_Approved~.,
                        parms=list(split="gini"), cp = 0.001)
tree_accuracy_gini_comp01 = 1 -mean(predict(tree_gini_comp01, type="class") != trainingSet$Request_Approved)

#print("tree complexity")
#print("cp=0.1, minsplit=20")
#print(sprintf("gini: %f",tree_accuracy_gini_comp1))
#print("cp=0.001, minsplit=20")
#print(sprintf("gini: %f",tree_accuracy_gini_comp01))

####---3 testing model---####

#predict testing set on the trees we build before 
#gini, cp=0.0045, minsplit=20
tree_gini_predict_default = predict(tree_gini_default, testingSet, type="class")
metrix_gini=confusionMatrix(tree_gini_predict_default, testingSet$Request_Approved)

#information gain, cp=0.0045 minsplit=20
tree_ig_predict_default = predict(tree_information_gain_default, testingSet, type="class")
metrix_ig=confusionMatrix(tree_ig_predict_default, testingSet$Request_Approved)

#gini, cp=0.007,minsplit=2
tree_gini2_predict = predict(tree_gini_minsplit_2, testingSet, type="class")
metrix_gini_2=confusionMatrix(tree_gini2_predict, testingSet$Request_Approved)

#gini, cp=0.007,minsplit=50
tree_gini50_predict = predict(tree_gini_minsplit_50, testingSet, type="class")
metrix_gini_50=confusionMatrix(tree_gini50_predict, testingSet$Request_Approved)

print(metrix_gini)
print(metrix_ig)
print(metrix_gini_2)
print(metrix_gini_50)
