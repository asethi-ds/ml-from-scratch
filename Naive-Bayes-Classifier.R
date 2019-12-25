#Here is an implementation of Naive Bayes Classifier with few simplifying assumptions. We use popular Titanic Dataset
#which has been preprocessed to contain only 2 categorical independent variables.


CalculatePosterior<-function(A_Prior,B_Prior,Post_Train,Post_Test){
  
  #We initialize the posterior probailities as repetitive prior probabilities calculated before, we chose this
  #instead of chosing NA since it aides in multiplicative tasks ahead
  ClassA_Post <- rep(A_Prior, nrow(Post_Test))
  ClassB_Post <- rep(B_Prior, nrow(Post_Test))
  
  for (i in 1:nrow(Post_Test)) { ##Now coming to test set....Can Be read as "for every tuple in test set"
    for (j in 1:ncol(Post_Test)) { ## "for every feature"
      #We calculate likelyhood for each
      count = length(which(Post_Train[,j] == Post_Test[i,j])) #the count of conditional occurances
      cat_div = nrows(Post.Train) + length(unique(rbind(Post_Test[i,j], Post_Train[i,j])))
      #divisor for distinct categories
      
      ## final posterior   = prior*Likelyhood
      ClassA_Post[i] = ClassA_Post[i] * (count / cat_div)
      ClassB_Post[i] = ClassB_Post[i] * (count / cat_div)
    }
    #initializing the vector that predicts the class label
    final.class.pred = vector()

   #If posterior probability is greater(or equal) than Class A
    if (ClassA_Post[i] >= ClassB_Post[i]){
      final.class.pred[i] = 1 #Predict Class A
  } else {
    final.class.pred[i] = 0 #Predict Class B
  } 
  cat(final.class.pred) #Outputing Final Class
}


###########################################################################
CalculatePrior<-function(Prior_Train,Prior_Test){
  #Prior Probability is simply the 'outer probability' The below statement can be read as, 'Number of counts of
  #ClassA occurances divided by total observations in training test
  ClassA_Prior = which(Prior_Train[, (ncol(Prior_Train))] == 1) / nrow(Prior_Train)
  ClassB_Prior = which(Prior_Test[, (ncol(Prior_Train))] == 0) / nrow(Prior_Test)
  #Since now we have prior probability we can call the function to calculate likelyhood and thereafter posterior
  #probability
  CalculatePosterior(ClassA_Prior,ClassB_Prior,Prior_Train,Prior_Test)
}


#############################################################################
#To test the algorithm we have taken a customised subset of popular Titianic Passengers Data
#Out Independent variables are Sex and Embarked, we predict survival rate

DataPrepare <- function(){
  #We split the data randomly (using dependent variable)
  library(caTools)
  DataFrame = read.csv("TitanicPreprocessed.csv")
  DataFrame<-DataFrame[,c(5,12,13)]
  spl = sample.split(DataFrame$Class, SplitRatio = 0.75)
  DataSet_Train = subset(DataFrame, spl == TRUE)
  Dataset_Test = subset(DataFrame, spl == FALSE)
  #Since we have obtained two seperate datasets we now call the function to calculate Prior Probability
  CalculatePrior(DataSet_Train,Dataset_Test)
}


##################################################################################
#Invoking the function to prepare data
DataPrepare() 
