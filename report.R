library(ggplot2)
library(dplyr)
library(caret)
library(randomForest)
library(e1071)
library(pROC)
library(reshape2)

df <- read.csv("heart_failure.csv")

summary(df)
str(df)

print(
  ggplot(df, aes(x = factor(fatal_mi))) +
    geom_bar(fill = "lightblue") +
    labs(title = "Target Variable Distribution", x = "Survival (0 = Alive, 1 = Deceased)", y = "Count")
)

print(
  ggplot(df, aes(x = ejection_fraction)) +
    geom_histogram(fill = "lightgreen", bins = 20, color = "black") +
    labs(title = "Ejection Fraction Distribution", x = "Ejection Fraction (%)", y = "Count")
)

print(
  ggplot(df, aes(x = serum_creatinine)) +
    geom_histogram(fill = "salmon", bins = 20, color = "black") +
    labs(title = "Serum Creatinine Distribution", x = "Serum Creatinine (mg/dL)", y = "Count")
)

df$fatal_mi <- as.factor(df$fatal_mi)

set.seed(42)
trainIndex <- createDataPartition(df$fatal_mi, p = 0.8, list = FALSE)
trainData <- df[trainIndex, ]
testData <- df[-trainIndex, ]

numeric_features <- c("creatinine_phosphokinase", "ejection_fraction", "platelets", "serum_creatinine", "serum_sodium", "time")
preProcValues <- preProcess(trainData[, numeric_features], method = c("center", "scale"))
trainData[, numeric_features] <- predict(preProcValues, trainData[, numeric_features])
testData[, numeric_features] <- predict(preProcValues, testData[, numeric_features])

logit_model <- glm(fatal_mi ~ ., data = trainData, family = binomial)

rf_model <- randomForest(fatal_mi ~ ., data = trainData, ntree = 100, importance = TRUE)

svm_model <- svm(fatal_mi ~ ., data = trainData, probability = TRUE)

y_pred_logit <- predict(logit_model, testData, type = "response")
y_pred_rf <- predict(rf_model, testData, type = "prob")[,2]  # Extract probability of death
y_pred_svm <- predict(svm_model, testData, probability = TRUE)

y_pred_svm_prob <- attr(y_pred_svm, "probabilities")
if (is.null(y_pred_svm_prob)) {
  stop("SVM predictions did not return probabilities. Please check model parameters.")
}

auc_logit <- roc(testData$fatal_mi, y_pred_logit)$auc
auc_rf <- roc(testData$fatal_mi, y_pred_rf)$auc
auc_svm <- roc(testData$fatal_mi, y_pred_svm_prob[,2])$auc

feature_importance <- as.data.frame(importance(rf_model))
feature_importance$Feature <- rownames(feature_importance)
print(
  ggplot(feature_importance, aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    labs(title = "Feature Importance (Random Forest)", x = "Feature", y = "Importance Score")
)

print(
  ggplot() +
    geom_line(aes(x = roc(testData$fatal_mi, y_pred_logit)$specificities, 
                  y = roc(testData$fatal_mi, y_pred_logit)$sensitivities, 
                  color = "Logistic Regression")) +
    geom_line(aes(x = roc(testData$fatal_mi, y_pred_rf)$specificities, 
                  y = roc(testData$fatal_mi, y_pred_rf)$sensitivities, 
                  color = "Random Forest")) +
    geom_line(aes(x = roc(testData$fatal_mi, y_pred_svm_prob[,2])$specificities, 
                  y = roc(testData$fatal_mi, y_pred_svm_prob[,2])$sensitivities, 
                  color = "SVM")) +
    scale_color_manual(values = c("Logistic Regression" = "blue", 
                                  "Random Forest" = "red", 
                                  "SVM" = "green")) +
    labs(title = "ROC Curve Comparison", x = "1 - Specificity", y = "Sensitivity")
)

conf_matrix <- confusionMatrix(predict(rf_model, testData), testData$fatal_mi)
cm_df <- as.data.frame(conf_matrix$table)

print(
  ggplot(cm_df, aes(Prediction, Reference, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = Freq), vjust = 1) +
    scale_fill_gradient(low = "white", high = "red") +
    labs(title = "Confusion Matrix (Random Forest)", x = "Predicted", y = "Actual")
)

auc_list <- list(Logistic_Regression = auc_logit, Random_Forest = auc_rf, SVM = auc_svm)
print(auc_list)