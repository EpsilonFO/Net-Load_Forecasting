rm(list=objects())
# Load necessary libraries
library(mgcv)
library(tidyverse)
library(randomForest)
library(xgboost)

#########################
### Import et Prepro ###
#########################
# Load the data
train <- read_csv('Data/train.csv')
test <- read_csv('Data/test.csv')

# Preprocess the data
Data0 <- train
Data0$Time <- as.numeric(Data0$Date)
Data1 <- test
Data1$Time <- as.numeric(Data1$Date)

# Convert categorical variables to factors
Data0$WeekDays <- as.factor(Data0$WeekDays)
Data1$WeekDays <- as.factor(Data1$WeekDays)

# Define the equation for the GAM model with advanced techniques
equation <- Net_demand ~
  s(Time, k = 3, bs = 'cr') +
  s(toy, k = 30, bs = 'cc') +
  ti(Temp, k = 10, bs = 'cr') +
  ti(Temp_s99, k = 10, bs = 'cr') +
  s(Load.1, bs = 'cr') +
  s(Load.7, bs = 'cr') +
  ti(Temp_s99, Temp, bs = c('cr', 'cr'), k = c(10, 10)) +
  as.factor(WeekDays) + BH +
  te(Temp_s95_max, Temp_s99_max) +
  Summer_break + Christmas_break +
  te(Temp_s95_min, Temp_s99_min) +
  s(Wind, bs = 'cr') +
  ti(Nebulosity_weighted) +
  ti(Wind_weighted, Temp, bs = 'ts')

formula <- Net_demand ~ Time + toy + Temp + Temp_s99 + Load.1 + Load.7 +
  WeekDays + BH + Temp_s95_max * Temp_s99_max +
  Summer_break + Christmas_break + Temp_s95_min * Temp_s99_min +
  Wind + Nebulosity_weighted + Wind_weighted * Temp


############
### GAM ###
############
# Train the GAM model with shrinkage
gam_model <- gam(equation, data = Data0, select = TRUE, gamma = 1.5)

# Make predictions on the test data
gam_forecast <- predict(gam_model, newdata = Data1)
###########
### RF ###
###########
# Train rf
rf_model = randomForest(formula, data=Data0)

# Make predictions with rf on the test data
rf_forecast = predict(rf_model, data=Data1)

### RF quantile ###
qrf<- ranger::ranger(formula, data = Data0, importance ='permutation', seed=1, quantreg=TRUE)
quant=0.9
qrf_forecast <- predict(qrf, data=Data1, quantiles =quant, type = "quantiles")$predictions
#pinball_loss(Data0[sel_b,]$Load, qrf.forecast,quant) 

#plot(Data0[sel_b,]$Load,  type='l')
#lines(qrf.forecast, col='blue')



###########################
### Submission for all ###
###########################

# Load the sample submission file
submit <- read_delim(file = "Data/sample_submission.csv", delim = ",")

# Assign the forecasted values to the submission file
submit$Net_demand <- gam_forecast

# Write the submission file to CSV
write.table(submit, file = "Data/submission_rf.csv", quote = FALSE, sep = ",", dec = '.', row.names = FALSE)
