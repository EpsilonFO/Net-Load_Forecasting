# Load necessary libraries
library(mgcv)
library(tidyverse)

# Load the data
train <- read_csv('Data/train.csv')
test <- read_csv('Data/test.csv')

# Preprocess the data
Data0 <- train
Data0$Time <- as.numeric(Data0$Date)
Data1 <- test
Data1$Time <- as.numeric(Data1$Date)

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

# Train the GAM model with shrinkage
gam_model <- gam(equation, data = Data0, select = TRUE, gamma = 1.5)

# Make predictions on the test data
gam_forecast <- predict(gam_model, newdata = Data1)

# Load the sample submission file
submit <- read_delim(file = "Data/sample_submission.csv", delim = ",")

# Assign the forecasted values to the submission file
submit$Net_demand <- gam_forecast

# Write the submission file to CSV
write.table(submit, file = "Data/submission_gam_advanced.csv", quote = FALSE, sep = ",", dec = '.', row.names = FALSE)
