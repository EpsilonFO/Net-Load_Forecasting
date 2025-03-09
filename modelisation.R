rm(list=objects())
graphics.off()
# Load necessary libraries
library(mgcv)
library(gt)
library(tidyverse)
library(ranger)
library(randomForest)
library(xgboost)
source('R/score.R')


#########################
### Import et Prepro ###
#########################
# Load the data
train <- read_csv('Data/train.csv') # for training and evaluating
test <- read_csv('Data/test.csv') # to make prediction

# Preprocess the data
Data0 <- train
Data0$Time <- as.numeric(Data0$Date)
Data1 <- test
Data1$Time <- as.numeric(Data1$Date)

# Convert categorical variables to factors
Data0$WeekDays <- as.factor(Data0$WeekDays)
Data1$WeekDays <- as.factor(Data1$WeekDays)

# Split Data0 into train/eval dataset
sel_a <- which(Data0$Year<=2021) # training index
sel_b <- which(Data0$Year>2021) # eval index

############
### GAM ###
############

gam_equation <- Net_demand ~
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

# Train the GAM model with shrinkage on training set
g <- gam(gam_equation, data = Data0[sel_a,])
g.forecast <- predict(g, newdata=Data0)
terms0 <- predict(g, newdata=Data0, type='terms')
colnames(terms0) <- paste0("gterms_", c(1:ncol(terms0)))
# Evaluation 
eval_pred = predict(g, newdata= Data0[sel_b,])
gam1_rmse = rmse.old(Data0$Net_demand[sel_b]-eval_pred)
gam1_mape = mape(Data0$Net_demand[sel_b], eval_pred)
gam1_pinball = pinball_loss2(Data0$Net_demand[sel_b]-eval_pred, 0.8)


###########
### RF ###
###########

# Define the equation for rf
rf_equation <- Net_demand ~ Time + toy + Temp + Temp_s99 + Load.1 + Load.7 +
  WeekDays + BH + Temp_s95_max * Temp_s99_max +
  Summer_break + Christmas_break + Temp_s95_min * Temp_s99_min +
  Wind + Nebulosity_weighted + Wind_weighted * Temp

# Train rf on train set
rf_model = randomForest(rf_equation, data=Data0[sel_a,])

# Evaluation 
eval_pred = predict(rf_model, newdata= Data0[sel_b,])
rf1_rmse = rmse.old(Data0$Net_demand[sel_b]-eval_pred)
rf1_mape = mape(Data0$Net_demand[sel_b], eval_pred)
rf1_pinball = pinball_loss2(Data0$Net_demand[sel_b]-eval_pred, 0.8)

residuals <- c(Block_residuals, Data0[sel_b,]$Net_demand-g.forecast[sel_b])
Data0_rf <- data.frame(Data0, terms0)
Data0_rf$res <- residuals
Data0_rf$res.48 <- c(residuals[1], residuals[1:(length(residuals)-1)])
Data0_rf$res.336 <- c(residuals[1:7], residuals[1:(length(residuals)-7)])

cov <- "Time + toy + Temp + Load.1 + Load.7 + Temp_s99 + WeekDays + BH + Temp_s95_max + Temp_s99_max + Summer_break  + Christmas_break + Temp_s95_min +Temp_s99_min + DLS  + "
gterm <-paste0("gterms_", c(1:ncol(terms0)))
gterm <- paste0(gterm, collapse='+')
cov <- paste0(cov, gterm, collapse = '+')
formule_rf <- paste0("Net_demand", "~", cov)
formule_rf

qrf_gam<- ranger::ranger(formule_rf, data = Data0_rf[sel_a,], importance =  'permutation', seed=1, , quantreg=TRUE)
quant=0.8
qrf_gam.forecast <- predict(qrf_gam, data=Data0_rf[sel_b,], quantiles =quant, type = "quantiles")$predictions%>%as.numeric+g.forecast[sel_b]
qrf_gam_rmse = rmse.old(Data0$Net_demand[sel_b]-qrf_gam.forecast)
qrf_gam_mape = mape(Data0$Net_demand[sel_b], qrf_gam.forecast)
qrf_gam_pinball = pinball_loss2(Data0$Net_demand[sel_b]-qrf_gam.forecast, quant)

###################
### Loss Table ###
###################

# Créer un DataFrame avec les noms des modèles et leurs pertes
model_losses = data.frame(
  Modèle = c("GAM 1", "RF 1", "QRF_GAM"),
  RMSE = c(gam1_rmse, rf1_rmse, qrf_gam_rmse),
  MAPE = c(gam1_mape, rf1_mape, qrf_gam_mape),
  Pinball = c(gam1_pinball, rf1_pinball, qrf_gam_pinball)
)

# Afficher le tableau
gt(model_losses) %>%
  tab_header(
    title = "Pertes par modèle"
  ) %>%
  tab_style(
  style = cell_text(weight = "bold"),
  locations = cells_body(
    columns = vars(Pinball),
    rows = Pinball < 700))


###########################
### Submission for pred ###
###########################

# Train our model on all the train dataset
gam_model = gam(gam_equation, data = Data0, select = TRUE, gamma = 1.5)
rf_model = randomForest(rf_equation, data=Data0)

# Make predictions on the test data
gam_forecast = predict(gam_model, newdata = Data1)
rf_forecast = predict(rf_model, newdata=Data1)

# Load the sample submission file
submit = read_delim(file = "Data/sample_submission.csv", delim = ",")

# Assign the forecasted values to the submission file
submit$Net_demand = gam_forecast
submit$Net_demand = rf_forecast
# Write the submission file to CSV
write.table(submit, file = "Data/submission_rf.csv", quote = FALSE, sep = ",", dec = '.', row.names = FALSE)
