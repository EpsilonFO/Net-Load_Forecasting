rm(list=objects())
graphics.off()
# Load necessary libraries
library(mgcv)
library(corrplot)
library(gt)
library(tidyverse)
library(ranger)
library(randomForest)
library(xgboost)
library(yarrr)
source('R/score.R')
# Options graphique
options(vsc.dev.args = list(width=1200, height=800, pointsize=10, res=96))
par(mar = c(5, 5, 5, 5))  # marges : bas, gauche, haut, droite
col <- yarrr::piratepal("basel") # couleur des graphiques


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
discret = c("WeekDays", "BH_before", "BH", "BH_after", 
            "DLS","Summer_break", "Christmas_break", 
            "Holiday", "Holiday_zone_a", "Holiday_zone_b", 
            "Holiday_zone_c", "BH_Holiday", "Month")
Data0[, discret] <- lapply(Data0[, discret], as.factor)
Data1[, discret] <- lapply(Data1[, discret], as.factor)

# Split Data0 into train/eval dataset
sel_a <- which(Data0$Year<=2021) # training index
sel_b <- which(Data0$Year>2021) # eval index

# Drop covariables that are not in test dataset : Load, Solar_power, Wind_power
Data0 = Data0[-c(2, 6, 7)]

###########
### RL ###
###########

# Define the equation for rl1 = rf backward // avec l'equation rf2 encore mieux
rl1_equation = Net_demand ~ Load.1 + Temp + Temp_s95_max + Temp_s99_min + Temp_s99_max +
    Wind + Wind_weighted + Nebulosity_weighted + toy + WeekDays +
    BH_before + BH + Year + Month + Christmas_break + BH_Holiday +
    Wind_power.1 + Net_demand.7 + Time + Summer_break 

rf2_equation = Net_demand ~ Load.1 + Temp + Temp_s95_max + Temp_s99_min + Temp_s99_max +
              Wind + Wind_weighted + Nebulosity_weighted + toy + WeekDays +
              BH_before + BH + Year + Month + Christmas_break + BH_Holiday +
              Wind_power.1 + Net_demand.7 + Time + Net_demand.1 + Temp_s99 + Temp_s95 +
              Load.7 + Temp_s95_min + Summer_break

# Train rf on train set
rl1 = lm(rf2_equation, data=Data0[sel_a,])

# Evaluation 
rl1_pred = predict(rl1, newdata= Data0[sel_b,])
rl1_rmse = rmse.old(Data0$Net_demand[sel_b]-rl1_pred)
rl1_mape = mape(Data0$Net_demand[sel_b], rl1_pred)
rl1_pinball = pinball_loss2(Data0$Net_demand[sel_b]-rl1_pred, 0.8)

par(mfrow=c(2,1))
plot(Data0$Date[sel_b], rl1_pred, type='l', col=col[1], main="Prediction et valeur réelle sur l'évalutaion pour RF mixte") 
lines(Data0$Date[sel_b], Data0$Net_demand[sel_b], type='l', col=col[2])
plot(Data0$Date[sel_b], Data0$Net_demand[sel_b]-rl1_pred, type='l', col=col[3], main="Résidus") # residus


###########
### RF ###
###########

# Define the equation for rf1 = rf complet
rf1_equation = Net_demand ~ .

# Train rf on train set
rf1 = ranger(rf1_equation, data=Data0[sel_a,], num.trees = 1000, sample.fraction = 0.1, importance="permutation")

# Evaluation 
rf1_pred = predict(rf1, data= Data0[sel_b,])$predictions
rf1_rmse = rmse.old(Data0$Net_demand[sel_b]-rf1_pred)
rf1_mape = mape(Data0$Net_demand[sel_b], rf1_pred)
rf1_pinball = pinball_loss2(Data0$Net_demand[sel_b]-rf1_pred, 0.8)

par(mfrow=c(2,1))
plot(Data0$Date[sel_b], rf1_pred, type='l', col=col[1], main="Prediction et valeur réelle sur l'évalutaion pour RF complet") 
lines(Data0$Date[sel_b], Data0$Net_demand[sel_b], type='l', col=col[2])
plot(Data0$Date[sel_b], Data0$Net_demand[sel_b]-rf1_pred, type='l', col=col[3], main="Résidus") # residus



# Define the equation for rf2 = rf mixte
rf2_equation = Net_demand ~ Load.1 + Temp + Temp_s95_max + Temp_s99_min + Temp_s99_max +
              Wind + Wind_weighted + Nebulosity_weighted + toy + WeekDays +
              BH_before + BH + Year + Month + Christmas_break + BH_Holiday +
              Wind_power.1 + Net_demand.7 + Time + Net_demand.1 + Temp_s99 + Temp_s95 +
              Load.7 + Temp_s95_min + Summer_break

# Train rf on train set
rf2 = ranger(rf2_equation, data=Data0[sel_a,], num.trees = 1000, sample.fraction = 0.1, importance="permutation")

# Evaluation 
rf2_pred = predict(rf2, data= Data0[sel_b,])$predictions
rf2_rmse = rmse.old(Data0$Net_demand[sel_b]-rf2_pred)
rf2_mape = mape(Data0$Net_demand[sel_b], rf2_pred)
rf2_pinball = pinball_loss2(Data0$Net_demand[sel_b]-rf2_pred, 0.8)

par(mfrow=c(2,1))
plot(Data0$Date[sel_b], rf2_pred, type='l', col=col[1], main="Prediction et valeur réelle sur l'évalutaion pour RF mixte") 
lines(Data0$Date[sel_b], Data0$Net_demand[sel_b], type='l', col=col[2])
plot(Data0$Date[sel_b], Data0$Net_demand[sel_b]-rf2_pred, type='l', col=col[3], main="Résidus") # residus


############
### GAM ###
############

# Define the equation for the GAM model with advanced techniques
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
  ti(Wind_weighted, Temp, bs = 'ts') +
  s(Net_demand.1, bs = 'cr') +
  s(Net_demand.7, bs = 'cr') 

# Train the GAM model with shrinkage on training set
gam_model <- gam(gam_equation, data = Data0[sel_a,], select = TRUE, gamma = 1.5)

# Evaluation 
gam1_pred = predict(gam_model, newdata= Data0[sel_b,])
gam1_rmse = rmse.old(Data0$Net_demand[sel_b]-gam1_pred)
gam1_mape = mape(Data0$Net_demand[sel_b], gam1_pred)
gam1_pinball = pinball_loss2(Data0$Net_demand[sel_b]-gam1_pred, 0.8)

par(mfrow=c(2,1))
plot(Data0$Date[sel_b], gam1_pred, type='l', col=col[1], main="Prediction et valeur réelle sur l'évalutaion pour GAM") 
lines(Data0$Date[sel_b], Data0$Net_demand[sel_b], type='l', col=col[2])
plot(Data0$Date[sel_b], Data0$Net_demand[sel_b]-gam1_pred, type='l', col=col[3], main="Résidus") # residus


# Define the equation for the GAM 2 model 
gamm_equation <- Net_demand ~
  s(Load.1, k = 3, bs = 'cc') +
  s(Load.7, k = 3, bs = 'cc') +
  s(Net_demand.1, k=3, bs='cc') +
  s(Net_demand.7, k=3, bs='cc') +
  s(Temp, k=3, bs='cr') + 
  te(Temp_s95_max, Temp_s95_min) +
  te(Temp_s99_max, Temp_s99_min) +
  ti(Temp_s99, k = 1, bs = 'cr') +
  ti(Temp_s95, k = 1, bs = 'cr') +
  WeekDays + BH + BH_before + Month +
  Christmas_break + Summer_break + 
  BH_Holiday 


gam2_equation <- Net_demand ~
  s(Time, k = 5, bs = 'cr') +                 # Augmenter k pour plus de flexibilité
  s(toy, k = 20, bs = 'cc') +                 # Réduire k si nécessaire
  ti(Temp, k = 8, bs = 'cr') +                 # Ajuster k
  ti(Temp_s99, k = 8, bs = 'cr') +             # Ajuster k
  s(Net_demand.1, bs = 'cr') +
  s(Net_demand.7, bs = 'cr') +
  ti(Temp_s99, Temp, bs = c('cr', 'cr'), k = c(8, 8)) +  # Ajuster k
  as.factor(WeekDays) + BH +
  te(Temp_s95_max, Temp_s99_max, k = c(5, 5)) +  # Ajouter k pour plus de flexibilité
  Summer_break + Christmas_break +
  te(Temp_s95_min, Temp_s99_min, k = c(5, 5)) +  # Ajouter k pour plus de flexibilité
  s(Wind, bs = 'cr') +
  ti(Nebulosity_weighted, bs = 'cr') +         # Changer la base de lissage si nécessaire
  ti(Wind_weighted, Temp, bs = 'ts')


# Train the GAM model with shrinkage on training set
gam2_model <- gam(gam2_equation, data = Data0[sel_a,], select = TRUE, gamma = 1.5)

# Evaluation 
gam2_pred = predict(gam2_model, newdata= Data0)
gam2_rmse = rmse.old(Data0$Net_demand-gam2_pred)
gam2_mape = mape(Data0$Net_demand, gam2_pred)
gam2_pinball = pinball_loss2(Data0$Net_demand-gam2_pred, 0.8)
gam2_pinball

par(mfrow=c(2,1))
plot(Data0$Date[sel_b], gam2_pred, type='l', col=col[1], main="Prediction et valeur réelle sur l'évalutaion pour GAM") 
lines(Data0$Date[sel_b], Data0$Net_demand[sel_b], type='l', col=col[2])
plot(Data0$Date[sel_b], Data0$Net_demand[sel_b]-gam2_pred, type='l', col=col[3], main="Résidus") # residus


res <- Data0$Net_demand[sel_a] - gam2_pred[sel_a]
hist(res, breaks=100)
mean(res)
sd(res)
quant <- qnorm(0.8, mean= mean(res), sd= sd(res))
pinball_loss(y=Data0$Net_demand[sel_b], rf_gam.forecast[sel_b]+quant, quant=0.8, output.vect=FALSE)


###################
### Loss Table ###
###################

# Créer un DataFrame avec les noms des modèles et leurs pertes
model_losses = data.frame(
  Modèle = c("RL 1", "RF 1", "RF 2", "GAM 1", "GAM 2"),
  RMSE = c(rl1_rmse, rf1_rmse, rf2_rmse, gam1_rmse, gam2_rmse),
  MAPE = c( rl1_mape, rf1_mape, rf2_mape, gam1_mape, gam2_mape),
  Pinball = round(c(rl1_pinball, rf1_pinball, rf2_pinball, gam1_pinball, gam2_pinball), digits=0)
)

# Afficher le tableau
gt(model_losses) %>%
  tab_header(
    title = "Pertes par modèle"
  ) %>%
  tab_style(
  style = cell_text(weight = "bold"),
  locations = cells_body(
    columns = c(Pinball),
    rows = Pinball < 650)) %>%
  tab_style(
  style = cell_text(weight = "bold"),
  locations = cells_body(
    columns = c(MAPE),
    rows = MAPE < 2.5)) %>%
  tab_style(
  style = cell_text(weight = "bold"),
  locations = cells_body(
    columns = c(RMSE),
    rows = RMSE < 1400))


###########################
### Submission for pred ###
###########################

# Train our model on all the train dataset
gam_model = gam(gam2_equation, data = Data0, select = TRUE, gamma = 1.5)
rf_model = ranger(rf2_equation, data=Data0, num.trees = 1000, sample.fraction = 0.1, importance="permutation")
rl_model = lm(rf2_equation, data=Data0)

# Make predictions on the test data
gam_forecast = predict(gam_model, newdata = Data1)
gam_forecastpast = predict(gam_model, newdata = Data0)
rf_forecast = predict(rf_model, data=Data1)$predictions
rl_forecast = predict(rl_model, newdata=Data1)

# Load the sample submission file
submit = read_delim(file = "Data/sample_submission.csv", delim = ",")

# Assign the forecasted values to the submission file
res <- Data0$Net_demand - gam_forecastpast
hist(res, breaks=100)
mean(res)
sd(res)
quant <- qnorm(0.7, mean= mean(res), sd= sd(res))
quant
submit$Net_demand = gam_forecast-quant

# Write the submission file to CSV
write.table(submit, file = "Data/submission_gam.csv", quote = FALSE, sep = ",", dec = '.', row.names = FALSE)



gam_forecast
submit$Net_demand
