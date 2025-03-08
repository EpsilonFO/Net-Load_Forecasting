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
            "Holiday_zone_c", "BH_Holiday")
Data0[, discret] <- lapply(Data0[, discret], as.factor)
Data1[, discret] <- lapply(Data1[, discret], as.factor)

str(Data0)

# Split Data0 into train/eval dataset
sel_a = which(Data0$Year<=2021) # training index
sel_b = which(Data0$Year>2021) # eval index


#########################
### Unidimensionnelle ###
#########################

# plot la demand_net dans le temps
par(mfrow=c(1,1))
plot(Data0$Date, Data0$Net_demand, type='l', xlim=range(Data0$Date, Data1$Date), main="Net_demand dans le temps")

# plot variables sur temps : Load, Net_demand, Solar, Wind
col <- yarrr::piratepal("basel")
par(mfrow=c(4,1))
plot(Data0$Date, Data0$Load, type='l', col=col[1], main="Load dans le temps")
plot(Data0$Date, Data0$Net_demand, type='l', col=col[2], main="Net_demand dans le temps")
plot(Data0$Date, Data0$Solar_power, type='l', col=col[3], main="Solar_power dans le temps")
plot(Data0$Date, Data0$Wind_power, type='l', col=col[4], main="Wind_power dans le temps")

# Load = Net_demand + Solar + Wind
par(mfrow=c(2,1))
plot(Data0$Date, Data0$Load, type='l', col=col[1], main="Load dans le temps")
plot(Data0$Date, Data0$Net_demand+Data0$Solar_power+Data0$Wind_power, type='l', col=col[2], main="Net_demand + Solar_power + Wind_power")


# plot variables sur temps : net_demand, toy, Temp
par(mfrow=c(3,1))
plot(Data0$Date, Data0$Net_demand, type='l', col=col[1], main="Net_demand dans le temps")
plot(Data0$Date, Data0$Temp, type='l', col=col[2], main="Température dans le temps")
plot(Data0$Date, Data0$toy, type='p', col=col[3], main="toy (time of year) dans le temps")

# en scalant Net_load et Temp, on voit que net_demand ~ -Temp
par(mfrow=c(4,1))
plot(Data0$Date, Data0$Net_demand, type='l', col=col[1], main="Net_demand dans le temps")
plot(Data0$Date, Data0$Temp, type='l', col=col[2], main="Température dans le temps")
plot(Data0$Date, scale(Data0$Net_demand), type='l', col=col[1], main="Net_demand et Temp dans le temps")
lines(Data0$Date, scale(Data0$Temp), type='l', col=col[2])
plot(Data0$Date, scale(Data0$Net_demand), type='l', col=col[1], main="Net_demand et (-Temp) dans le temps")
lines(Data0$Date, -scale(Data0$Temp), type='l', col=col[2])


#########################
### Bidimensionnelle ###
#########################

# boxplot pour voir la corrélation entre Net_demand et les variable discrete
par(mfrow=c(4,1))
boxplot(Data0$Net_demand ~ Data0$BH, col=col[1], main="Net_demand si jour férié")
boxplot(Data0$Net_demand ~ Data0$Summer_break, col=col[2], main="Net_demand si summer break")
boxplot(Data0$Net_demand ~ Data0$WeekDays, col=col[3], main="Net_demand selon jour de la semaine")
boxplot(Data0$Net_demand ~ Data0$DLS, col=col[4], main="Net_demand si heure d'été")

# nuage pour voir la corrélation entre Net_demand et les variables continues
par(mfrow=c(4,1))
plot(Data0$Net_demand ~ Data0$Temp, col=col[1], main="Net_demand selon Temp")
plot(Data0$Net_demand ~ Data0$Wind, col=col[2], main="Net_demand selon Wind")
plot(Data0$Net_demand ~ Data0$Nebulosity, col=col[3], main="Net_demand selon Nebulosity")
plot(Data0$Net_demand ~ Data0$toy, col=col[4], main="Net_demand selon toy (time of year)")

par(mfrow=c(4,1)) # mettre month en discret ? suit pas mal le toy
plot(Data0$Net_demand ~ Data0$Month, col=col[1], main="Net_demand selon Month")
plot(Data0$Net_demand ~ Data0$Wind_weighted, col=col[2], main="Net_demand selon Wind_weighted")
plot(Data0$Net_demand ~ Data0$Nebulosity_weighted, col=col[3], main="Net_demand selon Nebulosity_weighted")
plot(Data0$Net_demand ~ Data0$Temp_s95, col=col[4], main="Net_demand selon Temp_s95")


###################
### Conclusion ###
###################