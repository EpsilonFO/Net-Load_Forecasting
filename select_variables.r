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
sel_a = which(Data0$Year<=2021) # training index
sel_b = which(Data0$Year>2021) # eval index


#############################
### Corrélation Linéaire ###
#############################

# Calcul de la corrélation des variables avec Net_demand
cor_lin = cor(Data0[,sapply(Data0, is.numeric)], method = "pearson")["Net_demand", ]

# Définir un seuil et garder les variables linéairment corrélées
seuil = 0.3
variables_lincor = names(cor_lin[abs(cor_lin) > seuil])

# calcul des p-values pour savoir si c'est significative
p_values_cor_lin <- cor.mtest(Data0[,variables_lincor], conf.level = 0.95)$p#["Net_demand", ]

# Calculer et afficher la matrice de corrélation de la selection
cor_lin_mat = cor(Data0[,variables_lincor], method = "pearson")
corrplot(cor_lin_mat, method = "color",
         addCoef.col = "black",
         tl.col = "black", tl.srt = 45,
         number.cex = 0.75,
         sig.level = 0.05, # niveau de significativité à 5%
         p.mat = p_values_cor_lin,
         insig = "blank", # Ne pas afficher les corrélations non significatives
         addgrid.col = NA, cl.pos = "n")


#################################
### Corrélation non Linéaire ###
#################################

# Calcul de la corrélation des variables avec Net_demand
cor_nlin = cor(Data0[,sapply(Data0, is.numeric)], method = "spearman")["Net_demand", ]

# Garder les variables non-linéairment corrélées
seuil = 0.3
variables_nlincor = names(cor_nlin[abs(cor_nlin) > seuil])

# calcul des p-values pour savoir si c'est significative
p_values_cor_nlin <- cor.mtest(Data0[,variables_nlincor], conf.level = 0.95)$p#["Net_demand", ]

# Calculer et afficher la matrice de corrélation de la selection
cor_nlin_mat = cor(Data0[,variables_nlincor], method = "pearson")
corrplot(cor_nlin_mat, method = "color",
         addCoef.col = "black",
         tl.col = "black", tl.srt = 45,
         number.cex = 0.75,
         sig.level = 0.05, # niveau de significativité à 5%
         p.mat = p_values_cor_nlin,
         insig = "blank", # Ne pas afficher les corrélations non significatives
         addgrid.col = NA, cl.pos = "n")


###############################
### Importance plot avec RF ###
###############################

# choix de l'équation
equation <- Net_demand ~ . - Load # Load est très importante, se concentre sur le reste

# entrainement et prediction
rf <- ranger(equation, data=Data0[sel_a,], importance =  'permutation', num.trees = 1000, sample.fraction=0.1)
rf.forecast<-predict(rf, data= Data0[sel_b,])$prediction

# importance et graphique selection avec regle du coude
#windows(width = 10, height = 6)
#par(mar = c(5, 4, 4, 2) + 0.1)
imp <- rf$variable.importance
o <- order(imp, decreasing=T)
nom <- names(imp)
plot(c(1:length(imp)), imp[o], type='h', ylim = c(0, max(imp) + max(imp)/5), xlab='', ylab='Importance (permutation)')
K <- length(imp)
text(tail(c(1:length(imp)), K), tail(imp[o]+max(imp/8), K), labels= tail(nom[o], K), pos=3, srt=90, adj=1)
points(c(1:length(imp)), imp[o], pch=20)
