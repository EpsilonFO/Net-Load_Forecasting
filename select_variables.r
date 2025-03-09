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
sel_a = which(Data0$Year<=2021) # training index
sel_b = which(Data0$Year>2021) # eval index

# Drop covariables that are not in test dataset : Load, Solar_power, Wind_power
Data0 = Data0[-c(2, 6, 7)]


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

##############################
### Analyse de regréssion ###
##############################

# ajuste une reg lin complete
rl.complet <- lm(Net_demand ~. -Date -BH_after, data = Data0)
summary(rl.complet)
# test de Student : coef egale à 0 ?
# variable très significative non nulle (***) : 
# - Load.1
# - Temp 
# - Wind_weighted
# - WeekDays
# - BH_before
# - BH 
# - Holiday
# Les zones de vacances sont pas significatives 
# ça donne quoi en enlevant ? 

# ajuste un modele sans les zone de vacances
eq_sans_zone = Net_demand ~. -Date -BH_after -Holiday_zone_a -Holiday_zone_b - Holiday_zone_c
rl.sans_zone = lm(eq_sans_zone, data=Data0)

# test de modèle emboité, le modele complet n'est pas signif meilleur
anova(rl.sans_zone, rl.complet)

# Utiliser step() pour la (backward) sélection de variables
# basée sur le BIC pour pénalisé la dimension du model
n = dim(Data0)[1]
rl.backward <- step(rl.complet, direction = "backward", trace=0, k=log(n))
summary(rl.backward)
eq_backward = Net_demand ~ Load.1 + Temp + Temp_s95_max + Temp_s99_min + Temp_s99_max +
    Wind + Wind_weighted + Nebulosity_weighted + toy + WeekDays +
    BH_before + BH + Year + Month + Christmas_break + BH_Holiday +
    Wind_power.1 + Net_demand.7 + Time 

# test de modele emboité, le modele sans zone est significativement meilleur
anova(rl.backward, rl.sans_zone)

# mais le test F ne pénalise pas la dimension,
# avec les critère de BIC et d'AIC on pénalise la dimension
# voilà ce qu'on obtient : 
BIC = c(BIC(rl.complet), BIC(rl.sans_zone), BIC(rl.backward))
AIC = c(AIC(rl.complet), AIC(rl.sans_zone), AIC(rl.backward))



# ajustons ces rl au train et evaluons les avec une pinball loss
rl_eval.complet <- lm(Net_demand ~ . -Date -BH_after, data = Data0[sel_a,])
rl_pred.complet = predict.lm(rl_eval.complet, newdata = Data0[sel_b,])
res_complet = rl_pred.complet-Data0$Net_demand[sel_b]

rl_eval.sans_zone <- lm(eq_sans_zone, data = Data0[sel_a,])
rl_pred.sans_zone = predict.lm(rl_eval.sans_zone, newdata = Data0[sel_b,])
res_sans_zone = rl_pred.sans_zone-Data0$Net_demand[sel_b]

rl_eval.backward <- lm(eq_backward, data = Data0[sel_a,])
rl_pred.backward = predict.lm(rl_eval.backward, newdata = Data0[sel_b,])
res_backward = rl_pred.backward-Data0$Net_demand[sel_b]


rl_select = data.frame(
  Modèle = c("Complet", "Sans zone", "Backward"),
  BIC = round(c(BIC(rl.complet), BIC(rl.sans_zone), BIC(rl.backward)), digits = 0),
  AIC = round(c(AIC(rl.complet), AIC(rl.sans_zone), AIC(rl.backward)), digits=0),
  Pinball = round(c(pinball_loss2(res_complet, 0.8), pinball_loss2(res_sans_zone, 0.8), pinball_loss2(res_backward, 0.8)), digits=0)
)

# Afficher le tableau
gt(rl_select) %>%
  tab_header(
    title = "Selection de variables sur RL"
  ) %>%
  tab_style(
  style = cell_text(weight = "bold"),
  locations = cells_body(
    columns = vars(Pinball),
    rows = Pinball <= 601))

par(mfrow=c(2,1))
plot(Data0$Date[sel_b], rl_pred.backward, type='l', col=col[1], main="Prediction et valeur réelle sur l'évalutaion pour RL backward") 
lines(Data0$Date[sel_b], Data0$Net_demand[sel_b], type='l', col=col[2])
plot(Data0$Date[sel_b], res_backward, type='l', col=col[3], main="Résidus") # residus



####################################
### Importance plot avec les RF ###
####################################



###############################
### Importance plot avec RF ###
###############################

# choix de l'équation, toutes les variables
equation <- Net_demand ~ . 

# entrainement et prediction
rf.complet <- ranger(equation, data=Data0, importance =  'permutation', num.trees = 1000, sample.fraction=0.1)

# importance et graphique selection avec regle du coude
imp <- rf.complet$variable.importance
o <- order(imp, decreasing=T)
nom <- names(imp)
par(mfrow=c(1, 1))
plot(c(1:length(imp)), imp[o], type='h', ylim = c(0, max(imp) + max(imp)/5), xlab='', ylab='Importance (permutation)')
K <- length(imp)
text(tail(c(1:length(imp)), K), tail(imp[o]+max(imp/8), K), labels= tail(nom[o], K), pos=3, srt=90, adj=1)
points(c(1:length(imp)), imp[o], pch=20)

# règle du coude :
# - Net_demand.1
# - Load.1
# - Temp_s99_max
# - Temp_s95_max
# - Temp_s99
# - Temp_s95
# - Temp
# - Load.7
# - Temp_s99_min
# - Net_demand.7
# - Temp_s95_min
# - BH_Holiday
# - WeekDays

# sur les covariable de la Backward selection
# entrainement et prediction
rf.backward <- ranger(eq_backward, data=Data0, importance =  'permutation', num.trees = 1000, sample.fraction=0.1)

# importance et graphique selection avec regle du coude
imp <- rf.backward$variable.importance
o <- order(imp, decreasing=T)
nom <- names(imp)
par(mfrow=c(1, 1))
plot(c(1:length(imp)), imp[o], type='h', ylim = c(0, max(imp) + max(imp)/5), xlab='', ylab='Importance (permutation)')
K <- length(imp)
text(tail(c(1:length(imp)), K), tail(imp[o]+max(imp/8), K), labels= tail(nom[o], K), pos=3, srt=90, adj=1)
points(c(1:length(imp)), imp[o], pch=20)

# règle du coude :
# - Load.1
# - Temp_s95_max
# - Temp_s99_max
# - Temp_s99_min
# - Temp
# - Net_demand.7
# - BH_Holiday
# - WeekDays
# retrouve les importantes sur RF avec toutes les variables

# on va garder l'equation backward en ajoutant
# les covariables importante pour les RF
eq_mixte = Net_demand ~ Load.1 + Temp + Temp_s95_max + Temp_s99_min + Temp_s99_max +
    Wind + Wind_weighted + Nebulosity_weighted + toy + WeekDays +
    BH_before + BH + Year + Month + Christmas_break + BH_Holiday +
    Wind_power.1 + Net_demand.7 + Time + Net_demand.1 + Temp_s99 + Temp_s95 +
    Load.7 + Temp_s95_min 


rf.mixte <- ranger(eq_mixte, data=Data0, importance =  'permutation', num.trees = 1000, sample.fraction=0.1)

# importance et graphique selection avec regle du coude
imp <- rf.mixte$variable.importance
o <- order(imp, decreasing=T)
nom <- names(imp)
par(mfrow=c(1, 1))
plot(c(1:length(imp)), imp[o], type='h', ylim = c(0, max(imp) + max(imp)/5), xlab='', ylab='Importance (permutation)')
K <- length(imp)
text(tail(c(1:length(imp)), K), tail(imp[o]+max(imp/8), K), labels= tail(nom[o], K), pos=3, srt=90, adj=1)
points(c(1:length(imp)), imp[o], pch=20)
# les variables ajoutées sont bien conservées dans l'importance plot


### regardons les perf de ces 3 rf

# Eval complet 
rf_eval.complet = randomForest(equation, data=Data0[sel_a,])
rf1_pred = predict(rf_eval.complet, newdata= Data0[sel_b,])
rf1_rmse = rmse.old(Data0$Net_demand[sel_b]-rf1_pred)
rf1_mape = mape(Data0$Net_demand[sel_b], rf1_pred)
rf1_pinball = pinball_loss2(Data0$Net_demand[sel_b]-rf1_pred, 0.8)

# Eval backward
rf_eval.backward = randomForest(eq_backward, data=Data0[sel_a,])
rf2_pred = predict(rf_eval.backward, newdata= Data0[sel_b,])
rf2_rmse = rmse.old(Data0$Net_demand[sel_b]-rf2_pred)
rf2_mape = mape(Data0$Net_demand[sel_b], rf2_pred)
rf2_pinball = pinball_loss2(Data0$Net_demand[sel_b]-rf2_pred, 0.8)

# Eval mixte
rf_eval.mixte = randomForest(eq_mixte, data=Data0[sel_a,])
rf3_pred = predict(rf_eval.mixte, newdata= Data0[sel_b,])
rf3_rmse = rmse.old(Data0$Net_demand[sel_b]-rf3_pred)
rf3_mape = mape(Data0$Net_demand[sel_b], rf3_pred)
rf3_pinball = pinball_loss2(Data0$Net_demand[sel_b]-rf3_pred, 0.8)

rf1_pinball, rf2_pinball, rf3_pinball

rf_select = data.frame(
  Modèle = c("Complet", "Backward", "Mixte"),
  RMSE = round(c(rf1_rmse, rf2_rmse, rf3_rmse), digits = 0),
  MAPE = round(c(rf1_mape, rf2_mape, rf3_mape), digits=2),
  Pinball = round(c(rf1_pinball, rf2_pinball, rf3_pinball), digits=0)
)

# Afficher le tableau
gt(rf_select) %>%
  tab_header(
    title = "Selection de variables sur RF"
  ) %>%
  tab_style(
  style = cell_text(weight = "bold"),
  locations = cells_body(
    columns = c(RMSE, MAPE),
    rows = RMSE <= 2000)) %>%
  tab_style(
  style = cell_text(weight = "bold"),
  locations = cells_body(
    columns = c(Pinball),
    rows = Pinball <= 480))

par(mfrow=c(2,1))
plot(Data0$Date[sel_b], rf3_pred, type='l', col=col[1], main="Prediction et valeur réelle sur l'évalutaion pour RF mixte") 
lines(Data0$Date[sel_b], Data0$Net_demand[sel_b], type='l', col=col[2])
plot(Data0$Date[sel_b], Data0$Net_demand[sel_b]-rf3_pred, type='l', col=col[3], main="Résidus") # residus

