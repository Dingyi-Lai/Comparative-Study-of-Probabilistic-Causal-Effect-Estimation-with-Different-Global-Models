# Loading the libraries:
library(dplyr)
library(stringr)
library(ggplot2)
library(tidyverse)
library(hrbrthemes)
library(rstatix)
library(ggpubr)

# Reading the original and forecasting datasets
observed_data_total <- read.csv(file = "./datasets/text_data/EMS-MC/calls911_month_full.txt", sep = ',', header = FALSE)
train_ds <- read.csv(file = "./datasets/text_data/EMS-MC/callsMT2_dataset.txt", header = FALSE)
test_ds <- read.csv(file = "./datasets/text_data/EMS-MC/callsMT2_results.txt", sep = ';', header = FALSE)
forecast_total <- read.csv(file = "./results/nn_model_results/rnn/processed_ensemble_forecasts/callsMT215_without_stl_LSTMcell_cocob_without_stl_decomposition", sep = ',', header = FALSE)

# Adjusting layouts to plot
forecastfull_total <- cbind(observed_data_total[,1], train_ds, forecast_total)

observed_data_total_ds <- as.data.frame(t(observed_data_total))
forecastfull_total_ds <- as.data.frame(t(forecastfull_total))

names(observed_data_total_ds) <- observed_data_total_ds[1,]
observed_data_total_ds <- observed_data_total_ds[-1,]
names(forecastfull_total_ds) <- forecastfull_total_ds[1,]
forecastfull_total_ds <-forecastfull_total_ds[-1,]

observed_data_total_ds[] <- lapply(observed_data_total_ds, function(x) as.numeric(as.character(x)))
forecastfull_total_ds[] <- lapply(forecastfull_total_ds, function(x) as.numeric(as.character(x)))

# inserting the date information
date <-  data.frame(c("2015-12-01",
                      "2016-01-01","2016-02-01","2016-03-01","2016-04-01","2016-05-01","2016-06-01",
                      "2016-07-01","2016-08-01","2016-09-01","2016-10-01","2016-11-01","2016-12-01",
                      "2017-01-01","2017-02-01","2017-03-01","2017-04-01","2017-05-01","2017-06-01",
                      "2017-07-01","2017-08-01","2017-09-01","2017-10-01","2017-11-01","2017-12-01",
                      "2018-01-01","2018-02-01","2018-03-01","2018-04-01","2018-05-01","2018-06-01",
                      "2018-07-01","2018-08-01","2018-09-01","2018-10-01","2018-11-01","2018-12-01",
                      "2019-01-01","2019-02-01","2019-03-01","2019-04-01","2019-05-01","2019-06-01",
                      "2019-07-01","2019-08-01","2019-09-01","2019-10-01","2019-11-01","2019-12-01",
                      "2020-01-01","2020-02-01","2020-03-01","2020-04-01","2020-05-01","2020-06-01",
                      "2020-07-01"))

names(date)[1] <- c("DATE")

observed_data_total_ds[63] <- date
forecastfull_total_ds[63] <- date

observed_data_total_ds$DATE <- as.Date(observed_data_total_ds$DATE)
forecastfull_total_ds$DATE <- as.Date(forecastfull_total_ds$DATE)

observed_data_total_ds <- observed_data_total_ds[-1,]
forecastfull_total_ds <- forecastfull_total_ds[-1,]

# setting the treated and control groups
## control group: towns that adopted none or lighter lockdown restrictions = 12 townships
control_list3 <- c("BRIDGEPORT", "BRYN ATHYN", "DOUGLASS", "HATBORO", "HATFIELD BORO", 
                   "LOWER FREDERICK", "NEW HANOVER", "NORRISTOWN", "NORTH WALES", "SALFORD", 
                   "SPRINGFIELD", "TRAPPE",
                   "DATE")

## treated group: towns that adopted lockdown restrictions = 50 townships
treated_list3 <- c("ABINGTON",  "AMBLER",  "CHELTENHAM",  "COLLEGEVILLE",  "CONSHOHOCKEN", 
                   "EAST GREENVILLE",  "EAST NORRITON",  "FRANCONIA" , "GREEN LANE", "HATFIELD TOWNSHIP", 
                   "HORSHAM" , "JENKINTOWN",  "LANSDALE",  "LIMERICK",  "LOWER GWYNEDD", 
                   "LOWER MERION",  "LOWER MORELAND",  "LOWER POTTSGROVE",  "LOWER PROVIDENCE",  "LOWER SALFORD", 
                   "MARLBOROUGH",  "MONTGOMERY",  "NARBERTH",  "PENNSBURG",  "PERKIOMEN", 
                   "PLYMOUTH",  "POTTSTOWN",  "RED HILL",  "ROCKLEDGE",  "ROYERSFORD", 
                   "SCHWENKSVILLE",  "SKIPPACK",  "SOUDERTON",  "TELFORD",  "TOWAMENCIN", 
                   "UPPER DUBLIN",  "UPPER FREDERICK",  "UPPER GWYNEDD",  "UPPER HANOVER",  "UPPER MERION", 
                   "UPPER MORELAND",  "UPPER POTTSGROVE",  "UPPER PROVIDENCE",  "UPPER SALFORD",  "WEST CONSHOHOCKEN", 
                   "WEST NORRITON",  "WEST POTTSGROVE",  "WHITEMARSH",  "WHITPAIN",  "WORCESTER",
                   "DATE")

# spliting the dataframes into control and treated groups
forecastfull_total_ds_TR <- forecastfull_total_ds[,as.character(treated_list3)]
forecastfull_total_ds_CR <- forecastfull_total_ds[,as.character(control_list3)]

observed_data_total_ds_TR <- observed_data_total_ds[,as.character(treated_list3)]
observed_data_total_ds_CR <- observed_data_total_ds[,as.character(control_list3)]

# grouping the treated and control groups of towns summarizing by mean
observed_data_total_ds_TR$TOTAL <- rowMeans(observed_data_total_ds_TR[,1:50])
forecastfull_total_ds_TR$TOTAL <- rowMeans(forecastfull_total_ds_TR[,1:50])

observed_data_total_ds_CR$TOTAL <- rowMeans(observed_data_total_ds_CR[,1:12])
forecastfull_total_ds_CR$TOTAL <- rowMeans(forecastfull_total_ds_CR[,1:12])

# plotting
graph3 <- ggplot(observed_data_total_ds, aes(x = DATE)) +
  geom_line(aes(y = unlist(forecastfull_total_ds_TR["TOTAL"])), color="blue", size=1, linetype = "dotted") +
  geom_line(aes(y = unlist(observed_data_total_ds_TR["TOTAL"])), color="blue", size=1) +
  geom_line(aes(y = unlist(forecastfull_total_ds_CR["TOTAL"])), color="orange", size=1, linetype = "dotted") +
  geom_line(aes(y = unlist(observed_data_total_ds_CR["TOTAL"])), color="orange", size=1) +
  xlab("") +
  ylab("Monthly average demand for 911 emergency calls") +
  labs(title = "(B) Average demand of 911 emergency calls - Montgomery dataset") +
  scale_x_date(date_labels = "%Y-%m", date_breaks  ="2 months") +
  theme_ipsum() +
  theme(axis.text.x=element_text(angle=60, hjust=1, size = 8)) 

graph3

###########################
# MEAN DIFFERENCE TESTING

## 1. Preparing the data

## treated group
diff_I <- subset(forecastfull_total_ds_TR, DATE >= '2020-01-01', select=c(DATE, TOTAL))
diff_I$Y <- subset(observed_data_total_ds_TR, DATE >= '2020-01-01', select=c(TOTAL))
diff_I$ab_diff <- abs(diff_I$TOTAL - diff_I$Y)
diff_I$ab_sum_av <- (abs(diff_I$TOTAL) + abs(diff_I$Y))/2
diff_I$ab_diff_propY <- (diff_I$ab_diff) / (diff_I$Y)
diff_I$ab_diff_propT <- (diff_I$ab_diff) / (diff_I$TOTAL)
diff_I$ab_diff_propYF <- (diff_I$ab_diff) / (diff_I$ab_sum_av)

## control group
diff_NI <- subset(forecastfull_total_ds_CR, DATE >= '2020-01-01', select=c(DATE, TOTAL))
diff_NI$Y <- subset(observed_data_total_ds_CR, DATE >= '2020-01-01', select=c(TOTAL))
diff_NI$ab_diff <- abs(diff_NI$TOTAL - diff_NI$Y)
diff_NI$ab_sum_av <- (abs(diff_NI$TOTAL) + abs(diff_NI$Y))/2
diff_NI$ab_diff_propY <- (diff_NI$ab_diff) / (diff_NI$Y)
diff_NI$ab_diff_propT <- (diff_NI$ab_diff) / (diff_NI$TOTAL)
diff_NI$ab_diff_propYF <- (diff_NI$ab_diff) / (diff_NI$ab_sum_av)

## 2. MEAN SIGNIFICANCE TESTS FOR relative differences
diff_I_vec <- as.vector(unlist(diff_I$ab_diff_propT))*100
diff_NI_vec <- as.vector(unlist(diff_NI$ab_diff_propT))*100

cat(mean(diff_I_vec), median(diff_I_vec))
cat(mean(diff_NI_vec), median(diff_NI_vec))

# Shapiro-Wilk’s test for normality
shapiro.test(diff_I_vec) # if p-value > 0.05, so data are not significantly different from normal distribution. In other words, we can assume the normality.
shapiro.test(diff_NI_vec) # if p-value > 0.05, so data are not significantly different from normal distribution. In other words, we can assume the normality.

# Results are p-value = 0.8844 and p-value = 0.9045, so in this dataset the distribution of the relative differences follows 
# normal distribution, so we need to apply the t-test instead of Wilcoxon signed-rank exact test.

t.test(x=diff_I_vec, y=diff_NI_vec, paired = TRUE) # if p-value < 0.05, the difference between means is significant. In other words, the means are statistically different.
wilcox.test(diff_I_vec, diff_NI_vec, paired = TRUE) # if p-value < 0.05, the difference between means is significant. In other words, the means are statistically different.

# result of t-test --> p-value = 0.004604 < 0.05, so the distribution of errors are statistically different
# result of wilcox test --> p-value = 0.01563 < 0.05 so the distribution of errors are statistically different






