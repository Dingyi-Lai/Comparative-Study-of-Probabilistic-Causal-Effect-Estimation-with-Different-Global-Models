# Loading the libraries:
library(dplyr)
library(stringr)
library(ggplot2)
library(tidyverse)
library(hrbrthemes)
library(rstatix)
library(ggpubr)

# Reading the original and forecasting datasets
train_ds <- read.csv("./datasets/text_data/EMS/alchol_full_license_train.txt", header = FALSE)
test_ds <- read.csv("./datasets/text_data/EMS/alchol_full_license_test.txt", header = FALSE)
forecastCocob_ds <- read.csv("./results/nn_model_results/rnn/processed_ensemble_forecasts/ems12i15_without_stl_LSTMcell_cocob_without_stl_decomposition", header = FALSE)
forecastCocobEX_ds <- read.csv("./results/nn_model_results/rnn/processed_ensemble_forecasts/emsEX_without_stl_LSTMcell_cocob_without_stl_decomposition", header = FALSE)

# Adjusting layouts to plot
original_ds <- cbind(train_ds,test_ds)
full_forecastCocob_ds <- cbind(train_ds,forecastCocob_ds)
full_forecastCocobEX_ds <- cbind(train_ds,forecastCocobEX_ds)

original_ds <- as.data.frame(t(original_ds))
full_forecastCocob_ds <- as.data.frame(t(full_forecastCocob_ds))
full_forecastCocobEX_ds <- as.data.frame(t(full_forecastCocobEX_ds))

names(original_ds) <- original_ds[1,]
original_ds <- original_ds[-1,]
names(full_forecastCocob_ds) <- full_forecastCocob_ds[1,]
full_forecastCocob_ds <-full_forecastCocob_ds[-1,]
names(full_forecastCocobEX_ds) <- full_forecastCocobEX_ds[1,]
full_forecastCocobEX_ds <-full_forecastCocobEX_ds[-1,]

# inserting the date information
date <- data.frame(c("2015-01-01","2015-02-01","2015-03-01","2015-04-01","2015-05-01","2015-06-01",
                     "2015-07-01","2015-08-01","2015-09-01","2015-10-01","2015-11-01","2015-12-01",
                     "2016-01-01","2016-02-01","2016-03-01","2016-04-01","2016-05-01","2016-06-01",
                     "2016-07-01","2016-08-01","2016-09-01","2016-10-01","2016-11-01","2016-12-01",
                     "2017-01-01","2017-02-01","2017-03-01","2017-04-01","2017-05-01","2017-06-01",
                     "2017-07-01","2017-08-01","2017-09-01","2017-10-01","2017-11-01","2017-12-01",
                     "2018-01-01","2018-02-01","2018-03-01","2018-04-01","2018-05-01","2018-06-01",
                     "2018-07-01","2018-08-01","2018-09-01","2018-10-01","2018-11-01","2018-12-01",
                     "2019-01-01","2019-02-01","2019-03-01","2019-04-01","2019-05-01"))

names(date)[1] <- c("DATE")

original_ds[80] <- date
full_forecastCocob_ds[80] <- date
full_forecastCocobEX_ds[80] <- date

original_ds$DATE <- as.Date(original_ds$DATE)
full_forecastCocob_ds$DATE <- as.Date(full_forecastCocob_ds$DATE)
full_forecastCocobEX_ds$DATE <- as.Date(full_forecastCocobEX_ds$DATE)


# setting the treated and control groups
## treated group: LGAs showing Increase > 0% of ALI during the intervention period = 69 LGAs
Increase <- c(20110, 20830, 21670, 21750, 22490, 22620, 22830, 23190, 23810, 23940, 
              24250, 25150, 25990, 26260, 26610, 26670, 26810, 27170, 27450, 27630, 
              20260, 20570, 20660, 20740, 20910, 21110, 21180, 21450, 21610, 21830, 
              21890, 22110, 22170, 22250, 22310, 22410, 22750, 22910, 22980, 23110, 
              23270, 23350, 23430, 23670, 24130, 24210, 24330, 24410, 24600, 24650, 
              24780, 24900, 24970, 25060, 25250, 25340, 25490, 25620, 25710, 25900, 
              26080, 26170, 26350, 26490, 26730, 26980, 27070, 27260, 27350,
              "DATE") # 69 LGAs (these numbers are the LGAs codes)

## control group: LGAs showing constant or decrease in ALI <= 0% during the intervention period = 10 LGAs
NotIncrease <- c(21010, 21270, 21370, 22670, 24850, 25430, 25810, 26430, 26700, 26890,
                 "DATE") # 10 LGAs (these numbers are the LGAs codes)


# spliting the dataframes into control and treated groups
full_forecastCocob_ds_I <- full_forecastCocob_ds[,as.character(Increase)] # treated group
full_forecastCocob_ds_NI <- full_forecastCocob_ds[,as.character(NotIncrease)] # control group

full_forecastCocobEX_ds_I <- full_forecastCocobEX_ds[,as.character(Increase)] # treated group
full_forecastCocobEX_ds_NI <- full_forecastCocobEX_ds[,as.character(NotIncrease)] # control group

original_ds_I <- original_ds[,as.character(Increase)] # treated group
original_ds_NI <- original_ds[,as.character(NotIncrease)] # control group

# grouping the treated and control groups of LGAs summarizing by mean
original_ds_I$TOTAL <- rowMeans(original_ds_I[,1:69])
full_forecastCocob_ds_I$TOTAL <- rowMeans(full_forecastCocob_ds_I[,1:69])
full_forecastCocobEX_ds_I$TOTAL <- rowMeans(full_forecastCocobEX_ds_I[,1:69])

original_ds_NI$TOTAL <- rowMeans(original_ds_NI[,1:10])
full_forecastCocob_ds_NI$TOTAL <- rowMeans(full_forecastCocob_ds_NI[,1:10])
full_forecastCocobEX_ds_NI$TOTAL <- rowMeans(full_forecastCocobEX_ds_NI[,1:10])

# plotting
## DeepCPNet modelling variant - without consider ALI as exogenous variable
graph1 <- ggplot(original_ds_I, aes(x = DATE)) +
  geom_line(aes(y = unlist(full_forecastCocob_ds_I["TOTAL"])), color="blue", linetype = "dotted", size=1) +
  geom_line(aes(y = unlist(original_ds_I["TOTAL"])), color="blue", size=1) +
  geom_line(aes(y = unlist(full_forecastCocob_ds_NI["TOTAL"])), color="orange", linetype = "dotted", size=1) +
  geom_line(aes(y = unlist(original_ds_NI["TOTAL"])), color="orange", size=1) +
  scale_color_discrete(labels = c("Y2", "Y1"))+
  xlab("") +
  ylab("Monthly average demand for EMS-AI") +
  scale_x_date(date_labels = "%Y-%m", date_breaks  ="2 months") +
  labs(title = "Average demand of ambulance attendances related to alcohol intoxication - NASS dataset",
       subtitle = "DeepCPNet modelling variant") +
  theme_ipsum(plot_title_size = 13) +
  theme(axis.text.x=element_text(angle=60, hjust=1, size = 8)) 

## DeepCPNet-ALI modelling variant - considering ALI as exogenous variable
graph2 <- ggplot(original_ds_I, aes(x = DATE)) +
  geom_line(aes(y = unlist(original_ds_I["TOTAL"])), color="blue", size=1) +
  geom_line(aes(y = unlist(full_forecastCocobEX_ds_I["TOTAL"])), color="blue", linetype = "dotted", size=1) +
  geom_line(aes(y = unlist(original_ds_NI["TOTAL"])), color="orange", size=1) +
  geom_line(aes(y = unlist(full_forecastCocobEX_ds_NI["TOTAL"])), color="orange", linetype = "dotted", size=1) +
  xlab("") +
  ylab("Monthly average demand for EMS-AI") +
  labs(title = "(A) Average demand of ambulance attendances related to alcohol intoxication - NASS dataset",
       subtitle = "DeepCPNet-ALI modelling variant") +
  scale_x_date(date_labels = "%Y-%m", date_breaks  ="2 months") +
  theme_ipsum(plot_title_size = 12) + 
  theme(axis.text.x=element_text(angle=60, hjust=1, size = 8), legend.position = "top") 

graph1
graph2

###########################
# MEAN DIFFERENCE TESTING

## 1. Preparing the data

# using model without ALI as exogenous variable
## increasing group
diff_I <- subset(full_forecastCocob_ds_I, DATE >= '2018-06-01', select=c(DATE, TOTAL))
diff_I$Y <- subset(original_ds_I, DATE >= '2018-06-01', select=c(TOTAL))
diff_I$ab_diff <- abs(diff_I$TOTAL - diff_I$Y)
diff_I$ab_sum_av <- (abs(diff_I$TOTAL) + abs(diff_I$Y))/2
diff_I$ab_diff_propY <- (diff_I$ab_diff) / (diff_I$Y)
diff_I$ab_diff_propT <- (diff_I$ab_diff) / (diff_I$TOTAL)
diff_I$ab_diff_propYF <- (diff_I$ab_diff) / (diff_I$ab_sum_av)

## not increasing group
diff_NI <- subset(full_forecastCocob_ds_NI, DATE >= '2018-06-01', select=c(DATE, TOTAL))
diff_NI$Y <- subset(original_ds_NI, DATE >= '2018-06-01', select=c(TOTAL))
diff_NI$ab_diff <- abs(diff_NI$TOTAL - diff_NI$Y)
diff_NI$ab_sum_av <- (abs(diff_NI$TOTAL) + abs(diff_NI$Y))/2
diff_NI$ab_diff_propY <- (diff_NI$ab_diff) / (diff_NI$Y)
diff_NI$ab_diff_propT <- (diff_NI$ab_diff) / (diff_NI$TOTAL)
diff_NI$ab_diff_propYF <- (diff_NI$ab_diff) / (diff_NI$ab_sum_av)


# using model with ALI as exogenous variable
## increasing group
diffEX_I <- subset(full_forecastCocobEX_ds_I, DATE >= '2018-06-01', select=c(DATE, TOTAL))
diffEX_I$Y <- subset(original_ds_I, DATE >= '2018-06-01', select=c(TOTAL))
diffEX_I$ab_diff <- abs(diffEX_I$TOTAL - diffEX_I$Y)
diffEX_I$ab_sum_av <- (abs(diffEX_I$TOTAL) + abs(diffEX_I$Y))/2
diffEX_I$ab_diff_propY <- (diffEX_I$ab_diff) / (diffEX_I$Y)
diffEX_I$ab_diff_propT <- (diffEX_I$ab_diff) / (diffEX_I$TOTAL)
diffEX_I$ab_diff_propYF <- (diffEX_I$ab_diff) / (diffEX_I$ab_sum_av)

## not increasing group
diffEX_NI <- subset(full_forecastCocobEX_ds_NI, DATE >= '2018-06-01', select=c(DATE, TOTAL))
diffEX_NI$Y <- subset(original_ds_NI, DATE >= '2018-06-01', select=c(TOTAL))
diffEX_NI$ab_diff <- abs(diffEX_NI$TOTAL - diffEX_NI$Y)
diffEX_NI$ab_sum_av <- (abs(diffEX_NI$TOTAL) + abs(diffEX_NI$Y))/2
diffEX_NI$ab_diff_propY <- (diffEX_NI$ab_diff) / (diffEX_NI$Y)
diffEX_NI$ab_diff_propT <- (diffEX_NI$ab_diff) / (diffEX_NI$TOTAL)
diffEX_NI$ab_diff_propYF <- (diffEX_NI$ab_diff) / (diffEX_NI$ab_sum_av)


## 2. MEAN SIGNIFICANCE TESTS FOR relative differences
diff_I_vec <- as.vector(unlist(diff_I$ab_diff_propT))*100
diff_NI_vec <- as.vector(unlist(diff_NI$ab_diff_propT))*100

diffEX_I_vec <- as.vector(unlist(diffEX_I$ab_diff_propT))*100
diffEX_NI_vec <- as.vector(unlist(diffEX_NI$ab_diff_propT))*100

cat(mean(diff_I_vec), median(diff_I_vec))
cat(mean(diff_NI_vec), median(diff_NI_vec))
cat(mean(diffEX_I_vec), median(diffEX_I_vec))
cat(mean(diffEX_NI_vec), median(diffEX_NI_vec))

### 3. TESTS FOR DIFFERENCE OF MEANS - Wilcoxon signed-rank exact test 
#### if p-value < 0.05, the difference between means is significant. In other words, the means are statistically different.
#### using the default significance level of alpha = 0.05 
wilcox.test(diff_I_vec, diff_NI_vec, paired = TRUE) # DeepCPNet modelling variant --> result: p-value = 0.01221 < 0.05
wilcox.test(diffEX_I_vec, diffEX_NI_vec, paired = TRUE) # DeepCPNet-ALI modelling variant --> result: p-value = 0.021 < 0.05

