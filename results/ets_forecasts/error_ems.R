# final evaluation ETS

ems_forecasts <- read.csv(file = "./results/ets_forecasts/ems.txt", sep = ',', header = FALSE)
actual_results_df <- read.csv(file = "./datasets/text_data/EMS/ems_results.txt", sep = ';', header = FALSE)
actual_results_df <- actual_results_df[,-1]

#### SMAPE
time_series_wise_SMAPE <- 2 * abs(ems_forecasts - actual_results_df) / (abs(ems_forecasts) + abs(actual_results_df))
SMAPEPerSeries <- rowMeans(time_series_wise_SMAPE, na.rm = TRUE)

mean_SMAPE = mean(SMAPEPerSeries)
median_SMAPE = median(SMAPEPerSeries)
std_SMAPE = sd(SMAPEPerSeries)

mean_SMAPE
median_SMAPE
std_SMAPE

train_ds <- read.csv("./datasets/text_data/EMS/alchol_full_license_train.txt", sep = ',', header = FALSE)
time_series_wise_SMAPE_B <- time_series_wise_SMAPE
time_series_wise_SMAPE_B <- cbind(train_ds[,1],time_series_wise_SMAPE_B)
time_series_wise_SMAPE_C <- t(time_series_wise_SMAPE_B)

time_series_wise_SMAPE_B$TOTAL <- rowMeans(time_series_wise_SMAPE_B[,2:13])

Increase <- c(20110, 20830, 21670, 21750, 22490, 22620, 22830, 23190, 23810, 23940, 
              24250, 25150, 25990, 26260, 26610, 26670, 26810, 27170, 27450, 27630, 
              20260, 20570, 20660, 20740, 20910, 21110, 21180, 21450, 21610, 21830, 
              21890, 22110, 22170, 22250, 22310, 22410, 22750, 22910, 22980, 23110, 
              23270, 23350, 23430, 23670, 24130, 24210, 24330, 24410, 24600, 24650, 
              24780, 24900, 24970, 25060, 25250, 25340, 25490, 25620, 25710, 25900, 
              26080, 26170, 26350, 26490, 26730, 26980, 27070, 27260, 27350,
              "DATE", "TOTAL") # 69 LGAs

NotIncrease <- c(21010, 21270, 21370, 22670, 24850, 25430, 25810, 26430, 26700, 26890,
                 "DATE", "TOTAL") # 10 LGAs

names(time_series_wise_SMAPE_B)[1] <- "TWP"
colnames(time_series_wise_SMAPE_B)

time_series_wise_SMAPE_IN  <- subset(time_series_wise_SMAPE_B, TWP %in% Increase)
time_series_wise_SMAPE_DC <- subset(time_series_wise_SMAPE_B, TWP %in% NotIncrease)


SMAPE_results <- matrix(c(mean_SMAPE,
                          median_SMAPE,
                          std_SMAPE,
                          mean(time_series_wise_SMAPE_IN$TOTAL),
                          median(time_series_wise_SMAPE_IN$TOTAL),
                          sd(time_series_wise_SMAPE_IN$TOTAL),
                          mean(time_series_wise_SMAPE_DC$TOTAL),
                          median(time_series_wise_SMAPE_DC$TOTAL),
                          sd(time_series_wise_SMAPE_DC$TOTAL)),ncol=3,byrow=TRUE)

colnames(SMAPE_results) <- c("Mean","Median","StDev")
rownames(SMAPE_results) <- c("All LGAs","Increase Group", "Decrease Group")

SMAPE_results <- as.table(SMAPE_results)
SMAPE_results


