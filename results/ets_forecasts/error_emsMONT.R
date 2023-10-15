# final evaluation ETS

emsMONT_forecasts <- read.csv(file = "./results/ets_forecasts/emsMONT.txt", sep = ',', header = FALSE)
actual_results_df <- read.csv(file = "./datasets/text_data/EMS-MC/callsMT2_results.txt", sep = ';', header = FALSE)
actual_results_df <- actual_results_df[,-1]


#### SMAPE
time_series_wise_SMAPE <- 2 * abs(emsMONT_forecasts - actual_results_df) / (abs(emsMONT_forecasts) + abs(actual_results_df))
SMAPEPerSeries <- rowMeans(time_series_wise_SMAPE, na.rm = TRUE)

mean_SMAPE = mean(SMAPEPerSeries)
median_SMAPE = median(SMAPEPerSeries)
std_SMAPE = sd(SMAPEPerSeries)

mean_SMAPE
median_SMAPE
std_SMAPE

train_ds <- read.csv("./datasets/text_data/EMS-MC/calls911_month_train2.txt", sep = ',', header = FALSE)
time_series_wise_SMAPE_B <- time_series_wise_SMAPE
time_series_wise_SMAPE_B <- cbind(train_ds[,1],time_series_wise_SMAPE_B)
time_series_wise_SMAPE_C <- t(time_series_wise_SMAPE_B)

time_series_wise_SMAPE_B$TOTAL <- rowMeans(time_series_wise_SMAPE_B[,2:8])

control_list3 <- c("BRIDGEPORT", "BRYN ATHYN", "DOUGLASS", "HATBORO", "HATFIELD BORO", 
                   "LOWER FREDERICK", "NEW HANOVER", "NORRISTOWN", "NORTH WALES", "SALFORD", 
                   "SPRINGFIELD", "TRAPPE")

treated_list3 <- c("ABINGTON",  "AMBLER",  "CHELTENHAM",  "COLLEGEVILLE",  "CONSHOHOCKEN", 
                   "EAST GREENVILLE",  "EAST NORRITON",  "FRANCONIA" , "GREEN LANE", "HATFIELD TOWNSHIP", 
                   "HORSHAM" , "JENKINTOWN",  "LANSDALE",  "LIMERICK",  "LOWER GWYNEDD", 
                   "LOWER MERION",  "LOWER MORELAND",  "LOWER POTTSGROVE",  "LOWER PROVIDENCE",  "LOWER SALFORD", 
                   "MARLBOROUGH",  "MONTGOMERY",  "NARBERTH",  "PENNSBURG",  "PERKIOMEN", 
                   "PLYMOUTH",  "POTTSTOWN",  "RED HILL",  "ROCKLEDGE",  "ROYERSFORD", 
                   "SCHWENKSVILLE",  "SKIPPACK",  "SOUDERTON",  "TELFORD",  "TOWAMENCIN", 
                   "UPPER DUBLIN",  "UPPER FREDERICK",  "UPPER GWYNEDD",  "UPPER HANOVER",  "UPPER MERION", 
                   "UPPER MORELAND",  "UPPER POTTSGROVE",  "UPPER PROVIDENCE",  "UPPER SALFORD",  "WEST CONSHOHOCKEN", 
                   "WEST NORRITON",  "WEST POTTSGROVE",  "WHITEMARSH",  "WHITPAIN",  "WORCESTER")

names(time_series_wise_SMAPE_B)[1] <- "TWP"
colnames(time_series_wise_SMAPE_B)

time_series_wise_SMAPE_CL  <- subset(time_series_wise_SMAPE_B, TWP %in% control_list3)
time_series_wise_SMAPE_TR <- subset(time_series_wise_SMAPE_B, TWP %in% treated_list3)

SMAPE_results <- matrix(c(mean_SMAPE,
                          median_SMAPE,
                          std_SMAPE,
                          mean(time_series_wise_SMAPE_CL$TOTAL),
                          median(time_series_wise_SMAPE_CL$TOTAL),
                          sd(time_series_wise_SMAPE_CL$TOTAL),
                          mean(time_series_wise_SMAPE_TR$TOTAL),
                          median(time_series_wise_SMAPE_TR$TOTAL),
                          sd(time_series_wise_SMAPE_TR$TOTAL)),ncol=3,byrow=TRUE)

colnames(SMAPE_results) <- c("Mean","Median","StDev")
rownames(SMAPE_results) <- c("All TWP","Control Group", "Treated Group")

SMAPE_results <- as.table(SMAPE_results)
SMAPE_results

