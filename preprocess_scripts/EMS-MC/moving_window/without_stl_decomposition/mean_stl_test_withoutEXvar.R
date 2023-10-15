## DATASET: 911 EMERGENCY CALL MONTEGOMERY COUNTY
# FORECASTING TASK FOR COUNTERFACTUAL CONSIDERING THE EFFECT OF COVID-19 LOCKDOWN.
# FORECASTING OF COUNTERFACTUAL FOR PERIOD OF JAN-20 TO JUL-20, TRAINING DATASET: DEC-15 TO DEC-19

## Preprocessing script code that: (1) split the training dataset creating the testing file dataset; (2) applys the preprocessing steps to these testing dataset;

# Please uncomment the below command, in case you haven't installed the following pakcage in your enviornment.
# install.packages("forecast")
require(forecast)
library(forecast)


# reproducability.
set.seed(1234)

# Loading the training dataset.
input_file = "./datasets/text_data/EMS-MC/calls911_month_train2.txt" # train dataset with training period from Dec-15 to Dec-19
df_train <- read.csv(file=input_file, header = FALSE)
#df_licenses <- read.csv("license.txt", header = FALSE) # in this 2nd experiment there isn't an external exogenous variable to be added

# Defining output directory, input window size, forecasting horizon, and seasonality respectively.
OUTPUT_DIR = "./datasets/text_data/EMS-MC/moving_window/without_stl_decomposition/"
suppressWarnings(dir.create(OUTPUT_DIR, recursive=TRUE)) # create the output directory if not existing

input_size = 15 # it's the result of (seasonality_period * 1.25)
max_forecast_horizon <- 7
seasonality_period <- 12

for (idr in 1:nrow(df_train)) {
  print(idr)
  OUTPUT_PATH = paste(OUTPUT_DIR, "callsMT2_test_", sep = '/')
  OUTPUT_PATH = paste(OUTPUT_PATH, max_forecast_horizon, sep = '')
  OUTPUT_PATH = paste(OUTPUT_PATH, 'i', input_size, sep = '')
  OUTPUT_PATH = paste(OUTPUT_PATH, 'txt', sep = '.')
  
  time_series_data <- as.numeric(df_train[idr,c(2:50)])
  #licence_ts <- as.numeric(df_licenses[idr,c(2:42)])
  #licence_ts <- round(licence_ts*1.1)
  
  #licence_ts_mean <- mean(licence_ts)
  #licence_final_ts <- licence_ts / (licence_ts_mean)
  
  #aggregated_timeseries <- licence_final_ts
  
  time_series_mean <- mean(time_series_data)
  time_series_data <- time_series_data / (time_series_mean)
  
  # Performing log operation on the time series.
  time_series_log <- log(time_series_data)
  time_series_length = length(time_series_log)
  
  decomp_result = tryCatch({
    sstl = stl(ts(time_series_log, frequency = seasonality_period),
               s.window = "period")
    seasonal_vect = as.numeric(sstl$time.series[, 1])
    levels_vect = as.numeric(sstl$time.series[, 2])
    values_vect = as.numeric(sstl$time.series[, 2] + sstl$time.series[, 3])
    cbind(seasonal_vect, levels_vect, values_vect)
  }, error = function(e) {
    seasonal_vect = rep(0, length(time_series_length))#stl() may fail, and then we would go on with the seasonality vector=0
    levels_vect = time_series_log
    values_vect = time_series_log
    cbind(seasonal_vect, levels_vect, values_vect)
  })
  
  # Generating input and output windows using the original time series.
  input_windows = embed(time_series_log[1:(time_series_length)], input_size)[, input_size:1]
  # Generating seasonal components to use as exogenous variables.
  #exogenous_windows = embed(aggregated_timeseries[1:(time_series_length)], input_size)[, input_size:1]
  seasonality_windows = embed(decomp_result [1:(time_series_length), 1], input_size)[, input_size:1]
  seasonality_windows =  seasonality_windows[, c(15)] # c(value of input_size)
  # Generating the final window values.
  meanvalues <- rowMeans(input_windows)
  input_windows <- input_windows - meanvalues
  
  # Saving into a dataframe with the respective values.
  sav_df = matrix(NA,
                  ncol = (4 + input_size + 1),
                  nrow = nrow(input_windows))
  sav_df = as.data.frame(sav_df)
  sav_df[, 1] = paste(idr - 1, '|i', sep = '')
  #sav_df[, 2:(input_size + 1)] = exogenous_windows
  sav_df[, 2] = seasonality_windows
  #sav_df[, (input_size + 2)] = seasonality_windows
  sav_df[, 3:(input_size + 2)] = input_windows
  #sav_df[, (input_size + 3):(input_size*2 + 1 + 1)] = input_windows
  sav_df[, (input_size + 2 + 1)] = '|#'
  #sav_df[, (input_size*2 + 1 + 2)] = '|#'
  sav_df[, (input_size + 2 + 2)] = time_series_mean
  #sav_df[, (input_size*2 + 1 + 3)] = time_series_mean
  sav_df[, (input_size + 2 + 3)] = meanvalues
  #sav_df[, (input_size*2 + 1 + 4)] = meanvalues
  
  # Writing the dataframe into a file.
  write.table(
    sav_df,
    file = OUTPUT_PATH,
    row.names = F,
    col.names = F,
    sep = " ",
    quote = F,
    append = TRUE
  )
}
