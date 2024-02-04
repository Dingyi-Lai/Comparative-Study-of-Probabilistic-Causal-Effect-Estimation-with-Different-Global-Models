## DATASET: 911 EMERGENCY CALL MONTEGOMERY COUNTY
# FORECASTING TASK FOR COUNTERFACTUAL CONSIDERING THE EFFECT OF COVID-19 LOCKDOWN.
# FORECASTING OF COUNTERFACTUAL FOR PERIOD OF JAN-20 TO JUL-20, TRAINING DATASET: DEC-15 TO DEC-19

## Preprocessing script code that: (1) split the training dataset creating the training and validation files dataset; (2) applys the preprocessing steps to these both datasets

# Please uncomment the below command, in case you haven't installed the following pakcage in your enviornment.
# install.packages("forecast")
library(forecast)
  
# reproducability.
set.seed(1234)

# Loading the training dataset.
input_file = "./datasets/text_data/calls911/calls911_month_train2.txt" # train dataset with training period from Dec-15 to Dec-19
df_train <- read.csv(file=input_file, header = FALSE)
#df_licenses <- read.csv("license.txt", header = FALSE) # in this 2nd experiment there isn't an external exogenous variable to be added

# Defining output directory, input window size, forecasting horizon, and seasonality respectively.
OUTPUT_DIR = "./datasets/text_data/calls911/moving_window/"
suppressWarnings(dir.create(OUTPUT_DIR, recursive=TRUE)) # create the output directory if not existing

input_size = 15 # it's the result of (seasonality_period * 1.25)
max_forecast_horizon <- 7
seasonality_period <- 12
validation <- TRUE
for (validation in c(TRUE, FALSE)) {
  for (idr in 1 : nrow(df_train)) {
    print(idr)
    OUTPUT_PATH = paste(OUTPUT_DIR, "callsMT2_", sep = '/')
    if (validation) {
      OUTPUT_PATH = paste(OUTPUT_PATH, 'validation_', sep = '')
    }else{
      OUTPUT_PATH = paste(OUTPUT_PATH, 'train_', sep = '')
    }
    OUTPUT_PATH = paste(OUTPUT_PATH, max_forecast_horizon, sep = '')
    OUTPUT_PATH = paste(OUTPUT_PATH, '_', input_size,'new', sep = '')
    OUTPUT_PATH = paste(OUTPUT_PATH, 'txt', sep = '.')
      
    time_series_data <- as.numeric(df_train[idr,c(2:50)])
    #licence_ts <- as.numeric(df_licenses[idr,c(2:42)])
    #licence_ts <- round(licence_ts*1.1)
    
    #licence_ts_mean <- mean(licence_ts)
    #licence_final_ts <- licence_ts / (licence_ts_mean)
    
    #aggregated_timeseries <- licence_final_ts    
    # time_series_length = length(time_series_data)
    # if (! validation) { # first split the data, then do the standardization and stl decomposition
    #   time_series_length = time_series_length - max_forecast_horizon
    #   time_series_data = time_series_data[1 : time_series_length]
    # }

    time_series_mean <- mean(time_series_data)
    time_series_data <- time_series_data / (time_series_mean)
    
    time_series_log <- log(time_series_data)
    time_series_length = length(time_series_log)

    if (! validation) {
      time_series_length = time_series_length - max_forecast_horizon
      time_series_log = time_series_log[1 : time_series_length]
    }
    # apply stl
    decomp_result = tryCatch({
      sstl = stl(ts(time_series_log, frequency = seasonality_period), s.window = "period")
      seasonal_vect = as.numeric(sstl$time.series[,1])
      levels_vect = as.numeric(sstl$time.series[,2])
      values_vect=as.numeric(sstl$time.series[,2]+sstl$time.series[,3])
      cbind(seasonal_vect, levels_vect, values_vect)
      }, error = function(e) {
      seasonal_vect = rep(0, length(time_series_log))#stl() may fail, and then we would go on with the seasonality vector=0
      levels_vect = time_series_log
      values_vect = time_series_log
      cbind(seasonal_vect, levels_vect, values_vect)
      })
      
    # Generating input and output windows using the original time series.
    input_windows = embed(time_series_log[1:(time_series_length - max_forecast_horizon)], input_size)[, input_size:1]
    #exogenous_windows = embed(aggregated_timeseries[1:(time_series_length - max_forecast_horizon)], input_size)[, input_size:1]
    output_windows = embed(time_series_log[-(1:input_size)], max_forecast_horizon)[, max_forecast_horizon:1]
    # Generating seasonal components to use as exogenous variables.
    seasonality_windows = embed(decomp_result[1:(time_series_length - max_forecast_horizon), 1], input_size)[, input_size:1]
    seasonality_windows =  seasonality_windows[, c(input_size)] # c(value of input_size)
    # Generating the final window values.
    meanvalues <- rowMeans(input_windows)
    input_windows <- input_windows - meanvalues
    output_windows <- output_windows - meanvalues
    
    if (validation) {
      # Saving into a dataframe with the respective values.
      sav_df = matrix(
        NA,
        ncol = (5 + input_size + 1 + max_forecast_horizon),
        nrow = nrow(input_windows)
      )
      sav_df = as.data.frame(sav_df)
      sav_df[, 1] = paste(idr - 1, '|i', sep = '')
      #sav_df[, 2:(input_size + 1)] = exogenous_windows
      sav_df[, 2] = seasonality_windows
      #sav_df[, (input_size + 2)] = seasonality_windows
      sav_df[, 3:(input_size + 2)] = input_windows
      #sav_df[, (input_size + 3):(input_size*2 + 1 + 1)] = input_windows
      sav_df[, (input_size + 2 + 1)] = '|o'
      #sav_df[, (input_size*2 + 1 + 2)] = '|o'
      sav_df[, (input_size + 2 + 1 + 1):(input_size + 2 + 1 + max_forecast_horizon)] = output_windows
      #sav_df[, (input_size*2 + 1 + 3):(input_size*2 + 2 + max_forecast_horizon  + 1)] = output_windows
      sav_df[, (input_size + max_forecast_horizon + 4)] = '|#'
      #sav_df[, (input_size*2 + 1 + max_forecast_horizon  + 3)] = '|#'
      sav_df[, (input_size + max_forecast_horizon + 5)] = time_series_mean
      #sav_df[, (input_size*2 + 1 + max_forecast_horizon  + 4)] = time_series_mean
      sav_df[, (input_size + max_forecast_horizon + 6)] = meanvalues
      # #sav_df[, (input_size*2 + 1 + max_forecast_horizon + 5)] = meanvalues
    } else {
      sav_df = matrix(
        NA,
        ncol = (2 + input_size + 1 + max_forecast_horizon),
        nrow = nrow(input_windows)
      )
      sav_df = as.data.frame(sav_df)
      sav_df[, 1] = paste(idr - 1, '|i', sep = '')
      #sav_df[, 2:(input_size + 1)] = exogenous_windows
      sav_df[, 2] = seasonality_windows
      #sav_df[, (input_size + 2)] = seasonality_windows
      sav_df[, 3:(input_size + 2)] = input_windows
      #sav_df[, (input_size + 3):(input_size*2 + 1 + 1)] = input_windows
      sav_df[, (input_size + 2 + 1)] = '|o'
      #sav_df[, (input_size*2 + 1 + 2)] = '|o'
      sav_df[, (input_size + 2 + 1 + 1):(input_size + 2 + 1 + max_forecast_horizon)] = output_windows
      #sav_df[, (input_size*2 + 1 + 3):(input_size*2 + 2 + max_forecast_horizon  + 1)] = output_windows
    }
    
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
}
