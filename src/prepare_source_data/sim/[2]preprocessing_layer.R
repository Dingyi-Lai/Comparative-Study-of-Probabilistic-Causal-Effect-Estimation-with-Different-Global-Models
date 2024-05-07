# Take synthetic data as example
# 1. Split the training dataset creating the testing file dataset; 
# 2. For the usage of training, validation and testing, create moving windows accordingly and store them into seperate file 
# 3. Applys the preprocessing steps to these scenario:
# 3.1. Normalization
# 3.2. Log transformation
# 3.3. Decomposition: STL
# 4. store them into tfrecord files (in another .py file)

# reproducability.
set.seed(1234)

library(dplyr)

# Read the data
# setwd("data/text_data/sim/scenario_df.csv")
scenario_df <- read.csv("data/text_data/sim/scenario_df.csv")
scenario_df_true <- read.csv("data/text_data/sim/scenario_df_true.csv")

# Defining output directory, input window size, forecasting horizon, and seasonality respectively.
OUTPUT_DIR = "./data/text_data/sim/moving_window/"
suppressWarnings(dir.create(OUTPUT_DIR, recursive=TRUE)) # create the output directory if not existing

# Parametrs of scenarios
# Define the time series lengths
time_series_lengths <- c(60, 222)
# Define the amounts of time series
amount_of_time_series <- c(10, 101, 500)
# Define the type of intervention
te_intervention <- c("ho", "he")
# Define the DGPs (Linear and Nonlinear)
dgp <- c("l", "nl")

# from Grecov's codes
max_forecast_horizon <- 12
seasonality_period <- 12
input_size <- seasonality_period * 1.25 # 15

# Split train and test set in a loop
# count <- 0
# length <- 60
# amount <- 10
# dgp <- "linear"
# ln <- "linear"
# te <- "homogeneous"
for (length in time_series_lengths) {
  for (amount in amount_of_time_series) {
    for (ln in dgp) {
      for (te in te_intervention){ 
        
        # train data
        nam <- paste("sim", amount, length, ln, te, "train", sep = "_")
        data_train <- scenario_df %>%
            filter(time_series_length == length & 
            amount_of_time_series == amount &
            dgp == ln &
            te_intervention == te &
            time < length - 11) # Intervention at T0 = 49 or 211, prediction range is 12
        assign(nam, data_train)
        write.csv(data_train, paste0("data/text_data/sim/", nam, ".csv"), row.names = FALSE)

        for(file_type in c("train", "validation", "test")){
            for(i in c(1:amount)){
                # print(i)
                series_id_i <- paste(amount,i,length,ln,te, sep='_')
                data_train_i <- data_train[data_train$series_id == series_id_i, "value"]

                # mean-scale normalization
                # The division instead of subtraction further helps to scale all the time series to a similar range which helps the RNN learning process.
                time_series_mean <- mean(data_train_i)
                data_train_i <- data_train_i / time_series_mean

                # log transformation
                # Stabilizing the Variance
                data_train_i_log <- log(data_train_i)
                data_length <- length(data_train_i_log)
                # The reason why we have three cases of data formats is that the underlying patterns may change during this last part of the sequence if we split the data into train(data_length-48) - validation(12) - test(12).
                # Therefore, in this work, the aforementioned split is used only for the validation phase. For testing, the model is re-trained using the whole sequence without any data split. 
                # For each dataset, the models are trained using all the time series available, for the purpose of developing a global model. 
                # Different RNN architectures need to be fed data in different formats. (Hewamalage et al.)

            
                if(file_type=="train"){
                    # train
                    # l - n*2 - m
                    time_series_log <- data_train_i_log[1 : (data_length - max_forecast_horizon)]
                    time_series_len <- data_length - 2*max_forecast_horizon
                }
                if(file_type=="validation"){
                    # validation
                    # l - n - m
                    time_series_log <- data_train_i_log
                    time_series_len <- data_length - max_forecast_horizon
                }
                if(file_type=="test"){
                    # test
                    time_series_log <- data_train_i_log
                    time_series_len <- data_length
                }

                # The STL Decomposition is potentially able to allow the seasonality to change over time. 
                # However, we use STL in a deterministic way, where we assume the seasonality of all the time series to be fixed along the whole time span (Hewamalage et al.)
                # apply stl
                decomp_result = tryCatch({
                    sstl = stl(ts(time_series_log, frequency = seasonality_period), s.window = "period")
                    seasonal_vect = as.numeric(sstl$time.series[,1])
                    levels_vect = as.numeric(sstl$time.series[,2])
                    values_vect=as.numeric(sstl$time.series[,2]+
                        sstl$time.series[,3])
                    # By specifically setting the s.window parameter in the stl method to "periodic", we make the seasonality deterministic. 
                    # Hence, we remove only the deterministic seasonality component from the time series while other stochastic seasonality components may still remain. (Hewamalage et al.)
                    cbind(seasonal_vect, levels_vect, values_vect)
                    }, error = function(e) {
                    seasonal_vect = rep(0, length(time_series_log))#stl() may fail, and then we would go on with the seasonality vector=0
                    # this technique requires at least two full periods of the time series data to determine its seasonality component. 
                    # In extreme cases where the full length of the series is less than two periods, the technique considers such sequences as having no seasonality and returns 0 for the seasonality component. (Hewamalage et al.)
                    levels_vect = time_series_log
                    values_vect = time_series_log
                    cbind(seasonal_vect, levels_vect, values_vect)
                    })
                
                # Generating input and output windows using the original time series.
                input_windows = embed(time_series_log[1:time_series_len], input_size)[, input_size:1]
                #exogenous_windows = embed(aggregated_timeseries[1:time_series_len], input_size)[, input_size:1]
                # Generating seasonal components to use as exogenous variables.
                seasonality_windows = embed(decomp_result[1:time_series_len, 1], input_size)[, input_size:1]
                seasonality_windows =  seasonality_windows[, c(input_size)]
                # Generating the final window values.
                meanvalues <- rowMeans(input_windows)
                input_windows <- input_windows - meanvalues
                if(file_type!="test"){
                    output_windows = embed(time_series_log[-(1:input_size)], max_forecast_horizon)[, max_forecast_horizon:1]
                    output_windows <- output_windows - meanvalues
                }
                # Saving into a dataframe with the respective values.
                if(file_type=="train"){
                    # train
                    ser_id <- paste("sim", amount,length,ln,te,"train", sep='_')
                    OUTPUT_PATH = paste0(OUTPUT_DIR, paste(ser_id, max_forecast_horizon, input_size, sep = '_'))
                    sav_df = matrix(
                        NA,
                        ncol = (2 + input_size + 1 + max_forecast_horizon),
                        nrow = nrow(input_windows)
                    )
                    sav_df = as.data.frame(sav_df)
                    sav_df[, 1] = paste(i - 1, '|i', sep = '')
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
                if(file_type=="validation"){
                    # validation
                    ser_id <- paste("sim", amount,length,ln,te,"validation", sep='_')
                    OUTPUT_PATH = paste0(OUTPUT_DIR, paste(ser_id, max_forecast_horizon, input_size, sep = '_'))
                    sav_df = matrix(
                            NA,
                            ncol = (5 + input_size + 1 + max_forecast_horizon),
                            nrow = nrow(input_windows)
                        )
                    sav_df = as.data.frame(sav_df)
                    sav_df[, 1] = paste(i - 1, '|i', sep = '')
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
                    #sav_df[, (input_size*2 + 1 + max_forecast_horizon + 5)] = meanvalues
                }
                if(file_type=="test"){
                    # test
                    ser_id <- paste("sim", amount,length,ln,te,"test", sep='_')
                    OUTPUT_PATH = paste0(OUTPUT_DIR, paste(ser_id, max_forecast_horizon, input_size, sep = '_'))
                    sav_df = matrix(NA,
                                    ncol = (4 + input_size + 1),
                                    nrow = nrow(input_windows))
                    sav_df = as.data.frame(sav_df)
                    sav_df[, 1] = paste(i - 1, '|i', sep = '')
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
                }
                # Writing the dataframe into a file. 
                OUTPUT_PATH = paste(OUTPUT_PATH, 'txt', sep = '.')
                # print(dim(sav_df))
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
        
        # test data as actual data
        nam <- paste("sim", amount, length, ln, te, "test_actual", sep = "_")
        data_test <- scenario_df %>%
            filter(time_series_length == length & 
            amount_of_time_series == amount &
            dgp == ln &
            te_intervention == te &
            time >= length - 11)
        assign(nam, data_test)
        write.csv(data_test, paste0("data/text_data/sim/", nam, ".csv"), row.names = FALSE)
      }
    }
  }
}


for (length in time_series_lengths) {
  for (amount in amount_of_time_series) {
    for (ln in dgp) {
      for (te in te_intervention){ 
        
        # test data as actual data
        nam <- paste("sim", amount, length, ln, te, "true_counterfactual", sep = "_")
        data_test <- scenario_df_true %>%
            filter(time_series_length == length & 
            amount_of_time_series == amount &
            dgp == ln &
            te_intervention == te)
        assign(nam, data_test)
        write.csv(data_test, paste0("data/text_data/sim/", nam, ".csv"), row.names = FALSE)
      }
    }
  }
}
