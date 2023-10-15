## DATASET: NASS AUSTRALIAN EMS CALLS 
# FORECASTING TASK FOR COUNTERFACTUAL 
# FORECASTING OF COUNTERFACTUAL FOR PERIOD OF JUN-18 TO MAY-19 (12 time-period), TRAINING DATASET: JAN-15 TO MAY-18

# Please uncomment the below command, in case you haven't installed the following package in your enviornment.
# install.packages("forecast")
library(forecast)

# read the data
ems_dataset <- read.csv(file = "./datasets/text_data/EMS/ems_dataset.txt", sep = ',', header = FALSE) # data with training period from Jan-15 to May-18
ems_dataset <- as.matrix(ems_dataset)

output_file_name = "./results/arima_forecasts/ems.txt"

unlink(output_file_name)

# calculate the arima forecasts
for (i in 1 : nrow(ems_dataset)) {

    time_series = ems_dataset[i,]
    fit = NULL
    forecasts = NULL
     
    tryCatch({
        fit = auto.arima(ts(time_series, frequency = 12), lambda = 0) # considering seasonality of 12 (monthly data)
    }, warning = function(e) {
      print(e)
    })
    
    if(is.null(fit)){
      tryCatch({
        fit = auto.arima(ts(time_series, frequency = 12))
      }, warning = function(e) {
        print(e)
      })
      if(is.null(fit)){
        fit = auto.arima(ts(time_series, frequency = 12), seasonal = FALSE)
      }
      
    }
    forecasts = forecast(fit, h=12)$mean # forecasting horizon = 12 (Jun-18 to May-19)

    # write the arima forecasts to file
    write.table(t(forecasts), file = output_file_name, row.names = F, col.names = F, sep = ",", quote = F, append = TRUE)
}
