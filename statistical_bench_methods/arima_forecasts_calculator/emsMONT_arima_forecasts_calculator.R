## DATASET: 911 EMERGENCY CALL MONTEGOMERY COUNTY
# FORECASTING TASK FOR COUNTERFACTUAL CONSIDERING THE EFFECT OF COVID-19 LOCKDOWN.
# FORECASTING OF COUNTERFACTUAL FOR PERIOD OF JAN-20 TO JUL-20, TRAINING DATASET: DEC-15 TO DEC-19

# Please uncomment the below command, in case you haven't installed the following package in your enviornment.
# install.packages("forecast")
library(forecast)

# read the data
emsMONT_dataset <- read.csv(file = "./datasets/text_data/EMS-MC/callsMT2_dataset.txt", sep = ',', header = FALSE) # data with training period from Dec-15 to Dec-19
emsMONT_dataset <- as.matrix(emsMONT_dataset)

output_file_name = "./results/arima_forecasts/emsMONT.txt"

unlink(output_file_name)

# calculate the arima forecasts
for (i in 1 : nrow(emsMONT_dataset)) {

    time_series = emsMONT_dataset[i,]
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
    forecasts = forecast(fit, h=7)$mean # forecasting horizon = 7 (Jan-20 to Jul-20)

    # write the arima forecasts to file
    write.table(t(forecasts), file = output_file_name, row.names = F, col.names = F, sep = ",", quote = F, append = TRUE)
}
