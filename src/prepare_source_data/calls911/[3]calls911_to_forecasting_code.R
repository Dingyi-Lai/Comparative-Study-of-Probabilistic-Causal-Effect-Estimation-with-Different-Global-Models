library(tidyverse)

### FORMATTING THE DATASETS TO FORECASTING TASK
calls3 <- read.csv("./data/text_data/calls911/calls3.csv")
calls911_MONT <- calls3[,-c(1,2,3,4,5,6,7,9)]
head(calls911_MONT)
####################
## all 911 calls
####################
# building the total 911 calls dataset
OUTPUT_DIR = "./data/text_data/calls911/"

## all 911 calls
calls911_MONTmon <- calls911_MONT %>% group_by(twp, yearMonth) %>%
  summarise(calls.Counter = n(), .groups = 'drop') %>%
  spread(key = yearMonth, value = calls.Counter) 

sum(calls911_MONTmon[,2:57], na.rm = TRUE) # total of calls = 649696

calls911_MONTmon[is.na(calls911_MONTmon)] <- 1 # one case

sum(calls911_MONTmon[,2:57]) # total of calls = 649696 + 1

# writing the full dataset
output_file_name = 'calls911_month_full.txt'
output_file_full_name = paste(OUTPUT_DIR, output_file_name, sep = '')
write.table(calls911_MONTmon, output_file_full_name, sep = ",",  row.names = FALSE, col.names = TRUE)

# ## train data: dec/15 to mar/19
# calls911_MONTmon_train <- calls911_MONTmon[,1:41] 
# ## test data: apr/19 to jan/20
# calls911_MONTmon_test <- calls911_MONTmon[,42:51] 

# sum(calls911_MONTmon[,2:41]) + sum(calls911_MONTmon[,42:51]) + sum(calls911_MONTmon[,52:57]) == sum(calls911_MONTmon[,2:57])

# # writing the training and testing dataset - 1st forecasting task
# output_file_name2 = 'calls911_month_train.txt'
# output_file_full_name2 = paste(OUTPUT_DIR, output_file_name2, sep = '')
# write.table(calls911_MONTmon_train, output_file_full_name2, sep = ",",  row.names = FALSE, col.names = FALSE)

# output_file_name3 = 'calls911_month_test.txt'
# output_file_full_name3 = paste(OUTPUT_DIR, output_file_name3, sep = '')
# write.table(calls911_MONTmon_test, output_file_full_name3, sep = ",",  row.names = FALSE, col.names = FALSE)

## train data: dec/15 to dec/19
calls911_MONTmon_train2 <- calls911_MONTmon[,1:50] 
## test data: jan/20 to jul/20
calls911_MONTmon_test2 <- calls911_MONTmon[,51:57] 

sum(calls911_MONTmon[,2:50]) + sum(calls911_MONTmon[,51:57]) == sum(calls911_MONTmon[,2:57])

# writing the training and testing dataset - counterfactual forecasting task
output_file_name3 = 'calls911_month_train.txt'
output_file_full_name3 = paste(OUTPUT_DIR, output_file_name3, sep = '')
write.table(calls911_MONTmon_train2, output_file_full_name3, sep = ",",  row.names = FALSE, col.names = FALSE)

output_file_name4 = 'calls911_month_test.txt'
output_file_full_name4 = paste(OUTPUT_DIR, output_file_name4, sep = '')
write.table(calls911_MONTmon_test2, output_file_full_name4, sep = ",",  row.names = FALSE, col.names = FALSE)
