## DATASET: 911 EMERGENCY CALL MONTEGOMERY COUNTY
## SCRIPT TO SET THE TRAINING DATA TO RIGHT FORMAT TO BE READ BY FORECASTING CODE

OUTPUT_DIR="./data/text_data/calls911/"

input_file = "./data/text_data/calls911/calls911_month_train.txt"
file <-read.csv(file=input_file, sep=',', header = FALSE)
callsMT2_dataset <-as.data.frame(file[-c(1)])

output_file_name = 'callsMT2_train.csv'
output_file_full_name = paste(OUTPUT_DIR, output_file_name, sep = '')

# printing the dataset to the file
write.csv(callsMT2_dataset, output_file_full_name, sep = ",", row.names = FALSE, col.names = FALSE)
