## DATASET: 911 EMERGENCY CALL MONTEGOMERY COUNTY
## SCRIPT TO SET THE TRAINING DATA TO RIGHT FORMAT TO BE READ BY FORECASTING CODE

OUTPUT_DIR="./datasets/text_data/EMS-MC/"

input_file = "./datasets/text_data/EMS-MC/calls911_month_train2.txt"
file <-read.csv(file=input_file, sep=',', header = FALSE)
callsMT2_dataset <-as.data.frame(file[-c(1)])

output_file_name = 'callsMT2_dataset.txt'
output_file_full_name = paste(OUTPUT_DIR, output_file_name, sep = '')

# printing the dataset to the file
write.table(callsMT2_dataset, output_file_full_name, sep = ",", row.names = FALSE, col.names = FALSE)
