## DATASET: 911 EMERGENCY CALL MONTEGOMERY COUNTY
## SCRIPT TO SET THE TESTING DATA TO RIGHT FORMAT TO BE READ BY FORECASTING CODE

OUTPUT_DIR="./datasets/text_data/EMS-MC/"
output_file_name = 'callsMT2_results.txt'
output_file_full_name = paste(OUTPUT_DIR, output_file_name, sep = '')

input_file = "./datasets/text_data/EMS-MC/calls911_month_test2.txt"
file <-read.csv(file=input_file, header = FALSE)
callsMT2_result_dataset <-as.data.frame(file)

# printing the results to the file
write.table(callsMT2_result_dataset, output_file_full_name, sep = ";", row.names = TRUE, col.names = FALSE)
