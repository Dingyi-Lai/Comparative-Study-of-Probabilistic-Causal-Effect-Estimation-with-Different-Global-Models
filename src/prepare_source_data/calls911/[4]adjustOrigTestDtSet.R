## DATASET: 911 EMERGENCY CALL MONTEGOMERY COUNTY
## SCRIPT TO SET THE TESTING DATA TO RIGHT FORMAT TO BE READ BY FORECASTING CODE

OUTPUT_DIR="./data/text_data/calls911/"
output_file_name = 'callsMT2_test_actual.csv'
output_file_full_name = paste(OUTPUT_DIR, output_file_name, sep = '')

input_file = "./data/text_data/calls911/calls911_month_test.txt"
file <-read.csv(file=input_file, header = FALSE)
callsMT2_result_dataset <-as.data.frame(file)

# printing the results to the file
write.csv(callsMT2_result_dataset, output_file_full_name, col.names = FALSE)
