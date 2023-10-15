## DATASET: NASS AUSTRALIAN EMS CALLS 
## SCRIPT TO SET THE TESTING DATA TO RIGHT FORMAT TO BE READ BY FORECASTING CODE

OUTPUT_DIR="./datasets/text_data/EMS/"
output_file_name = 'ems_results.txt'
output_file_full_name = paste(OUTPUT_DIR, output_file_name, sep = '')

input_file = "./datasets/text_data/EMS/alchol_full_license_test.txt"
file <-read.csv(file=input_file, header = FALSE)
ems_result_dataset <-as.data.frame(file)

# printing the results to the file
write.table(ems_result_dataset, output_file_full_name, sep = ";", row.names = TRUE, col.names = FALSE)
