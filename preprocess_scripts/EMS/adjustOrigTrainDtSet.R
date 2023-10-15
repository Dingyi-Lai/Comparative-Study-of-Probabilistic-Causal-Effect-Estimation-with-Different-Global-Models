## DATASET: NASS AUSTRALIAN EMS CALLS 
## SCRIPT TO SET THE TRAINING DATA TO RIGHT FORMAT TO BE READ BY FORECASTING CODE

OUTPUT_DIR="./datasets/text_data/EMS/"

input_file = "./datasets/text_data/EMS/alchol_full_license_train.txt"
file <-read.csv(file=input_file, sep=',', header = FALSE)
ems_dataset <-as.data.frame(file[-1])

output_file_name = 'ems_dataset.txt'
output_file_full_name = paste(OUTPUT_DIR, output_file_name, sep = '')

# printing the dataset to the file
write.table(ems_dataset, output_file_full_name, sep = ",", row.names = FALSE, col.names = FALSE)
