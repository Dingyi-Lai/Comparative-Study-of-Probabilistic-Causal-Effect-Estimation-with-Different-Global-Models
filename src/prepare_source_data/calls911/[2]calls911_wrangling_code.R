## DATA WRANGLING process through the 911 emergency calls for Montgomery County - PA - USA

### Loading Data: the raw data was downloaded from this source:
### Dataset from Kaggle containing the 911 emergency calls from Montgomery County (Pennsylvania-USA) from Dec/2015, 
### and retrieved from https://www.kaggle.com/mchirico/montcoalert.

# libraries
library(stringr)
library(dplyr)

# After downloaded the raw data to my local machine (and naming it as '911.csv'), the following code was applied in order to wrangle the raw data.

# Loading the data
calls <- read.csv('911.csv', header=T, na.strings=c(""," ", "NA")) 

# spliting the variable `title` into `reason` and `code`, and cleaning the from some descriptions.
title_split <- str_split_fixed(calls$title,": ", n=2)
calls1 <- calls
calls1$reason <- title_split[,1]
calls1$code <- title_split[,2]
calls1$code <- str_replace(calls1$code, " -", "")

# formatting the date information
calls1$timeStamp2 <- as.POSIXct(calls1$timeStamp, format = "%Y-%m-%d %H:%M:%S")
calls1$date <- as.Date(calls1$timeStamp)
calls1$year <- as.numeric(format(calls1$date, "%Y"))
calls1$month <- as.numeric(format(calls1$date, "%m"))
calls1$dayWeek <- as.numeric(format(calls1$date, "%u"))
calls1$dayWeek2 <- recode_factor(calls1$dayWeek, 
                                 "1"="Monday",
                                 "2"="Tuesday",
                                 "3"="Wednesday",
                                 "4"="Thursday",
                                 "5"="Friday",
                                 "6"="Saturday",
                                 "7"="Sunday")
calls1$hour <- as.numeric(format(calls1$timeStamp2, "%H"))
calls1$yearMonth <- format(calls1$date, "%Y-%m")

# treating the missing data in 'hour' column -> no na
# calls1 %>% 
#   filter(is.na(hour))

# # Replacing missing values in hour variable
# calls1[,18][is.na(calls1[,18])] <- 2
# apply(calls1, 2, function(x) sum(is.na(x)))
# cat("Missings in hour:", sum(is.na(calls1$hour)))

# Cleaning the townships from everything different of Montgomery County
unique(calls1$twp)
calls2 <- calls1 %>%
  filter(!twp %in% c("BERKS COUNTY", "BUCKS COUNTY", "CHESTER COUNTY", "DELAWARE COUNTY", "LEHIGH COUNTY", "LYCOMING COUNTY", "PHILA COUNTY"))

unique(calls2$twp) # check it again in https://en.wikipedia.org/wiki/Category:Townships_in_Montgomery_County,_Pennsylvania
length(unique(calls2$twp))
apply(calls2, 2, function(x) sum(is.na(x)))

# The number of missings remaining in townships is 293, which corresponds to 0.045% of all records. 
# From this 293 missings, for 124 of them we have information about the zip, and from these 124 zips, 102 are unique lat/lon/zips. 
# As the task to fill the `twp` column searching for those 102 lat/lon/zips would be very time consuming, and the amount of 
# those missings are not representative, I've decided to clean these missings from the dataset.
calls2 %>%
  filter(is.na(twp)) %>%
  head()

# Cleaning missings townships from dataset
cat("Total of missings per each variable of final dataset:", "\n")
calls3 <- calls2 %>%
  filter(!is.na(twp))
apply(calls3, 2, function(x) sum(is.na(x)))
unique(calls3$twp)
cat("Final dimension of Dataset 1: ", dim(calls3)[1], " rows", " X ", dim(calls3)[2], " columns")

# saving dataframe calls3
write.csv(calls3, './data/text_data/calls911/calls3.csv') # save this dataset in some local directory



