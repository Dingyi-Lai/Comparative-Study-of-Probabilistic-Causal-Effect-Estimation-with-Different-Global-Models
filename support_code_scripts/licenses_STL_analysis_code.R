### STL decomposition analysis for the NASS time series
### From chapter 6 - Book: Forecasting: Principles and Practice

# necessary library
library(forecast)
library(dplyr)
library("readxl")
library(seasonal)
library(ggplot2)
library(fpp)
library(fpp3)
library(plotly)

input_file = "./datasets/text_data/EMS/license.txt"

# reading licenses dataset:
licenses <- read.csv(file=input_file, header = FALSE)
licenses1 <-  as.matrix(as.data.frame(lapply(licenses, as.numeric)))

##########################################################
# formatting  licenses dataset
licenses2 <- data.frame(t(licenses1))
names(licenses2) <- licenses2[1,]
licenses2 <- licenses2[-1,]
Date = c( 
         "2015-01-01","2015-02-01","2015-03-01","2015-04-01","2015-05-01","2015-06-01",
         "2015-07-01","2015-08-01","2015-09-01","2015-10-01","2015-11-01","2015-12-01",
         "2016-01-01","2016-02-01","2016-03-01","2016-04-01","2016-05-01","2016-06-01",
         "2016-07-01","2016-08-01","2016-09-01","2016-10-01","2016-11-01","2016-12-01",
         "2017-01-01","2017-02-01","2017-03-01","2017-04-01","2017-05-01","2017-06-01",
         "2017-07-01","2017-08-01","2017-09-01","2017-10-01","2017-11-01","2017-12-01",
         "2018-01-01","2018-02-01","2018-03-01","2018-04-01","2018-05-01","2018-06-01",
         "2018-07-01","2018-08-01","2018-09-01","2018-10-01","2018-11-01","2018-12-01",
         "2019-01-01","2019-02-01","2019-03-01","2019-04-01","2019-05-01")
licenses2 <- cbind(Date, licenses2)

licenses2$Date <- as.Date(licenses2$Date)

head(licenses2) # taking a look

##########################################################
# normalizing by population

## Reading the excel table filled with population data
LGA_pop <- read_excel("./datasets/text_data/EMS/licenses_raw.xlsx")
LGA_pop <- data.frame(LGA_pop[-1])
licence_norm_pop <- (t(licenses2[,2:80])/LGA_pop[,3])*100
licence_norm_pop <- cbind(Date, data.frame(t(licence_norm_pop)))
licence_norm_pop$Date <- as.Date(licence_norm_pop$Date)

##########################################################
# converting to time series format
licensesNormPop_ts_LGA <- ts(licence_norm_pop[,-1], frequency=12, start=c(2015))
licensesNormPop_ts_LGA # taking a look

##########################################################
# plotting time series

# now using licenses dataset
autoplot(licensesNormPop_ts_LGA) # graph with all LGAs license time-series normalized by population

##########################################################

##### licenses dataset: STL decomposition

# total calls normalized
licence_norm_popTOT <- licence_norm_pop
licence_norm_popTOT$TOTAL <- rowSums(licence_norm_popTOT[,2:80])
# plot1: STL decomposition graph (with separeted components) with total calls normalized
ts(licence_norm_popTOT[,"TOTAL"], frequency=12, start=c(2015)) %>%
  stl(s.window="period", robust=TRUE) -> fit3
autoplot(fit3)
### plot2: 
ts(licence_norm_popTOT[,"TOTAL"], frequency=12, start=c(2015)) %>%
  autoplot(series="Data") +
  autolayer(trendcycle(fit3), series="Trend") +
  autolayer(seasadj(fit3), series="Seasonally Adjusted") +
  xlab("Year") + ylab("Number of alcohol licenses issuing index") +
  ggtitle("Monthly Australian alcohol licenses issuing - normailized data by population") +
  scale_colour_manual(values=c("gray","blue","red"),
                      breaks=c("Data","Seasonally Adjusted","Trend"))
seasadj(fit3)

############ STL Decomposition
df <- licence_norm_pop[,-1]

LGAmax <- 79
licensesDessaz = data.frame(matrix(nrow = nrow(df)))

for (i in 1:LGAmax) {
  x=colnames(df)[i]
  LGA <- ts(df[,x], frequency=12, start=c(2015)) %>%
    stl(s.window="periodic", robust=TRUE) %>%
    seasadj() 
  licensesDessaz[ , paste0(x)] <- LGA
}
licensesDessaz <- cbind(Date, licensesDessaz[,-1])
licensesDessaz$Date <- as.Date(licensesDessaz$Date)
licensesDessazTS <- ts(licensesDessaz[,-1],
                             frequency=12, start=c(2015))

ggplotly(autoplot(licensesDessazTS)) # graph with all LGAs license time-series dessazonalized

############
# TAKING THE VARIATIONS
licensesDessaz_var <- data.frame(t(licensesDessaz))
names(licensesDessaz_var) <- c( 
                         "2015-01","2015-02","2015-03","2015-04","2015-05","2015-06","2015-07","2015-08","2015-09","2015-10","2015-11","2015-12",
                         "2016-01","2016-02","2016-03","2016-04","2016-05","2016-06","2016-07","2016-08","2016-09","2016-10","2016-11","2016-12",
                         "2017-01","2017-02","2017-03","2017-04","2017-05","2017-06","2017-07","2017-08","2017-09","2017-10","2017-11","2017-12",
                         "2018-01","2018-02","2018-03","2018-04","2018-05","2018-06","2018-07","2018-08","2018-09","2018-10","2018-11","2018-12",
                         "2019-01","2019-02","2019-03","2019-04","2019-05"
)

licensesDessaz_var <- licensesDessaz_var[-1,]
licensesDessaz_var = lapply(licensesDessaz_var, as.numeric)
licensesDessaz_var <- data.frame(licensesDessaz_var)
licensesDessaz_var <- cbind(LGA_pop$LGA.id,licensesDessaz_var)
names(licensesDessaz_var) <- c("LGA.id", 
                         "2015-01","2015-02","2015-03","2015-04","2015-05","2015-06","2015-07","2015-08","2015-09","2015-10","2015-11","2015-12",
                         "2016-01","2016-02","2016-03","2016-04","2016-05","2016-06","2016-07","2016-08","2016-09","2016-10","2016-11","2016-12",
                         "2017-01","2017-02","2017-03","2017-04","2017-05","2017-06","2017-07","2017-08","2017-09","2017-10","2017-11","2017-12",
                         "2018-01","2018-02","2018-03","2018-04","2018-05","2018-06","2018-07","2018-08","2018-09","2018-10","2018-11","2018-12",
                         "2019-01","2019-02","2019-03","2019-04","2019-05")
licensesDessaz_var$var_jun17 <- round(((licensesDessaz_var$`2019-05`/licensesDessaz_var$`2017-06`)-1)*100,2)
licensesDessaz_var$var_jun18 <- round(((licensesDessaz_var$`2019-05`/licensesDessaz_var$`2018-06`)-1)*100,2)
licensesDessaz_var$var_jan18 <- round(((licensesDessaz_var$`2019-05`/licensesDessaz_var$`2018-01`)-1)*100,2)
licensesDessaz_var$var_nov18_jun18 <- round(((licensesDessaz_var$`2018-11`/licensesDessaz_var$`2016-08`)-1)*100,2)


sort(licensesDessaz_var$var_jun17)
sort(licensesDessaz_var$var_jun18)
sort(licensesDessaz_var$var_jan18)
sort(licensesDessaz_var$var_nov18_jun18)

licensesDessazAgreg <- licensesDessaz_var %>%
  group_by(LGA.id, var_jun18, var_jun17, var_jan18, var_nov18_jun18) %>%
  summarize(count = n(), .groups = 'drop')


# clustering the timeseries in groups
licensesDessaz_var <- licensesDessaz_var %>%
  mutate(group1_junNov18 = case_when(var_nov18_jun18 >= 2 ~ 'I',
                                     var_nov18_jun18 >= 1 ~ 'C',
                                  TRUE ~ 'D'))

licensesDessazAgreg <- licensesDessaz_var %>%
  group_by(group1_junNov18, LGA.id, var_jan18, var_jun18, var_jun17, var_nov18_jun18) %>%
  summarize(count = n(), .groups = 'drop')

subset(licensesDessazAgreg, group1_junNov18 == "I")[,2]
subset(licensesDessazAgreg, group1_junNov18 %in% c("C"))[,2]
subset(licensesDessazAgreg, group1_junNov18 %in% c("D"))[,2]
subset(licensesDessazAgreg, group1_junNov18 %in% c("I","C"))[,2]

paste(subset(licensesDessazAgreg, group1_junNov18 == "I")[,2])
paste(subset(licensesDessazAgreg, group1_junNov18 %in% c("I","C"))[,2])
paste(subset(licensesDessazAgreg, group1_junNov18 == "C")[,2])
paste(subset(licensesDessazAgreg, group1_junNov18 == "D")[,2])

# dataset with the increasing group
## taking variation from jun-nov/2018 >= 0% for increasing group (69 LGAs)

myvars_Dec <- names(licensesDessaz) %in% c("X21010", "X21270", "X21370", "X22670", "X24850", 
                                           "X25430", "X25810", "X26430", "X26700", "X26890") # 10 LGAs (D)

## taking variation from jun-nov/2018 >= 1% for increasing group (59 LGAs)

myvars_Dec2 <- names(licensesDessaz) %in% c("X20110", "X20830", "X21010", "X21270", "X21370", 
                                            "X21670", "X22670", "X23810", "X23940", "X24250", 
                                            "X24850", "X25430", "X25810", "X25990", "X26430", 
                                            "X26700", "X26890", "X27170", "X27450", "X27630") # 20 LGAs (D)


# licensesDessaz_I <- licensesDessaz[!myvars_Dec2]
# licensesDessaz_I$TOTAL <- rowSums(licensesDessaz_I[,2:60])
# licensesDessaz_I_TS <- ts(licensesDessaz_I[,-1], frequency=12, start=c(2015))
# autoplot(licensesDessaz_I_TS[ ,c("TOTAL")])
# 
# licensesDessaz_I[,c("Date","TOTAL")]

###################################### doing the alternative graphs for the increase group
# total calls normalized
licence_norm_pop_INC <- licence_norm_pop[!myvars_Dec]
licence_norm_pop_INC$TOTAL <- rowSums(licence_norm_pop_INC[,2:60])
# plot1: STL decomposition graph (with separeted components) with total calls normalized
ts(licence_norm_pop_INC[,"TOTAL"], frequency=12, start=c(2015)) %>%
  stl(s.window="period", robust=TRUE) -> fit4
autoplot(fit4)
### plot2: 
ts(licence_norm_pop_INC[,"TOTAL"], frequency=12, start=c(2015)) %>%
  autoplot(series="Data") +
  autolayer(trendcycle(fit4), series="Trend") +
  autolayer(seasadj(fit4), series="Seasonally Adjusted") +
  xlab("Year") + ylab("Number of alcohol licenses issuing index") +
  ggtitle("Monthly alcohol licenses issuing - normailized data by population - increasing group") +
  scale_colour_manual(values=c("gray","blue","red"),
                      breaks=c("Data","Seasonally Adjusted","Trend"))
seasadj(fit4)
########################################
# dataset with decreasing group
## taking variation from jun-nov/2018 < 0% 
licensesDessaz_DEC <- licensesDessaz[myvars_Dec]
licensesDessaz_DEC$TOTAL <- rowSums(licensesDessaz_DEC[,1:10])
licensesDessaz_DEC_TS <- ts(licensesDessaz_DEC, frequency=12, start=c(2015))
autoplot(licensesDessaz_DEC_TS[ ,c("TOTAL")])

## doing the alternative graphs for the not increase group
licence_norm_pop_DEC <- licence_norm_pop[myvars_Dec]
licence_norm_pop_DEC$TOTAL <- rowSums(licence_norm_pop_DEC[,1:10])
# plot1: STL decomposition graph (with separeted components) with total calls normalized
ts(licence_norm_pop_DEC[,"TOTAL"], frequency=12, start=c(2015)) %>%
  stl(s.window="period", robust=TRUE) -> fit5
autoplot(fit5)
### plot2: 
ts(licence_norm_pop_DEC[,"TOTAL"], frequency=12, start=c(2015)) %>%
  autoplot(series="Data") +
  autolayer(trendcycle(fit5), series="Trend") +
  autolayer(seasadj(fit5), series="Seasonally Adjusted") +
  xlab("Year") + ylab("Number of alcohol licenses issuing index") +
  ggtitle("Monthly alcohol licenses issuing - normailized data by population - decreasing group") +
  scale_colour_manual(values=c("gray","blue","red"),
                      breaks=c("Data","Seasonally Adjusted","Trend"))
seasadj(fit5)

########################################
# dataset with total cases
licensesDessaz_Tot <- licensesDessaz
licensesDessaz_Tot$TOTAL <- rowSums(licensesDessaz_Tot[,2:80])
licensesDessaz_Tot_TS <- ts(licensesDessaz_Tot[,-1], frequency=12, start=c(2015))
autoplot(licensesDessaz_Tot_TS[ ,c("TOTAL")])
