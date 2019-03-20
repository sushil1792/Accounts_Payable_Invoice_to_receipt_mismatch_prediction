library(lubridate)
library(Hmisc)
library(caret)
library(rpart)
library(gapminder)
library(dplyr)
library(tidyr)
library(imputeMissings)
library(plyr)
library(devtools)
library(e1071)
library(pROC)
# install_bitbucket("mkuhn/parallelRandomForest", ref="parallelRandomForest")
###########################################################################
set.seed(321)
rm(list = ls())
finvoice = read.csv( file = "Freight Invoices.csv", header = T, sep = ",", na.strings = c("",NA))

############################
#### DATA DICTIONARY #######
############################

# GRIEF_CODE:	error code if analyst has added grief B,P,C,M. B-BOL Grief, C- Communication Grief, M- MM Facility, P- MM Carrier, R- Receipt
# CARR_PRO_NO:	Invoice number. Carrier assigned so not a unique identifier
# CORP_CARR_CD:	Carrier Code- unique identifier for each freight supplier assigned by Caterpillar
# FRT_BILL_TYP: Bill Line Type: A is line haul (the actual freight charge) B…Z (accessorial charger such as handling, taxes, fuel, etc)
# FAC_CD:	Catterpillar assigned Receipt Facility Code.
# DOCK_NO:	Dock code at receipt facility. Not Unique between facilities
# FRT_IN_OUT_IND:	1-inbound, 2-outbound. Only looking at type 1 for this problem
# ENTRY_DATE:	Invoice entry date- Date the Invoice was procedssed into the freight payable system
# FRT_CLRK_CD:	Freight Clerk Code- clerk that is assigned the record if grief is created. Will correlate to Corporate Carrier code
# EDI_IND:	% is EDI (Electronic) receipt of invoice information, blank is paper receipt of invoice
# SCND_CARR_PRO_NO:	Supplier invoice number from specific section on EDI record. Can match CARR_PRO_NO
# TRNSP_MODE:	Transportation Mode: Truck, Rail, Air, Ocean
# CARR_MODE:	truckload, Less-Than-TruckLoad, etc. Likely correlated with CORP_CARR_CD
# SUPP_DLR_FAC_TYP:	type of facility the material originated
# SUPP_DLR_FAC_CD:	Caterpillar assigned facility code for supplier. Will correlate to ORIG_CITY_STATE
# ORIG_CITY_STATE:	Code for origination city/state combination
# DEST_CITY_STATE:	Code for destination city/state combination
# FRT_TRN_WT_LB:	weight of shipment. A value will only appear on the invoice lines that correspond to the freight portion of the bill. Line type A
# FRT_CHRG:	Line Item charge
# SHIP_DATE:	Original Ship date of goods

###########################################
### Data cleaning, conversion and setup ###
###########################################

### Data Summary ###
summary(finvoice)

### Using records with FRT_IN_OUT_IND = 1 {inbound} ###
finvoice = finvoice[finvoice$FRT_IN_OUT_IND==1,]
finvoice = finvoice[!is.na(finvoice$FRT_IN_OUT_IND),]

### Date Conversions ###
finvoice$ship_date = dmy(as.character(finvoice$ship_date))
# finvoice$entry_date = dmy(as.character(finvoice$entry_date))

### Choosing records with ship_date on or after 1st Jan, 2010.
finvoice_mod = data.frame(finvoice[finvoice['ship_date'] > '2010-01-01',])

### Finding and removing records with negative freight weight
finvoice_mod$FRT_TRN_WT_LB = as.numeric(finvoice_mod$FRT_TRN_WT_LB)
finvoice_mod = finvoice_mod[finvoice_mod['FRT_TRN_WT_LB']>=0,]

### Removing records with freight charge less than 0
finvoice_mod = finvoice_mod[finvoice_mod['FRT_CHRG']>=0,]

### Encoding values for EDI_IND
finvoice_mod$EDI_IND = as.character(finvoice_mod$EDI_IND)
finvoice_mod$EDI_IND[finvoice_mod$EDI_IND == '%'] = 'Y'
finvoice_mod$EDI_IND[finvoice_mod$EDI_IND != 'Y'] = 'N'
finvoice_mod$EDI_IND[is.na(finvoice_mod$EDI_IND)] = 'N'
finvoice_mod$EDI_IND = as.factor(finvoice_mod$EDI_IND)

### Encoding values for FRT_CLRK_CD
finvoice_mod$FRT_CLRK_CD = as.character(finvoice_mod$FRT_CLRK_CD)
finvoice_mod$FRT_CLRK_CD[finvoice_mod$FRT_CLRK_CD == '%'] = 'NoC'
finvoice_mod$FRT_CLRK_CD = as.factor(finvoice_mod$FRT_CLRK_CD)

### Removing unwanted data fields
finvoice_mod$CARR_PRO_NO = NULL
finvoice_mod$SCND_CARR_PRO_NO = NULL
finvoice_mod$FRT_IN_OUT_IND = NULL
finvoice_mod$entry_date = NULL
finvoice_mod$ship_date = NULL
finvoice_mod$FRT_CLRK_CD = NULL
finvoice_mod$DOCK_NO = NULL


### Coding records without and with receipt grief
finvoice_mod$Grief_Code = as.character(finvoice_mod$Grief_Code)
finvoice_mod$Grief_Code[finvoice_mod$Grief_Code=='R'] = '1'
finvoice_mod$Grief_Code[finvoice_mod$Grief_Code=='B'] = '0'
finvoice_mod$Grief_Code[finvoice_mod$Grief_Code=='C'] = '0'
finvoice_mod$Grief_Code[finvoice_mod$Grief_Code=='M'] = '0'
finvoice_mod$Grief_Code[finvoice_mod$Grief_Code=='P'] = '0'
finvoice_mod$Grief_Code[is.na(finvoice_mod$Grief_Code)] = '0'
finvoice_mod$Grief_Code = as.factor(finvoice_mod$Grief_Code)
summary(finvoice_mod)

#############################################
############# Preprocessing #################
#############################################


newinv = compute(finvoice_mod, method = "median/mode")
for (i in names(finvoice_mod)){
  finvoice_mod[is.na(finvoice_mod[,i]),i] <- newinv[i]
}
rm(newinv)
summary(finvoice_mod)

### Category Binning for predictor variables ###
tr$SUPP_DLR_FAC_CD = as.character(tr$SUPP_DLR_FAC_CD)
variables = c('CS00000','47','Y27L','X8429D0','IS','26','28','Z109','USHOU','R6')
tr[,'SUPP_DLR_FAC_CD'] = ifelse(tr[,'SUPP_DLR_FAC_CD'] %in% variables,tr[,'SUPP_DLR_FAC_CD'],"OTH")
tr$SUPP_DLR_FAC_CD = factor(tr$SUPP_DLR_FAC_CD)

finvoice_mod$SUPP_DLR_FAC_CD = as.character(finvoice_mod$SUPP_DLR_FAC_CD)
variables = c('CS00000','47','Y27L','X8429D0','IS','26','28','Z109','USHOU','R6')
finvoice_mod[,'SUPP_DLR_FAC_CD'] = ifelse(finvoice_mod[,'SUPP_DLR_FAC_CD'] %in% variables,finvoice_mod[,'SUPP_DLR_FAC_CD'],"OTH")
finvoice_mod$SUPP_DLR_FAC_CD = factor(finvoice_mod$SUPP_DLR_FAC_CD)

finvoice_mod$ORIG_CITY_STATE = as.character(finvoice_mod$ORIG_CITY_STATE)
variables = c('121950','121430','CN0020','121965','121240','120830','AU1248','421740','MX5606','BR1344')
finvoice_mod[,'ORIG_CITY_STATE'] = ifelse(finvoice_mod[,'ORIG_CITY_STATE'] %in% variables,finvoice_mod[,'ORIG_CITY_STATE'],"OTH")
finvoice_mod$ORIG_CITY_STATE = factor(finvoice_mod$ORIG_CITY_STATE)

finvoice_mod$DEST_CITY_STATE = as.character(finvoice_mod$DEST_CITY_STATE)
variables = c('343116','120830','423040','121950','120540','122320','090934','120110','121430','100070')
finvoice_mod[,'DEST_CITY_STATE'] = ifelse(finvoice_mod[,'DEST_CITY_STATE'] %in% variables,finvoice_mod[,'DEST_CITY_STATE'],"others")
finvoice_mod$DEST_CITY_STATE = factor(finvoice_mod$DEST_CITY_STATE)
finvoice_mod$CORP_CARR_CD = as.character(finvoice_mod$CORP_CARR_CD)

finvoice_mod$Grief_Code = as.numeric(finvoice_mod$Grief_Code)

finvoice_mod$CORP_CARR_CD = as.character(finvoice_mod$CORP_CARR_CD)
i = "CORP_CARR_CD"
c = count(finvoice_mod,i)
grief_count = vector()
for (t in c[,i]){
  grief_count = c(grief_count,sum(finvoice_mod[finvoice_mod[,i]==t,]$Grief_Code))
}
c = data.frame(c,grief_count)
c = c[order(-c$grief_count),]
variables=c[1:10,i]
finvoice_mod[,i]=ifelse(finvoice_mod[,i] %in% variables,finvoice_mod[,i],"OTH")

finvoice_mod$FRT_BILL_TYP = as.character(finvoice_mod$FRT_BILL_TYP)
i = "FRT_BILL_TYP"
c = count(finvoice_mod,i)
grief_count = vector()
for (t in c[,i]){
  grief_count = c(grief_count,sum(finvoice_mod[finvoice_mod[,i]==t,]$Grief_Code))
}
c = data.frame(c,grief_count)
c = c[order(-c$grief_count),]
variables=c[1:10,i]
finvoice_mod[,i]=ifelse(finvoice_mod[,i] %in% variables,finvoice_mod[,i],"OTH")

finvoice_mod$FAC_CD = as.character(finvoice_mod$FAC_CD)
i = "FAC_CD"
c = count(finvoice_mod,i)
grief_count = vector()
for (t in c[,i]){
  grief_count = c(grief_count,sum(finvoice_mod[finvoice_mod[,i]==t,]$Grief_Code))
}
c = data.frame(c,grief_count)
c = c[order(-c$grief_count),]
variables=c[1:10,i]
finvoice_mod[,i]=ifelse(finvoice_mod[,i] %in% variables,finvoice_mod[,i],"OTH")

finvoice_mod$Grief_Code = as.character(finvoice_mod$Grief_Code)
finvoice_mod$Grief_Code[finvoice_mod$Grief_Code=='2'] = 'Y'
finvoice_mod$Grief_Code[finvoice_mod$Grief_Code!='Y'] = 'N'
finvoice_mod$Grief_Code = as.factor(finvoice_mod$Grief_Code)

### Relevelling the factors ###
finvoice_mod$TRNSP_MODE = factor(finvoice_mod$TRNSP_MODE)
finvoice_mod$CARR_MODE = factor(finvoice_mod$CARR_MODE)
finvoice_mod$SUPP_DLR_FAC_TYP = factor(finvoice_mod$SUPP_DLR_FAC_TYP)
finvoice_mod$DEST_CITY_STATE = factor(finvoice_mod$DEST_CITY_STATE)
finvoice_mod$ORIG_CITY_STATE = factor(finvoice_mod$ORIG_CITY_STATE)
finvoice_mod$SUPP_DLR_FAC_CD = factor(finvoice_mod$SUPP_DLR_FAC_CD)
finvoice_mod$CORP_CARR_CD = factor(finvoice_mod$CORP_CARR_CD)
finvoice_mod$FRT_BILL_TYP = factor(finvoice_mod$FRT_BILL_TYP)
finvoice_mod$EDI_IND = factor(finvoice_mod$EDI_IND)
finvoice_mod$FAC_CD = factor(finvoice_mod$FAC_CD)

### Renaming the categories to be compatible for model use ###
levels(finvoice_mod$CORP_CARR_CD) <- make.names(levels(factor(finvoice_mod$CORP_CARR_CD)))
levels(finvoice_mod$FRT_BILL_TYP) <- make.names(levels(factor(finvoice_mod$FRT_BILL_TYP)))
levels(finvoice_mod$FAC_CD) <- make.names(levels(factor(finvoice_mod$FAC_CD)))
levels(finvoice_mod$EDI_IND) <- make.names(levels(factor(finvoice_mod$EDI_IND)))
levels(finvoice_mod$SUPP_DLR_FAC_TYP) <- make.names(levels(factor(finvoice_mod$SUPP_DLR_FAC_TYP)))
levels(finvoice_mod$SUPP_DLR_FAC_CD) <- make.names(levels(factor(finvoice_mod$SUPP_DLR_FAC_CD)))
levels(finvoice_mod$ORIG_CITY_STATE) <- make.names(levels(factor(finvoice_mod$ORIG_CITY_STATE)))
levels(finvoice_mod$DEST_CITY_STATE) <- make.names(levels(factor(finvoice_mod$DEST_CITY_STATE)))
levels(finvoice_mod$TRNSP_MODE) <- make.names(levels(factor(finvoice_mod$TRNSP_MODE)))
levels(finvoice_mod$CARR_MODE) <- make.names(levels(factor(finvoice_mod$CARR_MODE)))

### Writing cleaned data set as csv file to system ###
write.csv(finvoice_mod,file='finaldata.csv',row.names = F)
