import os, re, random
import spacy
from spacy.tokens import DocBin
from pprint import pprint

# Create a blank English NLP model
nlp = spacy.blank("en")

ENT_NAMES = {"ISIN", "TENOR", "AMOUNT", "OFFER", "BID", "VALUE_DATE", "CURRENCY"}
REGEX_SUFFIX = r"[ \?\.]{1}"
REGEX_PREFIX = r"[ ]{1}"

isinRegexPattern = REGEX_PREFIX+r"(ISIN )?[A-Z]{2}\w{10}"+REGEX_SUFFIX
tenorRegexPattern = REGEX_PREFIX+r"\d+[- ]?(year[s]?|m|M|Mon[s]?|month[s]?|Month|Y|YEAR|yr|YR|Year[s]?)"+REGEX_SUFFIX
amountRegexPattern = REGEX_PREFIX+r"(amount )?\d+[ ]?(mio[s]?|Mio[s]?|mm|MM|million[s]?|Million[s]?|k|K)"+REGEX_SUFFIX
offerRegexPattern = REGEX_PREFIX+r"^(?!yr offer[s]?|your offer[s]?)(offer[s]?|OFFER|ofr[s]?|yr bid|your bid|YOUR BID|Your Bid|quote[s]?)"+REGEX_SUFFIX # client offer
bidRegexPattern = REGEX_PREFIX+r"(bid[s]?|BID|Bid[s]?|yr offer[s]?|your offer[s]?|your ofr[s]?|yr ofr[s]?|Your Offer)"+REGEX_SUFFIX # we bid
vdRegexPattern = REGEX_PREFIX+r"((vd |value date )(on )?)?(\d{2}\/\d{2}\/\d{2,4}|\d{2}-\d{2}-\d{2,4}|today|\d+\-[A-Z]{1}[A-Za-z]{2}(\-\d{2,4})?)?((T|t)\+\d+)?"+REGEX_SUFFIX
ccyRegexPattern = REGEX_PREFIX+r"[A-Z]{3}"+REGEX_SUFFIX

# List of example sentences and annotations
DATASET:list[set] = [
    "Hi Daniel, We have offers on ISIN US1234567890, 3-year tenor, and ISIN US1234567891, 5-year tenor, both T+2 Interested, all 5 Mio? Thanks, Laura",
    "Morning Laura, I'm interested. Could you provide the price for 10Mio USD on each? Cheers, Daniel",
    "Hello Anna, Is there any bid for ISIN DE9876543210, 7-year tenor, and ISIN DE9876543211, 10-year tenor, vd today Thanks, Mark",
    "Hi Mark, We can bid at 101.75 for 5M EUR on the 7-year and 102.25 for 5M EUR on the 10-year. Best, Anna",
    "Good afternoon Robert, Could you offer on ISIN JP5678901234, 2-year tenor in USD and 5-year tenor in JPY, vd today Thanks in advance, Nancy",
    "Hi Nancy, Yes, we can offer at 99.90 for 4 MM USD on the 2-year and 100.50 for 500k JPY on the 5-year. Regards, Robert",
    "Hi David, Bid for ISIN US5678901234, 3Y USD and ISIN US5678901235, 5Y EUR, T+0 Thanks, Emma",
    "Morning Emma, Offer ISIN GB0987654321, 2 year GBP and ISIN GB0987654322, 48 Month USD, value date today. Cheers, David",
    "Hello Max, Need bid for ISIN FR1234567890, 4 MILLION EUR and ISIN FR1234567891, 12 M USD, T+0 Regards, Alex",
    "Hi Alex, Offer ISIN JP8765432109, 3 mio JPY and ISIN JP8765432110, 60 Mon USD, T+0 Thanks, Max",
    "Hiii Emily, Bid for ISIN DE4567890123, 2 year EUR and ISIN DE4567890124, 60 Mon USD, T+0 Thx, Mike",
    "Morning Mike, Offer ISIN CH0987654321, 12 M CHF and ISIN CH0987654322, 3Y EUR, value date t+1. Cheers, Emily",
    "Hiii Sarah, Need bid for ISIN IT3456789012, 4 MILLION EUR and ISIN IT3456789013, 2 YEAR USD, T+0 Regards, John",
    "Hey John, Can you offer ISIN NL6543210987, 3 mio EUR and ISIN NL6543210988, 12 M USD, T+1 Thx, Sarah",
    "Morning Mike, Ofr for GB0987654321, 12 Month GBP and GB0987654322, 36 M EUR, value date 01/01/25. Cheers, Emily",
    "Hiii Sarah, Need yr bid for IT3456789012, 4 MILLION EUR and IT3456789013, 2Y USD, vd t+1. Regards, John",
    "Hey John, Can you ofr NL6543210987, 3 mio EUR and NL6543210988, 12 Month USD, value date Mar 10. Thx, Sarah",
    "Hi Chris, Pls bid SE9876543210, 2Y SEK and SE9876543211, 4Y USD, T+1 Thx, Olivia",
    "Hiii Olivia, We need yr ofr NO1234567890, 12 Month NOK and NO1234567891, 60 Month USD, value date t+0, t+1, Chris",
    "Hello Jane, Need quotes for JP9876543210, 3 mm JPY and JP9876543211, 5 Year USD, value date 07-24-24. Thx, David",
    "Hi David, Can you bid for CA5678901234, 1 year CAD and CA5678901235, 36 M USD, value date 01-Jan-25? Regards, Jane",
    "Hey Mike, Yr quotes for AU9876543210, 4 month AUD and AU9876543211, 24 Mon USD, value date 10/10/24. Cheers, Laura",
    "Hi Laura, Ofr for CN1234567890, 1 year CNY and CN1234567891, 4Y USD, value date 24-07-24. Thx, Mike",
    "Hiii Emily, Yr bid for US5678901234, 24 Mon EUR, amount 5M and US5678901235, 5 year USD, amount 10Mio, vd 12-07-24. Thx, Mike",
    "Morning Mike, Ofr for GB0987654321, 1 year GBP, amount 8 MM and GB0987654322, 36 Month EUR, amount 12M, vd 01/01/25. Cheers, Emily",
    "Hiii Sarah, Need yr bid for IT3456789012, 4 month EUR, amount 3 mm and IT3456789013, 24 Mon USD, amount 4MM, value on 15th Feb. Regards, John",
    "Hey John, Can you ofr NL6543210987, 3 mm EUR, amount 4 MM and NL6543210988, 1 year USD, amount 9 Millions, value date Mar 10. Thx, Sarah",
    "Hi Chris, Pls bid SE9876543210, 24 Mon SEK, amount 4 mil and SE9876543211, 4Y USD, amount 10MM, vd Jul 5th. Thx, Olivia",
    "Hiii Olivia, We need yr ofr NO1234567890, 1 YEAR NOK, amount 2MM and NO1234567891, 5Y USD, amount 8 Mio, vd 25-Dec. Best, Chris",
    "Hello Jane, Need quotes for JP9876543210, 3 month JPY, amount 5M and JP9876543211, 5Y USD, amount 15M, value on 07-24-24. Thx, David",
    "Hi David, Can you bid for CA5678901234, 1 YEAR CAD, amount 7MM and CA5678901235, 36 Month USD, amount 14 mio, vd 01-Jan-25? Regards, Jane",
    "Hi Mike, I need an offer for US3456789012, 2 Year USD, amount 10M and US3456789013, 10YR EUR, amount 500K, value date 25-Jul-24. Thanks, Lisa",
    "Hello Sara, Please quote for DE9876543210, 7 month EUR, amount 3Mio and DE9876543211, 160 month USD, amount 20 Mils, value date 24-Jul-24. Best, John",
    "Good afternoon Emily, Looking for bids on FR4567890123, 60 month EUR, amount 12MM and FR4567890124, 20YR USD, amount 7 mio, vd 25-Jul-24. Regards, Tom",
    "Hi Alex, Can you offer for IT6789012345, 3 YR EUR, amount 6 mio and IT6789012346, 30YR USD, amount 9MM, value date 24-Jul-24? Cheers, Natalie",
    "Hello Kate, Need quotes for ES2345678901, 1 year EUR, amount 8Mio and ES2345678902, 10YR USD, amount 4MM, value date 25-Jul-24. Thanks, Mark",
     "Hiii Olivia, We need yr ofr NO1234567890, 1 YEAR NOK, amount 2MM and NO1234567891, 5Y USD, amount 8 Mio, vd 25-Dec. Best, Chris",
    "Hello Jane, Need quotes for JP9876543210, 3 month JPY, amount 5M and JP9876543211, 5Y USD, amount 15M, value on 07-24-24. Thx, David",
    "Hi David, Can you bid for CA5678901234, 1 YEAR CAD, amount 7MM and CA5678901235, 36 Month USD, amount 14 mio, vd 01-Jan-25? Regards, Jane",
    "Hi Mike, I need an offer for US3456789012, 2 Year USD, amount 10MM and US3456789013, 10YR EUR, amount 500K, value date 25-Jul-24. Thanks, Lisa",
    "Hello Sara, Please quote for DE9876543210, 7 month EUR, amount 250 K and DE9876543211, 160 month USD, amount 20MM, value date 24-Jul-24. Best, John",
    "Good afternoon Emily, Looking for bids on FR4567890123, 60 month EUR, amount 12MM and FR4567890124, 20YR USD, amount 7 mio, vd 25-Jul-24. Regards, Tom",
    "Hi Alex, Can you offer for IT6789012345, 3YR EUR, amount 6 mio and IT6789012346, 30YR USD, amount 9MM, value date 24-Jul-24? Cheers, Natalie",
    "Hello Kate, Need quotes for ES2345678901, 1 year EUR, amount 8MM and ES2345678902, 10YR USD, amount 4MM, value date 25-Jul-24. Thanks, Mark",
    "Hi Tom, Require offer for AU1234567890, 5Y AUD, amount 2MM and AU1234567891, 3Y USD, amount 9 mio, vd 20-Aug. Best, Alex",
    "Hello Mike, Need bids for CH9876543210, 2 mils CHF, amount 1MM and CH9876543211, 4YR USD, amount 6 mio, value date 15-Sep-24. Thx, Sarah",
    "Good morning, Please quote for GB5678901234, 1Y GBP, amount 250 K and GB5678901235, 7YR USD, amount 12 mils, value on 30-Jul-24. Regards, Emma",
    "Hi Jessica, Can you offer for IT2345678901, 2Y EUR, amount 5MM and IT2345678902, 10YR USD, amount 10 mio, vd 01-Aug-24? Cheers, Michael",
    "Hello John, Need quotes for JP3456789012, 3 mio JPY, amount 4MM and JP3456789013, 60 month USD, amount 20MM, value date 15-Aug-24. Thanks, Laura",
    "Hi Emily, Please provide offer for DE4567890123, 7YR EUR, amount 8MM and DE4567890124, 12 Year USD, amount 5 mio, value on 25-Jul-24. Best, Alex",
    "Hello Mark, Looking for bids on FR5678901234, 1 year EUR, amount 450K and FR5678901235, 8YR USD, amount 15 mils, vd 24-Jul-24. Regards, Natalie",
    "Hi Chris, Need offer for AU6789012345, 3Y AUD, amount 7MM and AU6789012346, 60 month USD, amount 9 mio, value date 30-Jul-24. Thx, Olivia",
    "Hello Sarah, Require bids for CA7890123456, 1 y CAD, amount 400K and CA7890123457, 10YR USD, amount 20 mils, value date 01-Aug-24. Regards, Tom",
    "Hiii Olivia, We need yr ofr NO1234567890, 1 YEAR NOK, amount 2MM and NO1234567891, 5Y USD, amount 8 Mio, vd 25-Dec. Best, Chris",
    "Hello Jane, Need quotes for JP9876543210, 3 mio JPY, amount 5M and JP9876543211, 5Y USD, amount 15M, value on 07-24-24. Thx, David",
    "Hi David, Can you bid for CA5678901234, 1 YEAR CAD, amount 7MM and CA5678901235, 36 Month USD, amount 14 mio, vd 01-Jan-25? Regards, Jane",
    "Hi Mike, I need an offer for US3456789012, 2 Year USD, amount 10MM and US3456789013, 10YR EUR, amount 5MM, value date 25-Jul-24. Thanks, Lisa",
    "Hello Sara, Please quote for DE9876543210, 7 mio EUR, amount 250 K and DE9876543211, 160 month USD, amount 20MM, value date 24-Jul-24. Best, John",
    "Good afternoon Emily, Looking for bids on FR4567890123, 60 month EUR, amount 12MM and FR4567890124, 20YR USD, amount 7 mio, vd 25-Jul-24. Regards, Tom",
    "Hi Alex, Can you offer for IT6789012345, 3YR EUR, amount 6 mio and IT6789012346, 30YR USD, amount 9MM, value date 24-Jul-24? Cheers, Natalie",
    "Hello Kate, Need quotes for ES2345678901, 1 y EUR, amount 8MM and ES2345678902, 10YR USD, amount 400K, value date 25-Jul-24. Thanks, Mark",
    "Hi Tom, Require offer for AU1234567890, 5Y AUD, amount 2MM and AU1234567891, 3Y USD, amount 9 mio, vd 20-Aug. Best, Alex",
    "Hello Mike, Need bids for CH9876543210, 2 mils CHF, amount 1MM and CH9876543211, 4YR USD, amount 6 mio, value date 15-Sep-24. Thx, Sarah",
    "Good morning, Please quote for GB5678901234, 1Y GBP, amount 250 K and GB5678901235, 7YR USD, amount 12 mils, value on 30-Jul-24. Regards, Emma",
    "Hi Jessica, Can you offer for IT2345678901, 2Y EUR, amount 5MM and IT2345678902, 10YR USD, amount 10 mio, vd 01-Aug-24? Cheers, Michael",
    "Hello John, Need quotes for JP3456789012, 3 mio JPY, amount 400K and JP3456789013, 60 month USD, amount 20MM, value date 15-Aug-24. Thanks, Laura",
    "Hi Emily, Please provide offer for DE4567890123, 7YR EUR, amount 8MM and DE4567890124, 12 Year USD, amount 5 mio, value on 25-Jul-24. Best, Alex",
    "Hello Mark, Looking for bids on FR5678901234, 1 y EUR, amount 450K and FR5678901235, 8YR USD, amount 15 mils, vd 24-Jul-24. Regards, Natalie",
    "Hi Chris, Need offer for AU6789012345, 3Y AUD, amount 7MM and AU6789012346, 60 month USD, amount 9 mio, value date 30-Jul-24. Thx, Olivia",
    "Hello Sarah, Require bids for CA7890123456, 1 y CAD, amount 400K and CA7890123457, 10YR USD, amount 20 mils, value date 01-Aug-24. Regards, Tom",
    "Hi Alex, Looking for quotes on ES8901234567, 2 Year EUR, amount 5MM and ES8901234568, 160 month USD, amount 10 mio, vd 15-Aug-24. Cheers, Jessica",
    "Hello Michael, Need offer for GB9012345678, 3Y GBP, amount 450K and GB9012345679, 20YR USD, amount 8 mio, value date 25-Jul-24. Thanks, Emma",
    "Hi Laura, Can you bid for JP0123456789, 1 y JPY, amount 2MM and JP0123456790, 5 y USD, amount 12 mils, vd 24-Jul-24? Best, John",
    "Hi Rachel, Need offer for SE1234567890, 1Y SEK, amount 3 mio and SE1234567891, 10YR USD, amount 5MM, value date 25-Dec. Thanks, Chris",
    "Hello Eric, Require bids on DK9876543210, 5Y DKK, amount 4 mio and DK9876543211, 7YR USD, amount 8MM, value on 07-25-24. Regards, Sarah",
    "Hi Kevin, Please quote for FI5678901234, 2Y EUR, amount 450K and FI5678901235, 12 Year USD, amount 10 mio, value date 01-Jan-25. Thx, David",
    "Hello Olivia, Need quotes for SG3456789012, 3 mio SGD, amount 7MM and SG3456789013, 5 y\ USD, amount 15 mils, vd 24-Jul-24. Best, Michael",
    "Hi Sarah, Require offer for KR1234567890, 1Y KRW, amount 2 mio and KR1234567891, 8YR USD, amount 12MM, value on 25-Jul-24. Regards, John",
    "Hello Chris, Need quotes on RU2345678901, 3Y RUB, amount 450K and RU2345678902, 20YR USD, amount 15 mils, value on 25-Jul-24. Best, Eric",
    "Hi David, Please offer for AR3456789012, 5Y ARS, amount 250 K and AR3456789013, 10YR USD, amount 5 mio, vd 15-Aug-24. Thx, Lisa",
    "Hello Sarah, Require quotes for MX4567890123, 1 y MXN, amount 7MM and MX4567890124, 12 Year USD, amount 10 mils, value date 30-Jul-24. Regards, Natalie",
    "Hi Michael, Need bids for TR5678901234, 2Y TRY, amount 400K and TR5678901235, 8YR USD, amount 12 mio, vd 01-Aug-24. Best, Mark",
    "Good afternoon Laura, Looking for quotes on CL6789012345, 3YR CLP, amount 5MM and CL6789012346, 15 y USD, amount 20 mils, value date 24-Jul-24. Thanks, Emma",
     "Hiii Olivia, We need yr ofr NO1234567890, 1 YEAR NOK, amount 200 K and NO1234567891, 5Y USD, amount 8 Mio, vd 25-Dec. Best, Chris",
    "Hello Jane, Need quotes for JP9876543210, 3 mio JPY, amount 5M and JP9876543211, 5Y USD, amount 15M, value on 07-24-24. Thx, David",
    "Hi David, Can you bid for CA5678901234, 1 YEAR CAD, amount 7MM and CA5678901235, 36 Month USD, amount 14 mio, vd 01-Jan-25? Regards, Jane",
    "Hi Mike, I need an offer for US3456789012, 2 year USD, amount 10MM and US3456789013, 10YR EUR, amount 5MM, value date 25-Jul-24. Thanks, Lisa",
    "Hello Sara, Please quote for DE9876543210, 7 mio EUR, amount 250 K and DE9876543211, 15 y USD, amount 20MM, value date 24-Jul-24. Best, John",
    "Good afternoon Emily, Looking for bids on FR4567890123, 5 y EUR, amount 1200 K and FR4567890124, 20YR USD, amount 7 mio, vd 25-Jul-24. Regards, Tom",
    "Hi Alex, Can you offer for IT6789012345, 3YR EUR, amount 6 mio and IT6789012346, 30YR USD, amount 9MM, value date 24-Jul-24? Cheers, Natalie",
    "Hello Kate, Need quotes for ES2345678901, 1 y EUR, amount 8MM and ES2345678902, 10YR USD, amount 400 K, value date 25-Jul-24. Thanks, Mark",
    "Hi Tom, Require offer for AU1234567890, 5Y AUD, amount 200 K and AU1234567891, 3Y USD, amount 9 mio, vd 20-Aug. Best, Alex",
    "Hello Mike, Need bids for CH9876543210, 2 Mon CHF, amount 1MM and CH9876543211, 4YR USD, amount 6 mio, value date 15-Sep-24. Thx, Sarah",
    "Good morning, Please quote for GB5678901234, 1Y GBP, amount 350 K and GB5678901235, 7YR USD, amount 12 mils, value on 30-Jul-24. Regards, Emma",
    "Hi Jessica, Can you offer for IT2345678901, 2Y EUR, amount 5MM and IT2345678902, 10YR USD, amount 10 mio, vd 01-Aug-24? Cheers, Michael",
    "Hello John, Need quotes for JP3456789012, 3 mio JPY, amount 400 K and JP3456789013, 5 y USD, amount 20MM, value date 15-Aug-24. Thanks, Laura",
    "Hi Emily, Please provide offer for DE4567890123, 7YR EUR, amount 8MM and DE4567890124, 12 year USD, amount 5 mio, value on 25-Jul-24. Best, Alex",
    "Hello Mark, Looking for bids on FR5678901234, 1 y EUR, amount 6MM and FR5678901235, 8YR USD, amount 15 mils, vd 24-Jul-24. Regards, Natalie",
    "Hi Chris, Need offer for AU6789012345, 3Y AUD, amount 7MM and AU6789012346, 5 y USD, amount 9 mio, value date 30-Jul-24. Thx, Olivia",
    "Hello Sarah, Require bids for CA7890123456, 1 y CAD, amount 400 K and CA7890123457, 10YR USD, amount 20 mils, value date 01-Aug-24. Regards, Tom",
    "Hi Alex, Looking for quotes on ES8901234567, 2 year EUR, amount 5MM and ES8901234568, 15 y USD, amount 10 mio, vd 15-Aug-24. Cheers, Jessica",
    "Hello Michael, Need offer for GB9012345678, 3Y GBP, amount 6MM and GB9012345679, 20YR USD, amount 8 mio, value date 25-Jul-24. Thanks, Emma",
    "Hi Laura, Can you bid for JP0123456789, 1 y JPY, amount 200 K and JP0123456790, 5y USD, amount 12 mils, vd 24-Jul-24? Best, John",
    "Hi Rachel, Need offer for SE1234567890, 1Y SEK, amount 3 mio and SE1234567891, 10YR USD, amount 5MM, value date 25-Dec. Thanks, Chris",
    "Hello Eric, Require bids on DK9876543210, 5Y DKK, amount 4 mio and DK9876543211, 7YR USD, amount 8MM, value on 07-25-24. Regards, Sarah",
    "Hi Kevin, Please quote for FI5678901234, 2Y EUR, amount 6MM and FI5678901235, 12 y USD, amount 10 mio, value date 01-Jan-25. Thx, David",
    "Hello Olivia, Need quotes for SG3456789012, 3 mio SGD, amount 7MM and SG3456789013, 5y USD, amount 15 mils, vd 24-Jul-24. Best, Michael",
    "Hi Sarah, Require offer for KR1234567890, 1Y KRW, amount 2 mio and KR1234567891, 8YR USD, amount 1200 K, value on 25-Jul-24. Regards, John",
    "Good morning Lisa, Looking for bids on HK6789012345, 3YR HKD, amount 5MM and HK6789012346, 10YR USD, amount 20 mio, value date 15-Aug-24. Thanks, Emily",
    "Hi Natalie, Need offer for IN7890123456, 2Y INR, amount 350 K and IN7890123457, 15y USD, amount 10 mils, vd 30-Jul-24. Best, Mark",
    "Hello David, Please quote for ZA0123456789, 1Y ZAR, amount 400 K and ZA0123456790, 5y USD, amount 12 mio, value date 01-Aug-24. Thx, Rachel",
    "Hi Olivia, Require bids for BR1234567890, 2 y BRL, amount 5MM and BR1234567891, 7YR USD, amount 8 mio, vd 24-Jul-24. Regards, Kevin",
    "Hello Chris, Need quotes on RU2345678901, 3Y RUB, amount 6MM and RU2345678902, 20YR USD, amount 15 mils, value on 25-Jul-24. Best, Eric",
    "Hi David, Please offer for AR3456789012, 5Y ARS, amount 350 K and AR3456789013, 10YR USD, amount 5 mio, vd 15-Aug-24. Thx, Lisa",
    "Hello Sarah, Require quotes for MX4567890123, 1 y MXN, amount 7MM and MX4567890124, 12 y USD, amount 10 mils, value date 30-Jul-24. Regards, Natalie",
    "Hi Michael, Need bids for TR5678901234, 2Y TRY, amount 400 K and TR5678901235, 8YR USD, amount 12 mio, vd 01-Aug-24. Best, Mark",
    "Good afternoon Laura, Looking for quotes on CL6789012345, 3YR CLP, amount 5MM and CL6789012346, 15y USD, amount 20 mils, value date 24-Jul-24. Thanks, Emma",
    "Hi Eric, Require offer for PE7890123456, 1Y PEN, amount 6MM and PE7890123457, 10YR USD, amount 10 mio, value date 01-Jan-25. Regards, Jessica",
    "Hello Alex, Please quote for NZ0123456789, 3YR NZD, amount 200 k and NZ0123456790, 5YR USD, amount 4 mio, vd 07-24-24. Best, Mike",
    "Hi Olivia, Need bids on MY1234567890, 1 y MYR, amount 1 mio and MY1234567891, 15YR USD, amount 8MM, value date 24-Jul-24. Thx, John",
    "Hello Mark, Require offer for TH3456789012, 2Y THB, amount 7MM and TH3456789013, 7YR USD, amount 12 mio, value on 25-Jul-24. Regards, Laura",
    "Hi Sarah, Please quote for VN4567890123, 3 y VND, amount 400 thousands and VN4567890124, 20YR USD, amount 10 mils, vd 15-Aug-24. Thanks, Kevin",
    "Hello Jessica, Looking for bids on PH5678901234, 1YR PHP, amount 5MM and PH5678901235, 8YR USD, amount 6 mio, value date 01-Aug-24. Best, Chris",
    "Hi Mike, Need quotes for ID6789012345, 2Y IDR, amount 350 K and ID6789012346, 10YR USD, amount 15 mils, value date 30-Jul-24. Thx, Alex",
    "Hello Laura, Require offer on PK7890123456, 3 y PKR, amount 6MM and PK7890123457, 5YR USD, amount 8 mio, vd 24-Jul-24. Regards, David",
    "Hi Eric, Please quote for BD0123456789, 1Y BDT, amount 400 thousands and BD0123456790, 7YR USD, amount 10 mils, value on 25-Jul-24. Thanks, Sarah",
    "Hello Jessica, Need bids for EG1234567890, 2 y EGP, amount 7MM and EG1234567891, 15YR USD, amount 12 mio, value date 15-Aug-24. Best, John",
    "Hi Michael, Looking for quotes on NG3456789012, 3Y NGN, amount 5MM and NG3456789013, 20YR USD, amount 10 mils, vd 01-Aug-24. Thx, Lisa",
    "Hello John, Please offer for KE4567890123, 1Y KES, amount 200 k and KE4567890124, 10YR USD, amount 6 mio, value date 30-Jul-24. Regards, Emma",
    "Hi David, Require quotes for TZ5678901234, 2Y TZS, amount 350 K and TZ5678901235, 8YR USD, amount 12 mils, value on 24-Jul-24. Thanks, Olivia",
    "Hello Olivia, Need bids on UG6789012345, 3 y UGX, amount 6MM and UG6789012346, 5YR USD, amount 8 mio, vd 25-Jul-24. Best, Michael",
    "Hi Sarah, Please quote for ZM7890123456, 1Y ZMW, amount 400 thousands and ZM7890123457, 7YR USD, amount 10 mils, value date 15-Aug-24. Regards, Alex",
    "Hello John, Require offer on ZW0123456789, 2 y ZWL, amount 7MM and ZW0123456790, 15YR USD, amount 12 mio, value on 01-Aug-24. Thx, Rachel",
    "Hi Kevin, Looking for quotes on BW1234567890, 3Y BWP, amount 5MM and BW1234567891, 20YR USD, amount 10 mils, value date 30-Jul-24. Regards, Tom",
    "Hello Michael, Need bids for MW3456789012, 1Y MWK, amount 200 k and MW3456789013, 10YR USD, amount 6 mio, vd 24-Jul-24. Best, Emma",
    "Hi Lisa, Please offer on LS4567890123, 2Y LSL, amount 350 k and LS4567890124, 8YR USD, amount 12 mils, value date 25-Jul-24. Thx, David",
    "Hello Sarah, Require quotes for SZ5678901234, 3 y SZL, amount 6MM and SZ5678901235, 5YR USD, amount 8 mio, value on 15-Aug-24. Regards, Olivia",
    "Hi Chris, Need bids on MG6789012345, 1YR MGA, amount 400 k and MG6789012346, 7YR USD, amount 10 mils, vd 01-Aug-24. Thanks, Alex",
    "Hello Emma, Please quote for MU7890123456, 2Y MUR, amount 7MM and MU7890123457, 15YR USD, amount 12 mio, value date 30-Jul-24. Best, Michael",
    "Hi Tom, Require offer on NA0123456789, 3Y NAD, amount 5MM and NA0123456790, 20YR USD, amount 10 mils, value on 24-Jul-24. Thx, Sarah",
    "Hello Olivia, Need bids for RW1234567890, 1Y RWF, amount 200 k and RW1234567891, 10YR USD, amount 6 mio, vd 25-Jul-24. Regards, John",
    "Hi Kevin, Please quote on ST3456789012, 2Y STD, amount 350 k and ST3456789013, 8YR USD, amount 12 mils, value date 15-Aug-24. Best, David",
    "Hello Michael, Require offer for SZ4567890123, 3 year SZL, amount 6MM and SZ4567890124, 5YR USD, amount 8 mio, vd 01-Aug-24. Thx, Lisa",
    "Hi Sarah, Looking for quotes on MG5678901234, 1YR MGA, amount 400 k and MG5678901235, 7YR USD, amount 10 mils, value on 30-Jul-24. Thanks, Mark",
    "Hello John, Need bids for MU6789012345, 2Y MUR, amount 7MM and MU6789012346, 15YR USD, amount 12 mio, value date 24-Jul-24. Best, Emma",
    "Hi Rachel, Please offer on NA7890123456, 3Y NAD, amount 5MM and NA7890123457, 20YR USD, amount 10 mils, vd 25-Jul-24. Regards, Michael",
    "Hello Alex, Require quotes for RW0123456789, 1Y RWF, amount 200 k and RW0123456790, 10YR USD, amount 6 mio, value on 15-Aug-24. Thx, Olivia",
    "Hi Tom, Need bids on ST1234567890, 2Y STD, amount 350 k and ST1234567891, 8YR USD, amount 12 mils, value date 01-Aug-24. Regards, Kevin",
    "Hi Laura, Can you bid for JP0123456789, 1YR JPY, amount 200k and JP0123456790, 5YR USD, amount 12 mils, vd 24-Jul-24? Best, John",
    "Good morning Lisa, Looking for bids on HK6789012345, 3 year HKD, amount 5MM and HK6789012346, 10YR USD, amount 20 mio, value date 15-Aug-24. Thanks, Emily",
    "Hi Natalie, Need offer for IN7890123456, 2Y INR, amount 350 k and IN7890123457, 15YR USD, amount 10 mils, vd 30-Jul-24. Best, Mark",
    "Hello David, Please quote for ZA0123456789, 1Y ZAR, amount 400 k and ZA0123456790, 5YR USD, amount 12 mio, value date 01-Aug-24. Thx, Rachel",
    "Hi Olivia, Require bids for BR1234567890, 2 y BRL, amount 5MM and BR1234567891, 7YR USD, amount 8 mio, vd 24-Jul-24. Regards, Kevin",
    "Hi Olivia, We need your offer for NO1234567890, 1 YEAR NOK, amount 200k and NO1234567891, 5Y USD, amount 8 Mio, vd 25-Dec. Best, Chris",
    "Hello Jane, Need quotes for JP9876543210, 3 mio JPY, amount 5MM and JP9876543211, 5Y USD, amount 15MM, value on 07-24-24. Thx, David",
    "Hi David, Can you bid for CA5678901234, 1 YEAR CAD, amount 7MM and CA5678901235, 36 Month USD, amount 14 mio, vd 01-Jan-25? Regards, Jane",
    "Hi Mike, I need an offer for US3456789012, 2 y USD, amount 10MM and US3456789013, 10YR EUR, amount 5MM, value date 25-Jul-24. Thanks, Lisa",
    "Hello Sara, Please quote for DE9876543210, 7 mio EUR, amount 3MM and DE9876543211, 15YR USD, amount 20MM, value date 24-Jul-24. Best, John",
    "Good afternoon Emily, Looking for bids on FR4567890123, 5YR EUR, amount 1200K and FR4567890124, 20YR USD, amount 7 mio, vd 25-Jul-24. Regards, Tom",
    "Hi Alex, Can you offer for IT6789012345, 3 year EUR, amount 6 mio and IT6789012346, 30YR USD, amount 9MM, value date 24-Jul-24? Cheers, Natalie",
    "Hello Kate, Need quotes for ES2345678901, 1YR EUR, amount 8MM and ES2345678902, 10YR USD, amount 400 k, value date 25-Jul-24. Thanks, Mark",
    "Hi Tom, Require offer for AU1234567890, 5Y AUD, amount 200K and AU1234567891, 3Y USD, amount 9 mio, vd 20-Aug. Best, Alex",
    "Hello Mike, Need bids for CH9876543210, 2 Mon CHF, amount 500 k and CH9876543211, 4YR USD, amount 6 mio, value date 15-Sep-24. Thx, Sarah",
    "Good morning, Please quote for GB5678901234, 1Y GBP, amount 3MM and GB5678901235, 7YR USD, amount 12 mils, value on 30-Jul-24. Regards, Emma",
    "Hi Jessica, Can you offer for IT2345678901, 2Y EUR, amount 500K and IT2345678902, 10YR USD, amount 10 mio, vd 01-Aug-24? Cheers, Michael",
    "Hello John, Need quotes for JP3456789012, 3 mio JPY, amount 400 k and JP3456789013, 5 year USD, amount 20MM, value date 15-Aug-24. Thanks, Laura",
    "Hi Emily, Please provide offer for DE4567890123, 7YR EUR, amount 8MM and DE4567890124, 12 yr USD, amount 5 mio, value on 25-Jul-24. Best, Alex",
    "Hello Mark, Looking for bids on FR5678901234, 1YR EUR, amount 6MM and FR5678901235, 8YR USD, amount 15 mils, vd 24-Jul-24. Regards, Natalie",
    "Hi Chris, Need offer for AU6789012345, 3Y AUD, amount 7MM and AU6789012346, 5 year USD, amount 9 mio, value date 30-Jul-24. Thx, Olivia",
    "Hello Sarah, Require bids for CA7890123456, 1YR CAD, amount 4MM and CA7890123457, 10YR USD, amount 20 mils, value date 01-Aug-24. Regards, Tom",
    "Hi Alex, Looking for quotes on ES8901234567, 2 yr EUR, amount 500K and ES8901234568, 15 year USD, amount 10 mio, vd 15-Aug-24. Cheers, Jessica",
    "Hello Michael, Need offer for GB9012345678, 3Y GBP, amount 6MM and GB9012345679, 20YR USD, amount 8 mio, value date 25-Jul-24. Thanks, Emma",
    "Hi Laura, Can you bid for JP0123456789, 1YR JPY, amount 200K and JP0123456790, 5 year USD, amount 12 mils, vd 24-Jul-24? Best, John",
    "Hi Rachel, Need offer for SE1234567890, 1Y SEK, amount 3 mio and SE1234567891, 10YR USD, amount 500K, value date 25-Dec. Thanks, Chris",
    "Hello Eric, Require bids on DK9876543210, 5Y DKK, amount 4 mio and DK9876543211, 7YR USD, amount 8MM, value on 07-25-24. Regards, Sarah",
    "Hi Kevin, Please quote for FI5678901234, 2Y EUR, amount 450 k and FI5678901235, 12 yr USD, amount 10 mio, value date 01-Jan-25. Thx, David",
    "Hello Olivia, Need quotes for SG3456789012, 3 mio SGD, amount 7MM and SG3456789013, 5 year USD, amount 15 mils, vd 24-Jul-24. Best, Michael",
    "Hi Sarah, Require offer for KR1234567890, 1Y KRW, amount 2 mio and KR1234567891, 8YR USD, amount 1200K, value on 25-Jul-24. Regards, John",
    "Good morning Lisa, Looking for bids on HK6789012345, 3 year HKD, amount 500K and HK6789012346, 10YR USD, amount 20 mio, value date 15-Aug-24. Thanks, Emily",
    "Hi Natalie, Need offer for IN7890123456, 2Y INR, amount 3MM and IN7890123457, 15 year USD, amount 10 mils, vd 30-Jul-24. Best, Mark",
    "Hello David, Please quote for ZA0123456789, 1Y ZAR, amount 4MM and ZA0123456790, 5 year USD, amount 12 mio, value date 01-Aug-24. Thx, Rachel",
    "Hi Olivia, Require bids for BR1234567890, 2 yr BRL, amount 500K and BR1234567891, 7YR USD, amount 8 mio, vd 24-Jul-24. Regards, Kevin",
    "Hello Chris, Need quotes on RU2345678901, 3Y RUB, amount 450 k and RU2345678902, 20YR USD, amount 15 mils, value on 25-Jul-24. Best, Eric",
    "Hi David, Please offer for AR3456789012, 5Y ARS, amount 3MM and AR3456789013, 10YR USD, amount 5 mio, vd 15-Aug-24. Thx, Lisa",
    "Hello Sarah, Require quotes for MX4567890123, 12 Month MXN, amount 7MM and MX4567890124, 12YR USD, amount 10 mils, value date 30-Jul-24. Regards, Natalie",
    "Hi Michael, Need bids for TR5678901234, 2Y TRY, amount 4MM and TR5678901235, 8YR USD, amount 12 mio, vd 01-Aug-24. Best, Mark",
    "Good afternoon Laura, Looking for quotes on CL6789012345, 3 year CLP, amount 500K and CL6789012346, 15 year USD, amount 20 mils, value date 24-Jul-24. Thanks, Emma",
    "Hi Eric, Require offer for PE7890123456, 1Y PEN, amount 450 k and PE7890123457, 10YR USD, amount 10 mio, value date 01-Jan-25. Regards, Jessica",
    "Hello Alex, Please quote for NZ0123456789, 3 Year NZD, amount 200K and NZ0123456790, 5 year USD, amount 4 mio, vd 07-24-24. Best, Mike",
    "Hi Olivia, Need bids on MY1234567890, 12 Month MYR, amount 1 mio and MY1234567891, 15 year USD, amount 8MM, value date 24-Jul-24. Thx, John",
    "Hello Mark, Require offer for TH3456789012, 2Y THB, amount 7MM and TH3456789013, 7YR USD, amount 12 mio, value on 25-Jul-24. Regards, Laura",
    "Hi Sarah, Please quote for VN4567890123, 3 Year VND, amount 4MM and VN4567890124, 20YR USD, amount 10 mils, vd 15-Aug-24. Thanks, Kevin",
    "Hello Jessica, Looking for bids on PH5678901234, 12 Month PHP, amount 500K and PH5678901235, 8YR USD, amount 6 mio, value date 01-Aug-24. Best, Chris",
    "Hi Mike, Need quotes for ID6789012345, 2Y IDR, amount 3MM and ID6789012346, 10YR USD, amount 15 mils, value date 30-Jul-24. Thx, Alex",
    "Hello Laura, Require offer on PK7890123456, 3 Year PKR, amount 450 k and PK7890123457, 60 month USD, amount 8 mio, vd 24-Jul-24. Regards, David",
    "Hi Eric, Please quote for BD0123456789, 1Y BDT, amount 4MM and BD0123456790, 7YR USD, amount 10 mils, value on 25-Jul-24. Thanks, Sarah",
    "Hello Jessica, Need bids for EG1234567890, 2YR EGP, amount 7MM and EG1234567891, 160 month USD, amount 12 mio, value date 15-Aug-24. Best, John",
    "Hi Michael, Looking for quotes on NG3456789012, 3Y NGN, amount 500K and NG3456789013, 20YR USD, amount 10 mils, vd 01-Aug-24. Thx, Lisa",
    "Hello John, Please offer for KE4567890123, 1Y KES, amount 200K and KE4567890124, 10YR USD, amount 6 mio, value date 30-Jul-24. Regards, Emma",
    "Hi David, Require quotes for TZ5678901234, 2Y TZS, amount 3MM and TZ5678901235, 8YR USD, amount 12 mils, value on 24-Jul-24. Thanks, Olivia",
    "Hello Olivia, Need bids on UG6789012345, 3 Year UGX, amount 450 K and UG6789012346, 60 month USD, amount 8 mio, vd 25-Jul-24. Best, Michael",
    "Hi Sarah, Please quote for ZM7890123456, 1Y ZMW, amount 4MM and ZM7890123457, 7YR USD, amount 10 mils, value date 15-Aug-24. Regards, Alex",
    "Hello John, Require offer on ZW0123456789, 2YR ZWL, amount 7MM and ZW0123456790, 160 month USD, amount 12 mio, value on 01-Aug-24. Thx, Rachel",
    "Hi Kevin, Looking for quotes on BW1234567890, 3Y BWP, amount 500K and BW1234567891, 20YR USD, amount 10 mils, value date 30-Jul-24. Regards, Tom",
    "Hello Michael, Need bids for MW3456789012, 1Y MWK, amount 200K and MW3456789013, 10YR USD, amount 6 mio, vd 24-Jul-24. Best, Emma",
    "Hi Lisa, Please offer on LS4567890123, 2Y LSL, amount 250K and LS4567890124, 8YR USD, amount 12 mils, value date 25-Jul-24. Thx, David",
    "Hello Sarah, Require quotes for SZ5678901234, 3 Year SZL, amount 450 K and SZ5678901235, 60 month USD, amount 8 mio, value on 15-Aug-24. Regards, Olivia",
    "Hi Chris, Need bids on MG6789012345, 12 Month MGA, amount 4MM and MG6789012346, 7YR USD, amount 10 mils, vd 01-Aug-24. Thanks, Alex",
    "Hello Emma, Please quote for MU7890123456, 2Y MUR, amount 7MM and MU7890123457, 160 month USD, amount 12 mio, value date 30-Jul-24. Best, Michael",
    "Hi Tom, Require offer on NA0123456789, 3Y NAD, amount 500K and NA0123456790, 20YR USD, amount 10 mils, value on 24-Jul-24. Thx, Sarah",
    "Hello Olivia, Need bids for RW1234567890, 1Y RWF, amount 2MM and RW1234567891, 10YR USD, amount 6 mio, vd 25-Jul-24. Regards, John",
    "Hi Kevin, Please quote on ST3456789012, 2Y STD, amount 250K and ST3456789013, 8YR USD, amount 12 mils, value date 15-Aug-24. Best, David",
    "Hello Michael, Require offer for SZ4567890123, 3 Year SZL, amount 450 K and SZ4567890124, 60 month USD, amount 8 mio, vd 01-Aug-24. Thx, Lisa",
    "Hi Sarah, Looking for quotes on MG5678901234, 1 year MGA, amount 4MM and MG5678901235, 7YR USD, amount 10 mils, value on 30-Jul-24. Thanks, Mark",
    "Hi Tom, Could you offer ISIN AU1234567890, 1-year tenor in AUD, and ISIN AU1234567891, 3-year tenor in USD, both value date today? Thanks, Jessica",
    "Morning Jessica, We can offer 99.50 for 10Mio AUD on the 1-year and 101.20 for 5M USD on the 3-year. Cheers, Tom",
    "Hello Michael, Is there a bid for ISIN CA9876543210, 6-month tenor in CAD, and ISIN CA9876543211, 2-year tenor in USD, vd T+0 Thanks, Alice",
    "Hi Alice, We can bid 100.75 for 8 Mio CAD on the 6-month and 98.60 for 7MM USD on the 2-year. Best, Michael",
    "Good afternoon Eva, Do you have offers on ISIN FR5678901234, 4-year tenor in EUR, and ISIN FR5678901235, 7-year tenor in USD, vd t+1 Thanks in advance, Liam",
    "Hi Liam, Yes, we can offer at 102.10 for 9 Mios EUR on the 4-year and 103.25 for 4 mil USD on the 7-year. Regards, Eva",
    "Good afternoon Sara, Bid for ISIN IT6543210987, 1 YEAR EUR and ISIN IT6543210988, 24 Mon USD, T+1 Regards, John",
    "Hello John, Need offer for ISIN ES3456789012, 3Y EUR and ISIN ES3456789013, 48 M USD, T+1 Best, Sara",
    "Hi Chris, Pls bid ISIN SE9876543210, 24 Mon SEK and ISIN SE9876543211, 48 Mon USD, T+1 Thx, Olivia",
    "Hiii Olivia, We need offer ISIN NO1234567890, 1 YEAR NOK and ISIN NO1234567891, 5Y USD, T+2 Best, Chris",
    "Hey Mike, Yr quotes for AU9876543210, 4 mil AUD and AU9876543211, 24 Mon USD, value date 10/10/24. Cheers, Laura",
    "Hi Laura, Ofr for CN1234567890, 1 YEAR CNY and CN1234567891, 48 Mon USD, value date 24-07-24. Thx, Mike",
    "Hey Mike, Yr quotes for AU9876543210, 4 mil AUD, amount 4 MM and AU9876543211, 2 year USD, amount 9 Mios, value on 10/10/24. Cheers, Laura",
    "Hi Laura, Ofr for CN1234567890, 1 YEAR CNY, amount 10Mio and CN1234567891, 48 Month USD, amount 20M, vd 24-07-24. Thx, Mike",
    "Hi Alex, Looking for quotes on ES8901234567, 2 Year EUR, amount 500K and ES8901234568, 160 month USD, amount 10 mio, vd 15-Aug-24. Cheers, Jessica",
    "Hello Michael, Need offer for GB9012345678, 3Y GBP, amount 450K and GB9012345679, 20YR USD, amount 8 mio, value date 25-Jul-24. Thanks, Emma",
    "Hello John, Need bids for MU6789012345, 2Y MUR, amount 7MM and MU6789012346, 160 month USD, amount 12 mio, value date 24-Jul-24. Best, Emma",
    "Hi Rachel, Please offer on NA7890123456, 3Y NAD, amount 500K and NA7890123457, 20YR USD, amount 10 mils, vd 25-Jul-24. Regards, Michael",
    "Hello Alex, Require quotes for RW0123456789, 1Y RWF, amount 2MM and RW0123456790, 10YR USD, amount 6 mio, value on 15-Aug-24. Thx, Olivia",
    "Hi Tom, Need bids on ST1234567890, 2Y STD, amount 250K and ST1234567891, 8YR USD, amount 12 mils, value date 01-Aug-24. Regards, Kevin",
    "Hello David, Please quote for SZ2345678901, 3YR SZL, amount 450 K and SZ2345678902, 60 month USD, amount 8 mio, value on 30-Jul-24. Thanks, Sarah",
]

def label_ents(msgs:list[set]):
    labelled_msgs = []
    for text in msgs:
        annot = {"entities": []}
        for match in re.finditer(isinRegexPattern, text):
            if match.start()+1 < match.end()-1:
                annot["entities"].append((match.start()+1, match.end()-1, "ISIN"))
        for match in re.finditer(tenorRegexPattern, text):
            if match.start()+1 < match.end()-1:
                annot["entities"].append((match.start()+1, match.end()-1, "TENOR"))
        for match in re.finditer(offerRegexPattern, text):
            if match.start()+1 < match.end()-1:
                annot["entities"].append((match.start()+1, match.end()-1, "OFFER"))
        for match in re.finditer(bidRegexPattern, text):
            if match.start()+1 < match.end()-1:
                annot["entities"].append((match.start()+1, match.end()-1, "BID"))
        for match in re.finditer(ccyRegexPattern, text):
            if match.start()+1 < match.end()-1:
                annot["entities"].append((match.start()+1, match.end()-1, "CURRENCY"))
        for match in re.finditer(vdRegexPattern, text):
            if match.start()+1 < match.end()-1:
                annot["entities"].append((match.start()+1, match.end()-1, "VALUE_DATE"))
        for match in re.finditer(amountRegexPattern, text):
            if match.start()+1 < match.end()-1:
                annot["entities"].append((match.start()+1, match.end()-1, "AMOUNT"))
        labelled_msgs.append((text, annot))
    return labelled_msgs

def add_to_docbin(msgs:list[set]) -> DocBin:
    docBin = DocBin()
    for text, annot in msgs:
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annot["entities"]:
            span = doc.char_span(start, end, label=label)
            if span is None:
                print(f"Skipping entity at [{start}, {end}] in '{text}' due to alignment issues.")
            else:
                ents.append(span)
        try:
            doc.ents = ents
            docBin.add(doc)
        except Exception as e:
            print(e)
    return docBin

def split_to_train_eval(all_items:list, train_percentage:float=0.85) -> set[list[set], list[set]]:
    num_items_to_select = int(len(all_items) * (train_percentage))
    selected_indices = random.sample(range(len(all_items)), len(all_items))
    train_items = [all_items[idx] for idx in selected_indices[:num_items_to_select]]
    eval_items = [all_items[idx] for idx in selected_indices[num_items_to_select:]]
    return train_items, eval_items


data = label_ents(DATASET)
TRAIN_DATA, EVAL_DATA = split_to_train_eval(data)

dir_path = os.path.dirname(os.path.realpath(__file__))

# Save the DocBin object to a file
trainingDocBin = add_to_docbin(TRAIN_DATA)
output_training_path = os.path.join(dir_path, "train.spacy")
trainingDocBin.to_disk(output_training_path)
num_trains = len(TRAIN_DATA)
print(f"Training data (totaled {num_trains}) saved to {output_training_path}")

# Save the DocBin object to a file
evalDocBin = add_to_docbin(EVAL_DATA)
output_eval_path = os.path.join(dir_path, "dev.spacy")
evalDocBin.to_disk(output_eval_path)
num_evals = len(EVAL_DATA)
print(f"Evaluation data (totaled {num_evals}) saved to {output_eval_path}")

