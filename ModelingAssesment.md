Modeling Assesment
================

# Tickets Listed

## Load in data and Visualizing

``` r
library(tidyverse)
library(xgboost)
library(Matrix)
library(MLmetrics)
library(Metrics)

## load in the data
sales <- read.csv("~/Downloads/assessment_data.tsv", sep = "\t")

## make sure the date in the right format 
sales$listing_date <- as.Date(sales$listing_date)
sales$event_date <- as.Date(sales$event_datetime)

# Getting an idea for variable types 
colnames(sales)
```

    ##  [1] "event_id"              "listing_date"          "event_listing_date_id"
    ##  [4] "taxonomy"              "event_title"           "event_datetime"       
    ##  [7] "tickets_listed"        "mean_listing_price"    "performer_1"          
    ## [10] "performer_2"           "performer_3"           "performer_4"          
    ## [13] "venue_name"            "event_date"

``` r
# [1] "event_id"              "listing_date"          "event_listing_date_id" "taxonomy"             
# [5] "event_title"           "event_datetime"        "tickets_listed"        "mean_eventt_price"   
# [9] "performer_1"           "performer_2"           "performer_3"           "performer_4"          
# [13] "venue_name"     

## Take a look at the data
ggplot(sales, aes(x =listing_date,  y=tickets_listed, color = taxonomy)) +
  geom_point() +
  facet_wrap(~taxonomy)
```

![](ModelingAssesment_files/figure-gfm/Read%20In%20Data-1.png)<!-- -->

## Feature Engineering

First thing I noticed when looking at the data is that the different
venues/taxonomy have very different scales when it comes to the ouput
variables. For this reason, log transforming the values makes sense as
to be able to better assess error.

``` r
## log transform the values 
sales$mean_listing_price <- log1p(sales$mean_listing_price)
sales$tickets_listed <- log1p(sales$tickets_listed)
```

Next, given the temporal nature of the listings and day of event,
convert the date to time based features

``` r
##  convert to time-based features
sales$listing_month <- as.numeric(substr(sales$listing_date, 6, 7))
sales$listing_day <- as.numeric(as.factor(weekdays(as.Date(sales$listing_date)))) #day of week
sales$listing_dayDate <- as.numeric(substr(sales$listing_date, 9, 10))
sales$listing_year  <- substr(sales$listing_date, 1, 4)
sales$event_month <- as.numeric(substr(sales$event_date, 6, 7))
sales$event_day <- as.numeric(as.factor(weekdays(as.Date(sales$event_date)))) #day of week
sales$event_dayDate <- as.numeric(substr(sales$event_date, 9, 10))
sales$event_year  <- substr(sales$event_date, 1, 4)
```

Looking at the dataset under taxonomy, some were NA but clearly
orchestral.

``` r
sales %>% count(taxonomy)
```

    ##                taxonomy    n
    ## 1  Classical Orchestral 4043
    ## 2 Minor League Baseball 2054
    ## 3          MLB Baseball 2742
    ## 4                  <NA>  141

``` r
## fix 
sales[is.na(sales$taxonomy),] %>% head()
```

    ##      event_id listing_date event_listing_date_id taxonomy
    ## 4765  3798567   2017-07-14         3798567_17361     <NA>
    ## 4776  3798570   2017-07-14         3798570_17361     <NA>
    ## 4866  3798567   2017-07-15         3798567_17362     <NA>
    ## 4877  3798570   2017-07-15         3798570_17362     <NA>
    ## 4966  3798567   2017-07-16         3798567_17363     <NA>
    ## 4977  3798570   2017-07-16         3798570_17363     <NA>
    ##                                                               event_title
    ## 4765           M. Ward with Los Angeles Philharmonic and Rhiannon Giddens
    ## 4776 Natalia LaFourcade with Gustavo Dudamel and Los Angeles Philharmonic
    ## 4866           M. Ward with Los Angeles Philharmonic and Rhiannon Giddens
    ## 4877 Natalia LaFourcade with Gustavo Dudamel and Los Angeles Philharmonic
    ## 4966           M. Ward with Los Angeles Philharmonic and Rhiannon Giddens
    ## 4977 Natalia LaFourcade with Gustavo Dudamel and Los Angeles Philharmonic
    ##           event_datetime tickets_listed mean_listing_price
    ## 4765 2017-10-25 20:00:00       3.526361           6.195385
    ## 4776 2017-10-12 20:00:00       3.367296           6.375025
    ## 4866 2017-10-25 20:00:00       3.526361           6.195385
    ## 4877 2017-10-12 20:00:00       3.367296           6.375025
    ## 4966 2017-10-25 20:00:00       3.526361           6.124683
    ## 4977 2017-10-12 20:00:00       3.367296           6.448715
    ##                   performer_1              performer_2      performer_3
    ## 4765                  M. Ward Los Angeles Philharmonic Rhiannon Giddens
    ## 4776 Los Angeles Philharmonic       Natalia LaFourcade  Gustavo Dudamel
    ## 4866                  M. Ward Los Angeles Philharmonic Rhiannon Giddens
    ## 4877 Los Angeles Philharmonic       Natalia LaFourcade  Gustavo Dudamel
    ## 4966                  M. Ward Los Angeles Philharmonic Rhiannon Giddens
    ## 4977 Los Angeles Philharmonic       Natalia LaFourcade  Gustavo Dudamel
    ##      performer_4               venue_name event_date listing_month listing_day
    ## 4765        <NA> Walt Disney Concert Hall 2017-10-25             7           1
    ## 4776        <NA> Walt Disney Concert Hall 2017-10-12             7           1
    ## 4866        <NA> Walt Disney Concert Hall 2017-10-25             7           3
    ## 4877        <NA> Walt Disney Concert Hall 2017-10-12             7           3
    ## 4966        <NA> Walt Disney Concert Hall 2017-10-25             7           4
    ## 4977        <NA> Walt Disney Concert Hall 2017-10-12             7           4
    ##      listing_dayDate listing_year event_month event_day event_dayDate
    ## 4765              14         2017          10         7            25
    ## 4776              14         2017          10         5            12
    ## 4866              15         2017          10         7            25
    ## 4877              15         2017          10         5            12
    ## 4966              16         2017          10         7            25
    ## 4977              16         2017          10         5            12
    ##      event_year
    ## 4765       2017
    ## 4776       2017
    ## 4866       2017
    ## 4877       2017
    ## 4966       2017
    ## 4977       2017

``` r
sales$taxonomy[is.na(sales$taxonomy)] <- "Classical Orchestral"
```

Finally, I noticed when looking at some event IDs that tickets\_listed
seems to be correlated with how long it has been since the event was
first listed so exploring that and adding a variable for days since
listed.

``` r
## mlb
randombaseballgame <- sales %>% filter(event_id == "3594124")

ggplot(randombaseballgame, aes(x=listing_date, y = tickets_listed)) +
  geom_point() +
  ggtitle(randombaseballgame$event_title)
```

![](ModelingAssesment_files/figure-gfm/DaysSinceListed-1.png)<!-- -->

``` r
## Symphony
randomOrch <- sales %>% filter(event_id == "3799889")
ggplot(randomOrch, aes(x=listing_date, y = tickets_listed)) +
  geom_point() +
  ggtitle(randomOrch$event_title)
```

![](ModelingAssesment_files/figure-gfm/DaysSinceListed-2.png)<!-- -->

``` r
## minor league
RandomMinor <- sales %>% filter(event_id == "3750580")

ggplot(RandomMinor, aes(x=listing_date, y = tickets_listed)) +
  geom_point() +
  ggtitle(RandomMinor$event_title)
```

![](ModelingAssesment_files/figure-gfm/DaysSinceListed-3.png)<!-- -->

``` r
## ADD variable for days since listing
sales <- sales %>% group_by(event_id) %>% mutate(daysSinceListing = row_number() -1 )
```

Next, I wanted to see if there were any patterns between some of the
features and values

``` r
ggplot(sales, aes(x =factor(event_day),  y=tickets_listed)) +
  geom_boxplot() +
  facet_wrap(~taxonomy) +
  theme(legend.position = "none")
```

![](ModelingAssesment_files/figure-gfm/plot-1.png)<!-- -->

``` r
ggplot(sales, aes(x =factor(event_month),  y=tickets_listed)) +
  geom_boxplot() +
  facet_wrap(~taxonomy) +
  theme(legend.position = "none")
```

![](ModelingAssesment_files/figure-gfm/plot-2.png)<!-- -->

``` r
ggplot(sales, aes(x =factor(taxonomy),  y=tickets_listed)) +
  geom_boxplot() +
  theme(legend.position = "none")
```

![](ModelingAssesment_files/figure-gfm/plot-3.png)<!-- -->

## Model Building with xgrboost

``` r
## Test this type of model leaving out some of the known data to validate on
train <- sales %>% filter(listing_date < "2017-07-20")

## leave some behind to validate on
validation <- sales %>% filter(listing_date >= "2017-07-20", listing_date < "2017-08-01")
test <-  sales %>% filter(listing_date >= "2017-08-01")
label <- train$tickets_listed

## Great input train matrix
trainMatrix <- sparse.model.matrix(~ event_month + event_day + taxonomy + daysSinceListing + event_id,
                                   data = train,
                                   contrasts.arg =  c("event_month",
                                                      "event_day",
                                                      "event_year",
                                                      "taxonomy",
                                                      "daysSinceListing" ,
                                                      "event_id"),
                                   sparse = FALSE,
                                   sapsci =FALSE)

#Create input for xgboost
trainDMatrix <- xgb.DMatrix(data = trainMatrix, label = label)

params <- list(booster = "gbtree",
               objective = "reg:linear",
               eta=0.4,
               gamma=0
)

#Cross-validation
xgb.tab <- xgb.cv(data = trainDMatrix,
                  param = params,
                  maximize = FALSE, 
                  evaluation = "rmse", 
                  nrounds = 100, 
                  nthreads = 10,
                  nfold = 2,
                  early_stopping_round = 10)
```

    ## [19:42:45] WARNING: amalgamation/../src/objective/regression_obj.cu:171: reg:linear is now deprecated in favor of reg:squarederror.
    ## [19:42:45] WARNING: amalgamation/../src/objective/regression_obj.cu:171: reg:linear is now deprecated in favor of reg:squarederror.
    ## [1]  train-rmse:3.482259+0.012886    test-rmse:3.482467+0.012353 
    ## Multiple eval metrics are present. Will use test_rmse for early stopping.
    ## Will train until test_rmse hasn't improved in 10 rounds.
    ## 
    ## [2]  train-rmse:2.108574+0.007311    test-rmse:2.109422+0.005445 
    ## [3]  train-rmse:1.287266+0.002220    test-rmse:1.289585+0.002950 
    ## [4]  train-rmse:0.801846+0.000150    test-rmse:0.805654+0.000298 
    ## [5]  train-rmse:0.514771+0.002344    test-rmse:0.525097+0.004430 
    ## [6]  train-rmse:0.351867+0.002113    test-rmse:0.368741+0.005668 
    ## [7]  train-rmse:0.265928+0.000670    test-rmse:0.289721+0.006018 
    ## [8]  train-rmse:0.218693+0.002744    test-rmse:0.247668+0.001968 
    ## [9]  train-rmse:0.188965+0.001098    test-rmse:0.224829+0.000666 
    ## [10] train-rmse:0.174465+0.001726    test-rmse:0.214053+0.000321 
    ## [11] train-rmse:0.163200+0.004031    test-rmse:0.208116+0.000963 
    ## [12] train-rmse:0.150417+0.002806    test-rmse:0.200638+0.001220 
    ## [13] train-rmse:0.143620+0.003423    test-rmse:0.197991+0.001494 
    ## [14] train-rmse:0.136481+0.000729    test-rmse:0.194973+0.000964 
    ## [15] train-rmse:0.130717+0.000757    test-rmse:0.192887+0.000722 
    ## [16] train-rmse:0.126119+0.001356    test-rmse:0.191441+0.001273 
    ## [17] train-rmse:0.122621+0.000384    test-rmse:0.189726+0.000417 
    ## [18] train-rmse:0.118892+0.001659    test-rmse:0.187715+0.000415 
    ## [19] train-rmse:0.114326+0.000652    test-rmse:0.186794+0.000575 
    ## [20] train-rmse:0.110759+0.000749    test-rmse:0.186064+0.001726 
    ## [21] train-rmse:0.108773+0.000676    test-rmse:0.185967+0.002017 
    ## [22] train-rmse:0.105037+0.000157    test-rmse:0.183877+0.001901 
    ## [23] train-rmse:0.103357+0.000341    test-rmse:0.183658+0.001991 
    ## [24] train-rmse:0.100805+0.001172    test-rmse:0.183300+0.002111 
    ## [25] train-rmse:0.099099+0.001909    test-rmse:0.182466+0.001534 
    ## [26] train-rmse:0.097594+0.002665    test-rmse:0.182824+0.001905 
    ## [27] train-rmse:0.095692+0.003221    test-rmse:0.182322+0.002365 
    ## [28] train-rmse:0.093179+0.003490    test-rmse:0.181781+0.001842 
    ## [29] train-rmse:0.091085+0.003577    test-rmse:0.180688+0.001603 
    ## [30] train-rmse:0.089945+0.004272    test-rmse:0.180440+0.001011 
    ## [31] train-rmse:0.086623+0.002000    test-rmse:0.179069+0.001760 
    ## [32] train-rmse:0.085208+0.002220    test-rmse:0.178914+0.001991 
    ## [33] train-rmse:0.084412+0.002251    test-rmse:0.178690+0.001809 
    ## [34] train-rmse:0.082145+0.003250    test-rmse:0.178026+0.000934 
    ## [35] train-rmse:0.081593+0.003363    test-rmse:0.177919+0.000844 
    ## [36] train-rmse:0.077944+0.003365    test-rmse:0.177186+0.000403 
    ## [37] train-rmse:0.076837+0.003680    test-rmse:0.176580+0.000060 
    ## [38] train-rmse:0.075342+0.002651    test-rmse:0.176252+0.000343 
    ## [39] train-rmse:0.074206+0.002545    test-rmse:0.176320+0.000034 
    ## [40] train-rmse:0.072344+0.002862    test-rmse:0.176102+0.000192 
    ## [41] train-rmse:0.071114+0.003200    test-rmse:0.175797+0.000174 
    ## [42] train-rmse:0.070276+0.003934    test-rmse:0.175708+0.000214 
    ## [43] train-rmse:0.069469+0.004169    test-rmse:0.175627+0.000238 
    ## [44] train-rmse:0.067644+0.002716    test-rmse:0.175755+0.000448 
    ## [45] train-rmse:0.066562+0.002271    test-rmse:0.175966+0.000316 
    ## [46] train-rmse:0.065225+0.001989    test-rmse:0.176171+0.000227 
    ## [47] train-rmse:0.063868+0.002980    test-rmse:0.175743+0.000119 
    ## [48] train-rmse:0.062100+0.002585    test-rmse:0.175830+0.000286 
    ## [49] train-rmse:0.061417+0.002819    test-rmse:0.176142+0.000594 
    ## [50] train-rmse:0.060279+0.003521    test-rmse:0.175909+0.000367 
    ## [51] train-rmse:0.059067+0.002705    test-rmse:0.175959+0.000646 
    ## [52] train-rmse:0.058361+0.002883    test-rmse:0.175980+0.000726 
    ## [53] train-rmse:0.057450+0.003179    test-rmse:0.175616+0.000348 
    ## [54] train-rmse:0.056703+0.003067    test-rmse:0.175457+0.000312 
    ## [55] train-rmse:0.055084+0.001907    test-rmse:0.175152+0.000507 
    ## [56] train-rmse:0.054103+0.001803    test-rmse:0.174959+0.000491 
    ## [57] train-rmse:0.053231+0.001837    test-rmse:0.174984+0.000715 
    ## [58] train-rmse:0.052491+0.002171    test-rmse:0.174873+0.000634 
    ## [59] train-rmse:0.051854+0.001941    test-rmse:0.174793+0.000480 
    ## [60] train-rmse:0.051175+0.002020    test-rmse:0.174788+0.000550 
    ## [61] train-rmse:0.050019+0.001700    test-rmse:0.174868+0.000617 
    ## [62] train-rmse:0.048742+0.001887    test-rmse:0.174877+0.000673 
    ## [63] train-rmse:0.047925+0.001691    test-rmse:0.174830+0.000558 
    ## [64] train-rmse:0.047382+0.001833    test-rmse:0.175034+0.000533 
    ## [65] train-rmse:0.046377+0.001517    test-rmse:0.174844+0.000495 
    ## [66] train-rmse:0.045852+0.001455    test-rmse:0.174945+0.000441 
    ## [67] train-rmse:0.045408+0.001534    test-rmse:0.174978+0.000335 
    ## [68] train-rmse:0.045164+0.001519    test-rmse:0.175054+0.000401 
    ## [69] train-rmse:0.044824+0.001577    test-rmse:0.175042+0.000324 
    ## [70] train-rmse:0.044510+0.001849    test-rmse:0.175084+0.000286 
    ## Stopping. Best iteration:
    ## [60] train-rmse:0.051175+0.002020    test-rmse:0.174788+0.000550

``` r
#Number of rounds choosen above
num_iterations = xgb.tab$best_iteration


## Create model
model <- xgb.train(data = trainDMatrix,
                   param = params,
                   evaluation = 'rmse',
                   maximize = FALSE,
                   nrounds = num_iterations)
```

    ## [19:42:45] WARNING: amalgamation/../src/objective/regression_obj.cu:171: reg:linear is now deprecated in favor of reg:squarederror.
    ## [19:42:45] WARNING: amalgamation/../src/learner.cc:573: 
    ## Parameters: { "evaluation" } might not be used.
    ## 
    ##   This may not be accurate due to some parameters are only used in language bindings but
    ##   passed down to XGBoost core.  Or some parameters are not used but slip through this
    ##   verification. Please open an issue if you find above cases.

``` r
## Plot feature Importance
importance <- xgb.importance(feature_names = colnames(trainMatrix), model = model)
xgb.ggplot.importance(importance_matrix = importance)
```

![](ModelingAssesment_files/figure-gfm/model-1.png)<!-- -->

``` r
## Create input validationing matrix
validationMatrix <- sparse.model.matrix(~ event_month + event_day  + taxonomy + daysSinceListing + event_id,
                                  data = validation,
                                  contrasts.arg =  c("event_month",
                                                     "event_day",
                                                     "event_year",
                                                     "taxonomy",
                                                     "daysSinceListing",
                                                     "event_id"),
                                   sparse = FALSE,
                                   sci = FALSE)
```

## Forcast Future Values

Next, we need to use the model above to access the data

``` r
## Forcast future values for the validation
pred <- predict(model, validationMatrix)
validation$tickets_listed_pred <- pred

## Check the accuracy of the validation data
ML <- validation %>% select(tickets_listed, tickets_listed_pred, taxonomy)
ggplot(ML, aes(x=tickets_listed, y = tickets_listed_pred, color = taxonomy))+
  geom_point(alpha = 0.4)
```

![](ModelingAssesment_files/figure-gfm/forcastFutureValues-1.png)<!-- -->

``` r
## Access the validation set
rmse(ML$tickets_listed, ML$tickets_listed_pred)
```

    ## [1] 0.3805528

``` r
R2_Score(ML$tickets_listed, ML$tickets_listed_pred)
```

    ## [1] 0.9770685

``` r
## Convert back to readable form
ML$tickets_listed <- round(expm1(ML$tickets_listed))
ML$tickets_listed_pred <- round(expm1(ML$tickets_listed_pred))
head(ML, 20)
```

    ## # A tibble: 20 × 4
    ## # Groups:   event_id [20]
    ##    event_id tickets_listed tickets_listed_pred taxonomy            
    ##       <int>          <dbl>               <dbl> <chr>               
    ##  1  3594065           6029                6143 MLB Baseball        
    ##  2  3594070           8666                9509 MLB Baseball        
    ##  3  3594068           9102                9614 MLB Baseball        
    ##  4  3594103           4910                5902 MLB Baseball        
    ##  5  3594105           3334                4278 MLB Baseball        
    ##  6  3594101           6670                7050 MLB Baseball        
    ##  7  3594108           2234                3848 MLB Baseball        
    ##  8  3798568             24                  28 Classical Orchestral
    ##  9  3594080           8947                9424 MLB Baseball        
    ## 10  3594078           8599                8592 MLB Baseball        
    ## 11  3594063           8876                9167 MLB Baseball        
    ## 12  3594059           8190                8459 MLB Baseball        
    ## 13  3594057           8504                8668 MLB Baseball        
    ## 14  3594061           5662                5817 MLB Baseball        
    ## 15  3931481              7                   6 Classical Orchestral
    ## 16  3799893             39                  38 Classical Orchestral
    ## 17  3799894             33                  33 Classical Orchestral
    ## 18  3799887             39                  38 Classical Orchestral
    ## 19  3799899             39                  37 Classical Orchestral
    ## 20  3799888             19                  26 Classical Orchestral

## Predict the testing data for assessment

``` r
## Create input validationing matrix
testMatrix <- sparse.model.matrix(~ event_month + event_day  + taxonomy + daysSinceListing + event_id,
                                  data = test,
                                  contrasts.arg =  c("event_month",
                                                     "event_day",
                                                     "event_year",
                                                     "taxonomy",
                                                     "daysSinceListing",
                                                     "event_id"),
                                   sparse = FALSE,
                                   sci = FALSE)
## Forcast future values for the validation
pred <- predict(model, testMatrix)

test$tickets_listed <- pred
sales$tickets_listed[match(test$event_listing_date_id, sales$event_listing_date_id)] <- pred
sales$tickets_listed <- round(expm1(sales$tickets_listed))

write.table(sales, "~/Documents/asses/AssessmentOutput.tsv", 
            sep = "\t",
            quote = F, 
            col.names = T, 
            row.names = T)
```

# Mean Listing Price

## Feature Engineering

First I wanted to look how days since listing effects the price of
tickets. In this case, it didn’t seem to have as clear a correlation so
I will not include the mean\_listing\_price in my model

``` r
## mlb
randombaseballgame <- sales %>% filter(event_id == "3594124")

ggplot(randombaseballgame, aes(x=listing_date, y = mean_listing_price)) +
  geom_point() +
  ggtitle(randombaseballgame$event_title)
```

![](ModelingAssesment_files/figure-gfm/MLPDaysSinceListed-1.png)<!-- -->

``` r
## Symphony
randomOrch <- sales %>% filter(event_id == "3799889")
ggplot(randomOrch, aes(x=listing_date, y = mean_listing_price)) +
  geom_point() +
  ggtitle(randomOrch$event_title)
```

![](ModelingAssesment_files/figure-gfm/MLPDaysSinceListed-2.png)<!-- -->

``` r
## minor league
RandomMinor <- sales %>% filter(event_id == "3750580")

ggplot(RandomMinor, aes(x=listing_date, y = mean_listing_price)) +
  geom_point() +
  ggtitle(RandomMinor$event_title)
```

![](ModelingAssesment_files/figure-gfm/MLPDaysSinceListed-3.png)<!-- -->

Next, I wanted to see if there were any patterns between some of the
features and values

``` r
ggplot(sales, aes(x =factor(event_day),  y=mean_listing_price)) +
  geom_boxplot() +
  facet_wrap(~taxonomy) +
  theme(legend.position = "none")
```

![](ModelingAssesment_files/figure-gfm/MLPplot-1.png)<!-- -->

``` r
ggplot(sales, aes(x =factor(event_month),  y=mean_listing_price)) +
  geom_boxplot() +
  facet_wrap(~taxonomy) +
  theme(legend.position = "none")
```

![](ModelingAssesment_files/figure-gfm/MLPplot-2.png)<!-- -->

``` r
ggplot(sales, aes(x =factor(taxonomy),  y=mean_listing_price)) +
  geom_boxplot() +
  theme(legend.position = "none")
```

![](ModelingAssesment_files/figure-gfm/MLPplot-3.png)<!-- -->

## Model Building with xgrboost

``` r
## Test this type of model leaving out some of the known data to validate on
train <- sales %>% filter(listing_date < "2017-07-20")

## leave some behind to validate on
validation <- sales %>% filter(listing_date >= "2017-07-20", listing_date < "2017-08-01")
test <-  sales %>% filter(listing_date >= "2017-08-01")
label <- train$mean_listing_price

## Great input train matrix
trainMatrix <- sparse.model.matrix(~ event_month + event_day + taxonomy + event_id,
                                   data = train,
                                   contrasts.arg =  c("event_month",
                                                      "event_day",
                                                      "event_year",
                                                      "taxonomy",
                                                      "event_id"),
                                   sparse = FALSE,
                                   sapsci =FALSE)

#Create input for xgboost
trainDMatrix <- xgb.DMatrix(data = trainMatrix, label = label)

params <- list(booster = "gbtree"
               , objective = "reg:linear"
               , eta=0.4
               , gamma=0
)

#Cross-validation
xgb.tab <- xgb.cv(data = trainDMatrix
                  , param = params
                  , maximize = FALSE, evaluation = "rmse", nrounds = 100
                  , nthreads = 10, nfold = 2, early_stopping_round = 10)
```

    ## [19:42:48] WARNING: amalgamation/../src/objective/regression_obj.cu:171: reg:linear is now deprecated in favor of reg:squarederror.
    ## [19:42:48] WARNING: amalgamation/../src/objective/regression_obj.cu:171: reg:linear is now deprecated in favor of reg:squarederror.
    ## [1]  train-rmse:2.592301+0.013582    test-rmse:2.593138+0.021322 
    ## Multiple eval metrics are present. Will use test_rmse for early stopping.
    ## Will train until test_rmse hasn't improved in 10 rounds.
    ## 
    ## [2]  train-rmse:1.575830+0.007676    test-rmse:1.577245+0.021327 
    ## [3]  train-rmse:0.970049+0.004119    test-rmse:0.972457+0.020238 
    ## [4]  train-rmse:0.615803+0.001727    test-rmse:0.619261+0.017288 
    ## [5]  train-rmse:0.413277+0.001434    test-rmse:0.419882+0.015373 
    ## [6]  train-rmse:0.303898+0.003684    test-rmse:0.312285+0.010767 
    ## [7]  train-rmse:0.249196+0.004580    test-rmse:0.261254+0.007910 
    ## [8]  train-rmse:0.225507+0.004830    test-rmse:0.239279+0.004612 
    ## [9]  train-rmse:0.212632+0.003541    test-rmse:0.227686+0.002245 
    ## [10] train-rmse:0.204637+0.004078    test-rmse:0.219948+0.002211 
    ## [11] train-rmse:0.201182+0.005420    test-rmse:0.216241+0.002987 
    ## [12] train-rmse:0.199108+0.005236    test-rmse:0.214013+0.002551 
    ## [13] train-rmse:0.197325+0.004457    test-rmse:0.212725+0.001671 
    ## [14] train-rmse:0.196706+0.004771    test-rmse:0.212019+0.002092 
    ## [15] train-rmse:0.195816+0.004179    test-rmse:0.211890+0.002157 
    ## [16] train-rmse:0.195435+0.004124    test-rmse:0.211501+0.002155 
    ## [17] train-rmse:0.195034+0.003883    test-rmse:0.211271+0.001948 
    ## [18] train-rmse:0.194588+0.003750    test-rmse:0.210934+0.001942 
    ## [19] train-rmse:0.194482+0.003797    test-rmse:0.210879+0.001903 
    ## [20] train-rmse:0.194007+0.003631    test-rmse:0.211048+0.002572 
    ## [21] train-rmse:0.193853+0.003557    test-rmse:0.210975+0.002505 
    ## [22] train-rmse:0.193735+0.003633    test-rmse:0.210792+0.002654 
    ## [23] train-rmse:0.193640+0.003646    test-rmse:0.210691+0.002759 
    ## [24] train-rmse:0.193555+0.003615    test-rmse:0.210693+0.002773 
    ## [25] train-rmse:0.193392+0.003615    test-rmse:0.210693+0.003166 
    ## [26] train-rmse:0.193308+0.003584    test-rmse:0.210689+0.003275 
    ## [27] train-rmse:0.193259+0.003599    test-rmse:0.210620+0.003306 
    ## [28] train-rmse:0.193137+0.003548    test-rmse:0.210577+0.003495 
    ## [29] train-rmse:0.193070+0.003502    test-rmse:0.210684+0.003624 
    ## [30] train-rmse:0.193014+0.003536    test-rmse:0.210543+0.003796 
    ## [31] train-rmse:0.192985+0.003521    test-rmse:0.210530+0.003841 
    ## [32] train-rmse:0.192964+0.003526    test-rmse:0.210496+0.003889 
    ## [33] train-rmse:0.192927+0.003502    test-rmse:0.210378+0.003864 
    ## [34] train-rmse:0.192919+0.003506    test-rmse:0.210383+0.003849 
    ## [35] train-rmse:0.192898+0.003493    test-rmse:0.210375+0.003837 
    ## [36] train-rmse:0.192879+0.003480    test-rmse:0.210416+0.003880 
    ## [37] train-rmse:0.192860+0.003480    test-rmse:0.210395+0.003930 
    ## [38] train-rmse:0.192857+0.003482    test-rmse:0.210373+0.003957 
    ## [39] train-rmse:0.192854+0.003480    test-rmse:0.210380+0.003958 
    ## [40] train-rmse:0.192850+0.003483    test-rmse:0.210368+0.003961 
    ## [41] train-rmse:0.192843+0.003485    test-rmse:0.210373+0.003978 
    ## [42] train-rmse:0.192836+0.003489    test-rmse:0.210347+0.004010 
    ## [43] train-rmse:0.192826+0.003480    test-rmse:0.210369+0.004037 
    ## [44] train-rmse:0.192820+0.003477    test-rmse:0.210343+0.004008 
    ## [45] train-rmse:0.192819+0.003476    test-rmse:0.210345+0.004022 
    ## [46] train-rmse:0.192816+0.003478    test-rmse:0.210332+0.004037 
    ## [47] train-rmse:0.192815+0.003478    test-rmse:0.210335+0.004038 
    ## [48] train-rmse:0.192813+0.003477    test-rmse:0.210348+0.004059 
    ## [49] train-rmse:0.192807+0.003474    test-rmse:0.210357+0.004082 
    ## [50] train-rmse:0.192806+0.003473    test-rmse:0.210336+0.004060 
    ## [51] train-rmse:0.192805+0.003473    test-rmse:0.210333+0.004065 
    ## [52] train-rmse:0.192802+0.003473    test-rmse:0.210402+0.004139 
    ## [53] train-rmse:0.192802+0.003473    test-rmse:0.210403+0.004139 
    ## [54] train-rmse:0.192802+0.003472    test-rmse:0.210403+0.004145 
    ## [55] train-rmse:0.192800+0.003473    test-rmse:0.210455+0.004205 
    ## [56] train-rmse:0.192800+0.003472    test-rmse:0.210456+0.004203 
    ## Stopping. Best iteration:
    ## [46] train-rmse:0.192816+0.003478    test-rmse:0.210332+0.004037

``` r
#Number of rounds choosen above
num_iterations = xgb.tab$best_iteration


## Create model
model <- xgb.train(data = trainDMatrix
                   , param = params
                   , maximize = FALSE, evaluation = 'rmse', nrounds = num_iterations)
```

    ## [19:42:49] WARNING: amalgamation/../src/objective/regression_obj.cu:171: reg:linear is now deprecated in favor of reg:squarederror.
    ## [19:42:49] WARNING: amalgamation/../src/learner.cc:573: 
    ## Parameters: { "evaluation" } might not be used.
    ## 
    ##   This may not be accurate due to some parameters are only used in language bindings but
    ##   passed down to XGBoost core.  Or some parameters are not used but slip through this
    ##   verification. Please open an issue if you find above cases.

``` r
## Plot feature Importance
importance <- xgb.importance(feature_names = colnames(trainMatrix), model = model)
xgb.ggplot.importance(importance_matrix = importance)
```

![](ModelingAssesment_files/figure-gfm/MLPmodel-1.png)<!-- -->

``` r
## Create input validationing matrix
validationMatrix <- sparse.model.matrix(~ event_month + event_day  + taxonomy + event_id,
                                  data = validation,
                                  contrasts.arg =  c("event_month",
                                                     "event_day",
                                                     "event_year",
                                                     "taxonomy",
                                                     "event_id"),
                                   sparse = FALSE,
                                   sci = FALSE)
```

## Forcast Future Values

Next, we need to use the model above to access the data

``` r
## Forcast future values for the validation
pred <- predict(model, validationMatrix)
validation$mean_listing_price_pred <- pred

## Check the accuracy of the validation data
ML <- validation %>% select(mean_listing_price, mean_listing_price_pred, taxonomy)
ggplot(ML, aes(x=mean_listing_price, y = mean_listing_price_pred, color = taxonomy))+
  geom_point(alpha = 0.4)
```

![](ModelingAssesment_files/figure-gfm/MLPforcastFutureValues-1.png)<!-- -->

``` r
## Access the validation set
rmse(ML$mean_listing_price, ML$mean_listing_price_pred)
```

    ## [1] 0.1857735

``` r
R2_Score(ML$mean_listing_price, ML$mean_listing_price_pred)
```

    ## [1] 0.9776671

``` r
## Convert back to readable form
ML$mean_listing_price <- round(expm1(ML$mean_listing_price))
ML$mean_listing_price_pred <- round(expm1(ML$mean_listing_price_pred))
head(ML, 20)
```

    ## # A tibble: 20 × 4
    ## # Groups:   event_id [20]
    ##    event_id mean_listing_price mean_listing_price_pred taxonomy            
    ##       <int>              <dbl>                   <dbl> <chr>               
    ##  1  3594065                 71                      65 MLB Baseball        
    ##  2  3594070                 48                      49 MLB Baseball        
    ##  3  3594068                 33                      32 MLB Baseball        
    ##  4  3594103                 46                      47 MLB Baseball        
    ##  5  3594105                 54                      45 MLB Baseball        
    ##  6  3594101                 26                      37 MLB Baseball        
    ##  7  3594108                 28                      29 MLB Baseball        
    ##  8  3798568                452                     465 Classical Orchestral
    ##  9  3594080                 50                      50 MLB Baseball        
    ## 10  3594078                 43                      40 MLB Baseball        
    ## 11  3594063                 32                      33 MLB Baseball        
    ## 12  3594059                 48                      53 MLB Baseball        
    ## 13  3594057                 39                      44 MLB Baseball        
    ## 14  3594061                 99                      73 MLB Baseball        
    ## 15  3931481                348                     374 Classical Orchestral
    ## 16  3799893                598                     611 Classical Orchestral
    ## 17  3799894                600                     629 Classical Orchestral
    ## 18  3799887                598                     608 Classical Orchestral
    ## 19  3799899                598                     596 Classical Orchestral
    ## 20  3799888                352                     382 Classical Orchestral

## Predict the testing data for assessment

``` r
## Create input validationing matrix
testMatrix <- sparse.model.matrix(~ event_month + event_day  + taxonomy  + event_id,
                                  data = test,
                                  contrasts.arg =  c("event_month",
                                                     "event_day",
                                                     "event_year",
                                                     "taxonomy",
                                                     "event_id"),
                                   sparse = FALSE,
                                   sci = FALSE)
## Forcast future values for the validation
pred <- predict(model, testMatrix)

test$mean_listing_price <- pred
sales$mean_listing_price[match(test$event_listing_date_id, sales$event_listing_date_id)] <- pred

## Convert whole df back to nonlog
sales$mean_listing_price <- round(expm1(sales$mean_listing_price))
write.table(sales, "~/Documents/asses/AssessmentOutput.tsv", sep = "\t", quote = F, col.names = T, row.names = F)
```
