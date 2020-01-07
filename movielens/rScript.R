# Movie Lens Project : HarvardX: PH125.9x Data Science course.
# Frederico de Almeida Meirelles Palma
## https://github.com/fredpalma/machine-learning/movielens/


################################
# 2.1 - Creating edx set and validation set
################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead

test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# If required install some packages
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
library("ggplot2")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
library("dplyr")
```

# Data Analysis

## RMSE - The loss-function

RMSE <- function(predicted_ratings, true_ratings){
  sqrt(mean((predicted_ratings - true_ratings)^2))
}


## Exploring edx dataset

## r number of rows and columns

dim(edx)

## head

head(edx)

## rating range and the quantity of each

table(edx$rating)

## rating histogram

hist(edx$rating)


## Counting rating

edx %>%
  group_by(rating) %>%
  summarize(count = n()) %>%
  arrange(desc(count)) %>%
  knitr::kable()



## Rating Summary
  
summary(edx$rating)


## Summarize distinct users and movies

edx %>%
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))


## Getting 100 users and 100 movies 
## Matrix Visualization 

users <- sample(unique(edx$userId), 100)
rafalib::mypar()
edx %>% filter(userId %in% users) %>% 
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>% select(sample(ncol(.), 100)) %>% 
  as.matrix() %>% t(.) %>%
  image(1:100, 1:100,. , xlab="Movies", ylab="Users")
abline(h=0:100+0.5, v=0:100+0.5, col = "grey")

## Movie rating distribution, echo = FALSE}

edx %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
xlab("Number of Ratings") +
  ylab("Number of Movies") +
  ggtitle("Quantity of Ratings per Movies")


## Movies with 1 rating, echo = TRUE}

edx %>%
group_by(title) %>%
  summarize(count = n()) %>%
  filter(count == 1) %>%
knitr::kable()

## Active Users - histogram

edx %>%
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  ggtitle("Users")

# Model Development

## RMSE Function

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


## Average Rating Model

## edx rating mean

mu <- mean(edx$rating)
mu


## RMSE in test set

naive_rmse <- RMSE(validation$rating, mu)
naive_rmse


## RMSE Average Rating

rmse_ar <- naive_rmse
rmse_ar

## Movie Effect Model

## Movies averages

movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

##Ploting movie_avgs histogram

movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))


## Adding movie effect in the model

predicted_ratings <- mu + validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i

model_2_rmse <- RMSE(predicted_ratings, validation$rating)

rmse_me <- model_2_rmse 

rmse_me

## User Effect Model

user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

## Ploting user_avgs histogram

user_avgs %>% qplot(b_u, geom ="histogram", bins = 10, data = ., color = I("black"))

## Adding User Effect Model

predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

model_3_rmse <- RMSE(predicted_ratings, validation$rating)

rmse_ue <- model_3_rmse

rmse_ue


## Regularization

## cross-validation

lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  mu <- mean(edx$rating)
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- 
    edx %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, edx$rating))
})

## Lambdas plot

qplot(lambdas, rmses)  


## Best lambda

lambda <- lambdas[which.min(rmses)]
lambda


## Adding regularization:
  
## regularized movie and user effect model

rmse_r <- min(rmses)
rmse_r

# Results

## The best model for this project is the Regularized Movie and User Effect:
## r best rmse

rmse_r

# System Info
## r Version

print("System Information:")
version