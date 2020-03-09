#---
#  title: "HarvardX: PH125.9x Understanding Recommendation Systems"
#author: "Sky Bowthorpe"
#date: "2/3/2020"
#output: pdf_document
#---
#  ```{r setup, include=FALSE}
#knitr::opts_chunk$set(echo = TRUE)
#```
## 1. Overview 
#  This is a project for the Data Science: Capstone course in the Data Science Professional Certificate Program
#The science project will be focused on creating a recommendation system model based on Netflix challenge in 2006, 
#the goal is to improve recommendation algorithm by at least 10%. The Dataset contains movie ratings by a user and includes the user id, 
#movie id, rating, time of the rating, title of the movie, and genres of the movie.
  
## 2. Loading the dataset and wrangling
#  The data is from the GroupLens research lab at the University of Minnesota.  
#The dataset can be found in the following links  
##-[MovieLens 10M dataset] https://grouplens.org/datasets/movielens/10m/
#  -[MovieLens 10M dataset - zip file] http://files.grouplens.org/datasets/movielens/ml-10m.zip

################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

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

##Data exploration and visualization
#Before creating models, lets explore the data to inform the variables to adjust in the model.  

#How many users, movies, and genres are in the dataset?  
  edx %>% summarise(
    n_distinct(userId),
    n_distinct(movieId),
    n_distinct(genres))

# How many movies do users normally rate? 
edx %>% 
  count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(color = "black", bins=30, fill="black") +
  scale_x_log10() +
  xlab("Number of ratings by individual reviewers") +
  ylab("Number of Users") +
  ggtitle("Number of Users Ratings") +
  theme(plot.title = element_text(hjust = 0.5))

# How many ratings do Movies normally get? 
edx%>%count(movieId)%>%
  ggplot(aes(n))+geom_histogram(col='Black',bins = 50,fill='Black')+
  scale_x_log10()+xlab('moviesrating count') 

#Load required libraries
library(tidyverse)
library(caret)
library(data.table)

#  Now I have a better idea of what the dataset includes, lets start by creating a naive model to compare against then a model that accounts for users and one that accounts for movies. 
#This naive model will just be the average rating of all movies. If we need to continue to adjust after accounting for users and movies we will go back to exploring the dataset.
  
#  In order to be able to test the robustness of the model we will need to patrician a part of the data to test iterations against.
#The goal is to build a model that can predict the ratings a user may give a movie based on the ratings they have given other movies with a RMSE <= 0.8649.

# Create a test set
test_index <- createDataPartition(y = edx$rating, times = 1,
                                  p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

#People tend to give a whole number score, Half-Star votes are less common than “Full-Star” votes.  
#There are 10 rating and the average is 3.512.   
#Most of the ratings are above 2, and the most common is 4. 


# Create a naive set for comparison >> this is just to help understand how much the effects are informing the model
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))}

mu_hat <- mean(train_set$rating)
naive_rmse <- RMSE(test_set$rating, mu_hat)
predictions <- rep(2.5, nrow(test_set))
rmse_results <- tibble(method = "Just the average", RMSE = naive_rmse)



#Now lets add the effect of the movie, we would expect good movies to have better ratings than bad movies.  Now we will make the model reflect that.
#"The Movie Effect"

# Improve the movel by adding B-i "Movie Effect" 
# fit <- lm(rating ~ as.factor(userId), data = edx)  # This will take a very long time if you run it.
mu <- mean(train_set$rating) 
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))
predicted_ratings <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i
model_1_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie Effect Model",
                                 RMSE = model_1_rmse ))
rmse_results %>% knitr::kable()


#To further improve the movel we will now ad the effect of users.  The effect aims to account for users that generally rate movies high or lower.

#"The User Effect"
# Improve model by adding B-u "User effect"
# lm(rating ~ as.factor(edx) + as.factor(userId)) # This will take a very long time if you run it.
user_avgs <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred
model_2_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie + User Effects Model",  
                                 RMSE = model_2_rmse ))
rmse_results %>% knitr::kable()

  
###Conclusion
#By accounting for Average movie scores as well as average User ratings we were able to achieve a RMSE below the target of 0.8649.

#final model results
rmse_results %>% knitr::kable()

#This project was a great way to finish this program and gave me some great ideas about the upcoming choose your own project!  Congrats for making it this far guys! :)

