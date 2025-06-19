# Movie_Rating_Predictor
This project build a machine learning model to predict the movie ratings based on genres

**PROJECT REPORT** <br/>

## Collect Data Process
1. Collecting data 
    - [TMDB](https://developer.themoviedb.org/docs/getting-started), The Movie Database to access movies data
    - First to have an account and generate your own API key with given instructions by this link [Sign up](https://www.themoviedb.org/settings/api). The API key will be used to access movies information
    - When signing up, it will ask for application URL or the link of the project that you will use the movie database for, I did because I didn't have any link for the project: (http://127.0.0.1:5000/not/a/URL)
    
2. Download the [dataset](./tmdb_5000_credits.csv) to access movie id. Can be opened in excel to view the file
3. Set up **environment**: 
    - While collecting the data, I used the Ubuntu Linux subsystem for my Windows computer. Instead of downloading Python packages system-wide, I used virtual environment and downloaded the packages there 
    - Before creating the virtual environment, download the python3-venv by `sudo apt install python3.12-venv`, this is system-wide download
    - Create virtual environment by `python3 -m venv venv` 
    - Activate the environment by `source venv/bin/activate`
    - [Here](.\requirements.txt) are all the packages I used 
    - To download, use `pip install -r requirement.txt`, this downloads all the packages at the same time
    - Refer to my [Code](.\getData.py) here

## Analysis, all genres 
Preview of Data: ![alt text](.\Images\Data_Preview.png)
[Here](.\movie_data.xlsx) is the excel of the Genres and Ratings data

### Movies ratings and statistics f_0 analysis for linear regression model:
- ![alt text](.\Images\OLS_ratings.png)
- I split the original data set into the train set and test set with 80% and 20%. Both R^2 values are almost the same, meaning the model generalize the unseen or new data very well , low chance of overfitting 
- I got 0.109 for R^2 value for the train set
- I got 0.093 for R^2 value for the test set
- Movie ratings include many factors such as actors, movie contents, genres, and etc. Though the model explains about 10% of the variance for rating vote average based on the genres, it still matters in this context 
- For some of the genres have p-values smaller than the α=0.05, such as Animation, Drama, Comedy, Horror, etc. , they might contribute valuable information 

### Movies residuals analysis 
- ![alt text](.\Images\ratings_residual_and_histogram.png)
- The picture is for the first working process without removing the non-contributing features yet 
- The residual plot appear to be scattered about the 0 line rather than skewing or pattern-like. This indicates the variance is constant 
- There are some outliers that are outside the main cluster of the plot on the bottom, this might indicate that the model might overpredicted some movies ratings, leading to negative residual 
- The histogram shows bell-shaped(normal distribution) and a little bit skewed to the left, similarly to how there are some values on the bottom of the plot graph. 
- The sample average is 0 
- Summary: The residual and ANOVA shows that at least 1 of the genres has linear relationship with the ratings data 

## Analysis, eliminate genres that have p-values greater than α=0.05

**As the model has to train through all the genres, including those that are considered non-contributing. Now I want to redo that model with only contributing genres**

### Movie ratings and statistics f_0 after removing genres has large p-values:
- ![alt text](.\Images\adjusted_OLS_ratings.png)
- From the picture, there are not much of change, new R^2 is close the old one 0.107 and 0.109 
- Without non-contributing features, now the model can predict faster 
### Residual analysis 
- ![alt text](.\Images\adjusted_ratings_residual_plot.png)
- For the plot, it appears still scattered around 0 with some outliers on the bottom, similar to before removing the genres. This also indicates the variance is constant, no patterns or U shape 
- The histogram still shows the bell shape(which what we want) but with a bit skewed on the left side due to some overpredictions of ratings 
- The sample average is also 0 

**CONCLUSION:** The performance did not change based on the R^2 value after removing the non-contributing features
The genres-only model only explains about 10% of the actual ratings but still matters, and more features can be applied to predict the ratings data. The model shows there are still some linear relationship between the genres and the movie ratings

