import concurrent.futures as c 
import pandas as pd 
import requests 
# import numpy as np 
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.model_selection import train_test_split
#import statsmodels.api as sm

# Take data from file
data = pd.read_csv("tmdb_5000_credits.csv")

movieID = data["movie_id"] # access the movie ids 

allGenre_URL = "https://api.themoviedb.org/3/genre/movie/list?language=en"
genre_headers = {
    "accept": "application/json",
    "Authorization": "Your API key"
}

genre_response = requests.get(allGenre_URL, headers=genre_headers)

# NOTE: parse the JSON response to a Python dictionary 
genres  = genre_response.json()["genres"]  
all_genres = [genre["name"] for genre in genres]


# NOTE: fetch movies genres and ratings data  
def fetch_genres(movie_id):
    movie_data = {}
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?language=en-US"
    movie_headers = {
                "accept": "application/json",
                "Authorization": "Your API key"
            }
    try:
        response = requests.get(url, headers=movie_headers)
        response.raise_for_status()
        # data = response.json()
        rating = response.json()['vote_average']
        genresList = [genre["name"] for genre in response.json().get("genres", [])]

        for genre in all_genres:
            movie_data[genre] = int(genre in genresList)
        movie_data["vote_average"] = rating
        return str(movie_id), movie_data
    
    except Exception as e:
        print(f"Failed to fetch data for movie ID {movie_id}: {e}")
        return str(movie_id), None


# NOTE: fetch movie genres and revenues data 
def fetch_revenue(movie_id): 
    movie_data = {}
    URL = f"https://api.themoviedb.org/3/movie/{movie_id}?language=en-US"
    movie_headers = {
                "accept": "application/json",
                "Authorization": "Your API key"
            }
    try:
        response = requests.get(URL, headers=movie_headers)
        response.raise_for_status()
        revenue = response.json()['revenue']
        genresList = [genre["name"] for genre in response.json().get("genres", [])]
        for genre in all_genres:
            movie_data[genre] = int(genre in genresList)
        movie_data["revenue"] = revenue
        return str(movie_id), movie_data
    
    except Exception as e:
        print(f"Failed to fetch data for movie ID {movie_id}: {e}")
        return str(movie_id), None
    
# NOTE: Run with multiple processes to fetch data 
movies_ratings = {}
movies_revenues = {}
with c.ProcessPoolExecutor() as executor:
    results = executor.map(fetch_genres, movieID)
    results2 = executor.map(fetch_revenue, movieID)
    for movie_id, movie_data in results:
        if movie_data:
            movies_ratings[movie_id] = movie_data
    for movie_id, movie_data in results2: 
        if movie_data: 
            movies_revenues[movie_id] = movie_data

test = pd.DataFrame.from_dict(movies_ratings,orient="index")
test2 = pd.DataFrame.from_dict(movies_revenues, orient="index")


# NOTE: write Data to excel file 
# test = pd.DataFrame(movies_ratings) 
# test.to_excel("movies_genre_data2.xlsx", index_label="movie_id" , sheet_name="Movies Revenue")
with pd.ExcelWriter('movie_data.xlsx', engine='openpyxl') as writer:
    test.to_excel(writer, sheet_name='Movie Genres', index_label="movie_id")
    test2.to_excel(writer, sheet_name='Movie Revenues', index_label="movie_id")

