import time 
import concurrent.futures as c 
import pandas as pd 
import requests 

# Take data from file
data = pd.read_csv("./Data/tmdb_5000_credits.csv")

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
        rating = response.json()['vote_average'] # get movie rating
        genresList = [genre["name"] for genre in response.json().get("genres", [])] # parses JSON -> dict object 
        # then get all genres , else empty list []

        for genre in all_genres:
            movie_data[genre] = int(genre in genresList) # convert boolean to int 1 or 0 for True or False 
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
# movies_revenues = {}
# Use ProcessPoolExecutor for faster data fetching
def main(): 

    start = time.perf_counter()
    with c.ProcessPoolExecutor() as executor:
        results = executor.map(fetch_genres, movieID,chunksize=200)
        # results2 = executor.map(fetch_revenue, movieID)
        for movie_id, movie_data in results:
            if movie_data:
                movies_ratings[movie_id] = movie_data
        # for movie_id, movie_data in results2: 
        #     if movie_data: 
        #         movies_revenues[movie_id] = movie_data
    end = time.perf_counter()
    print(f"Data fetching completed in {end - start:.4f} seconds")

    # from_dict() method: more info when hover over 
    test = pd.DataFrame.from_dict(movies_ratings,orient="index") # orient="index" means the keys of the dictionary will be the row labels
                                                                 # so movie_id will be row labels and columns will be genres + vote_average
    # test2 = pd.DataFrame.from_dict(movies_revenues, orient="index")
    print(test.head(10))

    # NOTE: write Data to excel file 
    # test = pd.DataFrame(movies_ratings) 
    # test.to_excel("movies_genre_data2.xlsx", index_label="movie_id" , sheet_name="Movies Revenue")
    # with pd.ExcelWriter('movie_data.xlsx', engine='openpyxl') as writer:
    #     test.to_excel(writer, sheet_name='Movie Genres', index_label="movie_id")
        # test2.to_excel(writer, sheet_name='Movie Revenues', index_label="movie_id")
if __name__ == "__main__": 
    main() 
