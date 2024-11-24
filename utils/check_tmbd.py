import requests


def main():
    # TMDB API settings
    tmdb_access_token = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJkNGE1OGRmMzUxNWFhZmVlZWJmNTNiMjgxMDU2YjdkYSIsIm5iZiI6MTcyOTg3OTAzMy4yODExNTUsInN1YiI6IjY3MWJkYjRhMjY4NWNiNjU2M2MwZGY3NiIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.nEtAdJi-bRb81NmtL3of2gPfqjcPnGPFWPBgxWhNYEo"
    tmdb_api_key = "d4a58df3515aafeeebf53b281056b7da"

    # API endpoint and parameters
    url = "https://api.themoviedb.org/3/discover/movie"
    params = {
        "with_genres": "27",  # Horror genre
        "sort_by": "vote_average.desc",  # Sort by highest vote average
    }

    # Headers for authentication
    headers = {
        "Authorization": f"Bearer {tmdb_access_token}",
        "Content-Type": "application/json;charset=utf-8",
    }

    # Make the API call
    response = requests.get(url, params=params, headers=headers)

    # Check response
    if response.status_code == 200:
        movies = response.json()
        print("Top horror movies sorted by vote average:")
        for movie in movies["results"][:10]:  # Show top 10 movies
            print(f"- {movie['title']} ({movie['vote_average']})")
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        print("Response:", response.text)


if __name__ == "__main__":
    main()
