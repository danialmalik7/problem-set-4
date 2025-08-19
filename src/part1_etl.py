'''
PART 1: ETL the dataset and save in `data/`

Here is the imbd_movie data:
https://github.com/cbuntain/umd.inst414/blob/main/data/imdb_movies_2000to2022.prolific.json?raw=true

It is in JSON format, so you'll need to handle accordingly and also figure out what's the best format for the two analysis parts. 
'''

import os
import pandas as pd
import json
from urllib.request import urlopen
from datetime import datetime

# Create '/data' directory if it doesn't exist
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(data_dir, exist_ok=True)

def download_data():
    """Download the IMDB movie data."""
    url = "https://github.com/cbuntain/umd.inst414/blob/main/data/imdb_movies_2000to2022.prolific.json?raw=true"
    try:
        with urlopen(url) as response:
            content = response.read().decode('utf-8')
            try:
                data = json.loads(content)
                return data
            except json.JSONDecodeError:
                data = []
                for line in content.strip().split('\n'):
                    if line.strip():
                        try:
                            movie = json.loads(line.strip())
                            data.append(movie)
                        except:
                            continue
                return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

def process_data(movies_data):
    """Process the movie data."""
    processed_data = []
    
    for movie in movies_data:
        movie_id = movie.get('id', f"movie_{len(processed_data)}")
        
        movie_info = {
            'movie_id': movie_id,
            'title': movie.get('title', ''),
            'year': movie.get('year', ''),
            'rating': movie.get('rating', ''),
            'genres': movie.get('genres', [])
        }
        
        actors = movie.get('actors', [])
        for actor_id, actor_name in actors:
            actor_record = movie_info.copy()
            actor_record['actor_id'] = actor_id
            actor_record['actor_name'] = actor_name
            processed_data.append(actor_record)
    
    return processed_data

def save_data(data):
    """Save data to CSV files."""
    df_main = pd.DataFrame(data)
    
    # Save main data
    main_filename = f"imdb_movies_actors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    main_path = os.path.join(data_dir, main_filename)
    df_main.to_csv(main_path, index=False)
    
    # Save network data
    df_network = df_main[['actor_id', 'actor_name', 'movie_id', 'title']].copy()
    network_filename = f"network_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    network_path = os.path.join(data_dir, network_filename)
    df_network.to_csv(network_path, index=False)
    
    # Save genre data
    df_genre = df_main[['actor_id', 'actor_name', 'genres']].copy()
    genre_filename = f"genre_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    genre_path = os.path.join(data_dir, genre_filename)
    df_genre.to_csv(genre_path, index=False)
    
    return main_path, network_path, genre_path

# Load datasets and save to '/data'
def etl():
    """Main ETL function."""
    print("Starting ETL...")
    
    movies_data = download_data()
    if not movies_data:
        return
    
    print(f"Downloaded {len(movies_data)} movies")
    
    processed_data = process_data(movies_data)
    print(f"Processed {len(processed_data)} records")
    
    main_path, network_path, genre_path = save_data(processed_data)
    print("ETL completed")

if __name__ == "__main__":
    etl()
