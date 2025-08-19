'''
PART 2: SIMILAR ACTROS BY GENRE

Using the imbd_movies dataset:
- Create a data frame, where each row corresponds to an actor, each column represents a genre, and each cell captures how many times that row's actor has appeared in that column’s genre 
- Using this data frame as your “feature matrix”, select an actor (called your “query”) for whom you want to find the top 10 most similar actors based on the genres in which they’ve starred 
- - As an example, select the row from your data frame associated with Chris Hemsworth, actor ID “nm1165110”, as your “query” actor
- Use sklearn.metrics.DistanceMetric to calculate the euclidean distances between your query actor and all other actors based on their genre appearances
- - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.DistanceMetric.html
- Output a CSV continaing the top ten actors most similar to your query actor using cosine distance 
- - Name it 'similar_actors_genre_{current_datetime}.csv' to `/data`
- - For example, the top 10 for Chris Hemsworth are:  
        nm1165110 Chris Hemsworth
        nm0000129 Tom Cruise
        nm0147147 Henry Cavill
        nm0829032 Ray Stevenson
        nm5899377 Tiger Shroff
        nm1679372 Sudeep
        nm0003244 Jordi Mollà
        nm0636280 Richard Norton
        nm0607884 Mark Mortimer
        nm2018237 Taylor Kitsch
- Describe in a print() statement how this list changes based on Euclidean distance
- Make sure your code is in line with the standards we're using in this class
'''

#Write your code below
import pandas as pd
from sklearn.metrics import DistanceMetric, pairwise_distances
import numpy as np
import datetime

def sag():
    # Load dataset (read directly from the source URL)
    df = pd.read_json(
        "https://github.com/cbuntain/umd.inst414/blob/main/data/imdb_movies_2000to2022.prolific.json?raw=true",
        lines=True
    )

    # Build feature matrix (actor x genre)
    actor_genre = {}
    for _, row in df.iterrows():
        genres = row.get("genres", [])
        actors = row.get("actors", [])
        if not isinstance(actors, list):
            continue
        for a in actors:
            # Each "a" is like ["nm1165110", "Chris Hemsworth"]
            if isinstance(a, list) and len(a) == 2:
                actor_id, actor_name = a
                if actor_id not in actor_genre:
                    actor_genre[actor_id] = {"name": actor_name, "counts": {}}
                for g in genres:
                    actor_genre[actor_id]["counts"][g] = actor_genre[actor_id]["counts"].get(g, 0) + 1

    # Create dataframe
    all_genres = sorted({g for v in actor_genre.values() for g in v["counts"]})
    data = []
    ids = []
    names = []
    for actor_id, v in actor_genre.items():
        row = [v["counts"].get(g, 0) for g in all_genres]
        data.append(row)
        ids.append(actor_id)
        names.append(v["name"])

    feature_matrix = pd.DataFrame(data, index=ids, columns=all_genres)
    feature_matrix["name"] = names

    # Query actor: Chris Hemsworth
    query_id = "nm1165110"
    if query_id not in feature_matrix.index:
        print(f"Query actor_id {query_id} not found in feature matrix.")
        return
    query_vec = feature_matrix.loc[query_id, all_genres].to_numpy().reshape(1, -1)

    # Distance metrics
    dist_metric_euclidean = DistanceMetric.get_metric("euclidean")  # as required by the prompt
    X = feature_matrix[all_genres].to_numpy()
    eucl_dists = dist_metric_euclidean.pairwise(query_vec, X)[0]

    # Use pairwise_distances for cosine (works across sklearn versions)
    cos_dists = pairwise_distances(query_vec, X, metric="cosine")[0]

    feature_matrix["euclidean"] = eucl_dists
    feature_matrix["cosine"] = cos_dists

    # Get top 10 by cosine (skip the query itself at distance 0)
    top10_cosine = feature_matrix.sort_values("cosine")[1:11][["name", "cosine"]]

    # Save CSV
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = f"data/similar_actors_genre_{ts}.csv"
    top10_cosine.to_csv(outpath, index=True, index_label="actor_id")

    # Print
    print("Top 10 similar actors to Chris Hemsworth by cosine distance:")
    print(top10_cosine)

    top10_euclidean = feature_matrix.sort_values("euclidean")[1:11][["name", "euclidean"]]
    print("\nIf you use Euclidean instead of Cosine, the list changes to:")
    print(top10_euclidean)
