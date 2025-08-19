'''
You will run this problem set from main.py, so set things up accordingly
'''

import part1_etl as etl
import part2_network_centrality as nc
import part3_similar_actors_genre as sag

# Call functions / instanciate objects from the .py files
def main():
    # PART 1: Instanciate etl, saving the dataset in `./data/`
    etl.etl()

    # PART 2: Call functions/instanciate objects for the network centrality analysis
    nc.nc()

    # PART 3: Call functions/instanciate objects for similar actors by genre
    sag.sag()

if __name__ == "__main__":
    main()
