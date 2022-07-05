#!/bin/bash

set -e

# 1 - edhrec data link; 2 - date
edhrec_date="$(basename -- $1)"

# get and unpack edhrec data
wget $1
mkdir data
tar -zxf $edhrec_date -C data

# preprocess the data. no_split passed because we cannot parallelize the submaps.
python -u scripts/preprocess_data.py --data_dir data \
                                     --edhrec_to_scryfall edhrec-to-scryfall.json \
                                     --out_dir $2 \
                                     --duplicates duplicates.txt \
                                     --slug_image_override theme-tribe-override.csv \
                                     --include_commanders \
                                     --n_split 1

# generate the main map clusters
python -u scripts/generate_main_map_coordinates.py --preprocessed_data $2 \
                                                   --n_neighbors 30

# generate the main map coordinates
python -u generate_main_map_clusters.py --preprocessed_data $2 \
                                        --duplicates duplicates.txt \
                                        --include_commanders

# generate all of the submaps - this takes by far the longest
python -u scripts/generate_submaps.py --preprocessed_data $2 \
                                      --submap_file grouped_data.csv \
                                      --duplicates duplicates.txt \
                                      --include_commanders

# move decks into the data folder - comment this line out if no decks
mv decks $2

# combine main map coordinates and clusters
awk -F "," '{print $NF}' commander-map-clusters.csv > clusters.csv
paste -d , commander-map-coordinates.csv clusters.csv > $2/edh-map.csv

# move the cluster data into our output
mv clusters $2
mv edh-map-clusters.json $2/map_intermediates/edh-map-clusters.json