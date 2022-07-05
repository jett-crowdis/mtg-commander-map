# Lucky Paper Commander Map

This repository serves as the location for the code to make the [Lucky Paper Commander Map](https://luckypaper.co/resources/commander-map) from [EDHREC](https://edhrec.com/) data.

# Required Files

There are a number of files that are required to make the Commander Map. They are currently as follows:

- `duplicates.txt`: a text file that lists cards that are commonly duplicated in decklists. These are used on site to warn users that when they download a list containing these cards, it is likely to be inaccurate.
- `edhrec-to-scryfall.json`: a json file that maps incorrect card names from EDHREC to their correct names in [Scryfall](https://scryfall.com/). The code to make the map will intentionally fail when it finds a card that does not have a [Scryfall](https://scryfall.com/) entry. You will need to provide the correct Scryfall name for these cards in this mapping.
- `theme-tribe-override.csv`: overrides for tribe and theme slugs and images. `internal_slug` is used for folder names in the data the site uses. `edhrec_slug` gives the appropriate slug for [EDHREC](https://edhrec.com/) links. `name` gives the name of the theme or tribe on site (visible to users). `image` provides a card image associated with each tribe or theme. If no value is provided for a column, the code uses informative defaults: for tribes, the plural version of `internal_slug`, and for themes, the most defining card for that theme's submap.

# Creating the Commander Map - Code

The logic and explanations for the Commander Map can be found in `commander-map.ipynb`. This notebook excludes a number of tedious formatting steps, so it does _not_ generate the data required for the site. It is primarily a way to explain the code to Lucky Paper readers (and potentially) debug.

The data for the site can be generated in two ways:

**1) Locally**

- To create the map locally, Python ≥ 3.7.9 is required
- Use `pip install -r requirements.txt` to install all necessary packages.
- Run the bash script using the following syntax:

```
bash make-map.sh {edhrec_data} {date}
```

e.g. `bash make-map.sh https://fake-edhrec-link.com 2022-05-13`. The code accomplishes the 4 main tasks in creating the Commander Map:

1. Preprocesses the EDHREC data and calculates submap defining cards and traits (e.g. common cards in Krenko decks).
2. Embeds the entire EDHREC data into 2 dimensions using [UMAP](https://umap-learn.readthedocs.io/en/latest/api.html). These are the main map coordinates.
3. Clusters the entire EDHREC data using [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html), then calculates defining cards and traits for each cluster. These are the main map clusters and their associated defining features.
4. For each submap, embeds data from those decks into 2 dimensions using [UMAP](https://umap-learn.readthedocs.io/en/latest/api.html), clusters the data using [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html), then calculates defining cards and traits for each cluster. This yields the coordinates of each submap and its associated clusters.

Because the data is extremely large (over 1.2 million decks) and there are _many_ submaps (a few thousand), you should expect this code to take at least a day to run. It may also fail if your computer doesn't have enough RAM, as embedding the entire map data is extremely memory intensive.

**2) In the Cloud**

The much easier thing to do is to generate the map in the cloud. I don't have experience with native Google or AWS virtual machines, but I do have experience with a niche Google platform traditionally used for genomics: [Terra](https://app.terra.bio/). It's not hard to set up, as you just need a google cloud account and billing project. You can then configure a `.wdl` file that specifies all your inputs and run the "workflow" to create and upload the map. It takes care of requesting Google virtual machines with the appropriate resource amounts and uses [Docker](https://www.docker.com/) images to allow those virtual machines to run the code. `generate_commander_map.wdl` and `Dockerfile` (and its accompany docker image `jcrowdis/commander-map:version`) were created for this purpose. It costs about $3 to generate the map.
