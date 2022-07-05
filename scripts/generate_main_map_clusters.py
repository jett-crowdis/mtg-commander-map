import load
import numpy as np
import pandas as pd
import argparse
import warnings
import json
import os

import load
import map_classes

# ignore warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Determines the clusters of the Lucky Paper Commander Map'
    )

    parser.add_argument('--preprocessed_data', type=str,
                        help='directory containing preprocessed data')
    parser.add_argument('--include_commanders',
                        default=False, action='store_true')

    args = parser.parse_args()
    data_dir = args.preprocessed_data
    include_commanders = args.include_commanders

    print('\nGenerating clusters for Commander Map...\n' + '-'*40 + '\n')
    print('Loading magic cards...', end='')
    magic_cards = load.load_magic_cards()
    print(f'loaded {len(magic_cards)} cards')

    print('\nLoading preprocessed data...', end='')
    commander_decks_path = f'{data_dir}/map_intermediates/commander-decks.csv'
    sparse_decklist_path = f'{data_dir}/map_intermediates/sparse-decklists.npz'
    sparse_columns_path = f'{data_dir}/map_intermediates/sparse-columns.txt'
    date_matrix_path = f'{data_dir}/map_intermediates/date-matrix.csv'
    ci_matrix_path = f'{data_dir}/map_intermediates/coloridentity-matrix.csv'
    trait_mapping_path = f'{data_dir}/trait-mapping.csv'

    commander_decks = pd.read_csv(commander_decks_path, dtype={
                                  'savedate': str}).fillna('')

    decklist_matrix, card_idx_lookup = load.load_decklists(
        sparse_decklist_path, sparse_columns_path)
    date_matrix, deck_date_idx_lookup, card_date_idx_lookup = load.load_date_matrix(
        date_matrix_path, commander_decks, card_idx_lookup, magic_cards)
    ci_matrix, deck_ci_idx_lookup, card_ci_idx_lookup = load.load_ci_matrix(
        ci_matrix_path, commander_decks, card_idx_lookup, magic_cards)

    # Create the Commander Map object
    cdecks = load.load_cdecks(
        commander_decks, decklist_matrix, card_idx_lookup)
    commander_map = map_classes.CommanderMap(
        decklist_matrix, commander_decks, cdecks)

    # package in reference lookups
    commander_map.date_matrix = date_matrix
    commander_map.ci_matrix = ci_matrix

    # key = card, value = column index in decklist_matrix
    commander_map.card_idx_lookup = card_idx_lookup
    # key = deck_id, value = column index in date_matrix
    commander_map.deck_date_idx_lookup = deck_date_idx_lookup
    # key = card, value = row index in date_matrix
    commander_map.card_date_idx_lookup = card_date_idx_lookup
    # key = deck_id, value = column index in color identity matrix
    commander_map.deck_ci_idx_lookup = deck_ci_idx_lookup
    # key = card, value = row index in ci_matrix
    commander_map.card_ci_idx_lookup = card_ci_idx_lookup
    print('done')

    trait_mapping = load.build_trait_mapping(
        trait_mapping_path=trait_mapping_path)

    # run the first UMAP - 6D embedding
    print('\nRunning the 6D main map embedding...', end='')
    commander_map.reduce_dimensionality(
        method='UMAP', n_dims=6, coordinates=False, metric='jaccard', n_neighbors=25, min_dist=0)

    # Export the dimensions
    export_embedding = pd.DataFrame(commander_map.cluster_embedding).to_csv(
        'map-embedding.csv', index=False)
    print('done')

    cluster_labels = commander_map.cluster_decks(
        method='HDBSCAN', min_cluster_size=15)
    commander_map.assign_unclustered()

    print('\nCalculating cluster traits of main map...', end='')
    cluster_traits = commander_map.get_cluster_traits()
    print('done')

    print('\nVerify the cluster traits below. See the defining traits for the most Krenko-focused cluster:\n')
    krenko_clusters = cluster_traits.query(
        'value == "Krenko, Mob Boss"').sort_values(by='percent', ascending=False)
    krenko_cluster = krenko_clusters['clusterID'].iloc[0]
    print(cluster_traits[cluster_traits['clusterID'] == krenko_cluster])

    print('\nCalculating defining cards...')
    commander_map.get_cluster_card_counts(
        color_rule='ignore', include_commanders=include_commanders, verbose=True)

    defining_cards = commander_map.get_defining_cards(
        include_synergy=True, n_scope=1000, verbose=True)

    print('done')

    # Do some checking so that we can verify in the logs
    print('\nVerify the clusters below. See the defining cards for Winota-focused clusters:\n')
    winota_clusters = cluster_traits.query(
        'value == "Winota, Joiner of Forces"').sort_values(by='percent', ascending=False)
    print(winota_clusters.head(3))

    winota_defining_cards = defining_cards[defining_cards['clusterID'].isin(
        winota_clusters.head(3)['clusterID'])]
    print('\n', winota_defining_cards.groupby('clusterID').head(4))

    # calculate the average decklist of each cluster
    average_decklists = commander_map.calculate_average_decklists(verbose=True)

    print('\nExporting cluster files...', end='')
    cluster_json = commander_map.jsonify_map(
        magic_cards, trait_mapping=trait_mapping)

    with open('edh-map-clusters.json', "w") as out_json:
        json.dump(cluster_json, out_json)

    commander_map.commander_decks[['clusterID']].to_csv(
        f'commander-map-clusters.csv', index=False)

    # also export each cluster individually
    os.mkdir('clusters')
    for clust, c_json in enumerate(cluster_json):
        with open(f'clusters/{clust}.json', "w") as out_json:
            json.dump(c_json, out_json)
    print('done')
