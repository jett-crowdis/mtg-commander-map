import load
import pandas as pd
import argparse
import warnings

import calculate
import load
import map_classes

# ignore warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Determines the x,y coordinates of the Lucky Paper Commander Map'
    )

    parser.add_argument('--preprocessed_data', type=str,
                        help='directory containing preprocessed data')
    parser.add_argument('--n_neighbors', type=int,
                        help='n_neighbors parameters for umap')

    args = parser.parse_args()
    data_dir = args.preprocessed_data
    n_neighbors = args.n_neighbors

    print('\nCreating the 2D coordinates for the Commander Map...\n' + '-'*50 + '\n')

    print('Loading preprocessed data...', end='')
    commander_decks_path = f'{data_dir}/map_intermediates/commander-decks.csv'
    sparse_decklist_path = f'{data_dir}/map_intermediates/sparse-decklists.npz'
    sparse_columns_path = f'{data_dir}/map_intermediates/sparse-columns.txt'
    trait_mapping_path = f'{data_dir}/trait-mapping.csv'

    commander_decks = pd.read_csv(commander_decks_path, dtype={
                                  'savedate': str}).fillna('')

    decklist_matrix, card_idx_lookup = load.load_decklists(
        sparse_decklist_path, sparse_columns_path)
    trait_mapping = load.build_trait_mapping(
        trait_mapping_path=trait_mapping_path)

    # Creating the Commander Map object
    commander_map = map_classes.CommanderMap(
        decklist_matrix, commander_decks, None)

    # run the first UMAP - 2D embedding
    print('\nRunning the 2D main map embedding...')
    commander_map.reduce_dimensionality(
        method='UMAP', metric='jaccard', coordinates=True, n_dims=2, n_neighbors=n_neighbors)
    commander_map.commander_decks = calculate.replace_traits_with_ints(
        commander_map.commander_decks, trait_mapping)

    # change some column names
    col_order = ['siteID', 'path', 'x', 'y', 'commanderID', 'partnerID',
                 'colorIdentityID', 'tribeID', 'themeID', 'price']
    export_df = commander_map.commander_decks.copy()[col_order]
    export_df[['x', 'y']] = export_df[['x', 'y']].round(6)
    export_df.to_csv('commander-map-coordinates.csv', index=False)
