import scipy
import numpy as np
import pandas as pd
import argparse
import warnings
import json
import inflect
from collections import defaultdict

import map_classes
import card
import load
import calculate

# ignore warnings
warnings.filterwarnings('ignore')


def create_submaps(grouped_data, commander_map, magic_cards, include_commanders,
                   trait_mapping, override_dict, out_dir):
    '''Given grouped data and input data, create submaps and export their data to out_dir

    Params:
    -------
    grouped_data: df, dataframe of submaps
    commander_map: CommanderMap object
    magic_cards: dict, cardname to magic_properties
    include_commanders: bool, whether commanders should be included in the decklist matrix
    trait_mapping: dict, to map traits to integers
    out_dir: str, output directory

    Returns:
    --------
    0
    '''

    # set seed
    np.random.seed(0)

    for index in grouped_data.index:
        category, value = grouped_data.loc[index, ['category', 'value']]

        # first, we handle the name of the folder. In all cases except color identity, this should match the EDHREC slug.
        # This varies depending on the category.
        if category == 'colorIdentityID':
            out_value = value.lower() if value != '' else 'c'
        elif category == 'themeID':
            out_value = override_dict['themeID'].get(
                value, {}).get('edhrec_slug', value).lower()
        elif category == 'tribeID':
            override_value = override_dict['tribeID'].get(
                value, {}).get('edhrec_slug', '').lower()
            if override_value != '':
                out_value = override_value
            else:
                p = inflect.engine()
                out_value = p.plural(value).lower()
        elif category in ['commander-partnerID', 'partnerID']:
            out_value = card.kebab(value.lower())

        output_dir = f'{out_dir}/submaps/{category}/{out_value}'

        # partners are accessed through commander-partnerID
        if category == 'partnerID':
            output_dir = output_dir.replace('partnerID', 'commander-partnerID')

        # extract the appropriate data by subsetting the main map data to the category in question
        if category != 'partnerID':
            submap_decks = commander_map.commander_decks[commander_map.commander_decks[category] == value].reset_index(
                drop=True).copy()
        else:
            submap_decks = commander_map.commander_decks[(commander_map.commander_decks['commanderID'] == value) |
                                                         (commander_map.commander_decks['partnerID'] == value)].reset_index(drop=True).copy()

        # get the appropriate decklist matrix and cdecks as well
        submap_decklist_matrix = commander_map.decklist_matrix[submap_decks['deckID'].values, :]
        submap_cdecks = {
            deck_id: commander_map.cdecks[deck_id] for deck_id in submap_decks['deckID'].values}
        print(index, value, f'{len(submap_decks)} decklists')

        # remove cards that are absent in these decks and establish new lookups
        card_counts = submap_decklist_matrix.sum(axis=0).A1
        played_cards = np.array(list(commander_map.card_idx_lookup.keys()))[
            card_counts > 0]
        submap_decklist_matrix = submap_decklist_matrix[:, card_counts > 0]
        submap_card_idx_lookup = {cardname: i for i,
                                  cardname in enumerate(played_cards)}

        # establish the submap object
        commander_submap = map_classes.CommanderMap(
            submap_decklist_matrix, submap_decks, submap_cdecks)

        # most lookup references are the same
        commander_submap.date_matrix = date_matrix
        commander_submap.ci_matrix = ci_matrix
        # key = card, value = column index in decklist_matrix
        commander_submap.card_idx_lookup = submap_card_idx_lookup
        # key = deck_id, value = column index in date_matrix
        commander_submap.deck_date_idx_lookup = deck_date_idx_lookup
        # key = card, value = row index in date_matrix
        commander_submap.card_date_idx_lookup = card_date_idx_lookup
        # key = deck_id, value = column index in color identity matrix
        commander_submap.deck_ci_idx_lookup = deck_ci_idx_lookup
        # key = card, value = row index in ci_matrix
        commander_submap.card_ci_idx_lookup = card_ci_idx_lookup

        print('Running UMAP and clustering...')
        # run 2D embedding
        commander_submap.reduce_dimensionality(
            method='UMAP', n_dims=2, coordinates=True)

        # run embedding for clustering, then run clustering
        nneighbors, minclustersize = calculate.get_parameters(
            len(commander_submap.commander_decks))
        entropy, max_cluster_perc = 0, 1

        # we also exit if the minclustersize has dropped below one, as this throws an error
        num_iter = 0
        print('\n')
        while entropy < 1 and max_cluster_perc > 0.8 and minclustersize > 1:
            print(f'On iteration {num_iter}...')

            commander_submap.reduce_dimensionality(method='UMAP', n_dims=4, coordinates=False,
                                                   n_neighbors=nneighbors, min_dist=0)
            commander_submap.cluster_decks(
                method='HDBSCAN', min_cluster_size=minclustersize)
            commander_submap.assign_unclustered(n_neighbors=15)

            # next we assess the entropy and max cluster size
            cluster_sizes = commander_submap.commander_decks.groupby(
                ['clusterID']).size()
            cluster_sizes = cluster_sizes/cluster_sizes.sum()
            entropy = scipy.stats.entropy(cluster_sizes)
            max_cluster_perc = np.max(cluster_sizes)
            num_iter += 1

            # if there are less than 200 decks, we always exit
            if len(commander_submap.commander_decks) <= 200:
                entropy, max_cluster_perc = np.inf, 0

            # otherwise, we recluster after dropping the minimum cluster size
            else:
                minclustersize -= 1

        # get the defining traits
        drop_categories = [
            category] if category != 'commander-partnerID' else [category, 'coloridentityID']
        commander_submap.get_cluster_traits(drop_categories=drop_categories)
        commander_submap.get_cluster_card_counts(color_rule='ignore',
                                                 include_commanders=include_commanders,
                                                 verbose=True)

        # if there is only one cluster, synergy doesn't make any sense
        include_synergy = False if len(
            set(commander_submap.cluster_labels)) == 1 else True
        commander_submap.get_defining_cards(
            include_synergy=include_synergy, n_scope=200, verbose=True)
        commander_submap.calculate_average_decklists(
            verbose=True)

        # export all clusters
        cluster_json = commander_submap.jsonify_map(
            magic_cards, trait_mapping=trait_mapping)

        with open(output_dir + '/edh-submap-clusters.json', 'w') as outfile:
            json.dump(cluster_json, outfile)

        # export the submap coordinates - replace with ints for trait mapping
        commander_submap.commander_decks = calculate.replace_traits_with_ints(
            commander_submap.commander_decks, trait_mapping)

        # change some column names
        col_order = ['deckID', 'siteID', 'path', 'x', 'y', 'commanderID', 'partnerID',
                     'colorIdentityID', 'tribeID', 'themeID', 'price', 'clusterID']
        export_df = commander_submap.commander_decks[col_order].copy()
        export_df[['x', 'y']] = export_df[['x', 'y']].round(6)
        export_df.to_csv(output_dir + '/edh-submap.csv', index=False)

        print('\n')

    return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Generates the submaps for the Lucky Paper Commander Map'
    )

    parser.add_argument('--preprocessed_data', type=str,
                        help='directory containing preprocessed data')
    parser.add_argument('--submap_file', type=str,
                        help='csv for submaps to create')
    parser.add_argument('--include_commanders',
                        default=False, action='store_true')

    args = parser.parse_args()
    data_dir = args.preprocessed_data
    submap_file = args.submap_file
    include_commanders = args.include_commanders

    submap_file = pd.read_csv(submap_file).fillna('')

    print('\nCreating submaps for the Lucky Paper Commander Map...\n' + '-'*53 + '\n')
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
    print('done')

    print('\nCreating the Commander Map object...', end='')
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

    # we need to overwrite some of the folder names to match EDHREC slugs. trait-mapping contains this
    slug_override = pd.read_csv(trait_mapping_path).fillna('')
    override_dict = defaultdict(dict)
    for _, row in slug_override.iterrows():
        info_dict = {}
        category, internal_slug, edhrec_slug = row[[
            'category', 'internal_slug', 'folder_slug']]
        info_dict['edhrec_slug'] = edhrec_slug
        override_dict[category][internal_slug] = info_dict

    # load the mapping for traits
    trait_mapping = load.build_trait_mapping(
        trait_mapping_path=trait_mapping_path)

    print('\nCreating submaps...\n' + '-'*20)
    create_submaps(submap_file, commander_map, magic_cards, include_commanders,
                   trait_mapping, override_dict, data_dir)
    print('Finished exporting all submaps')
