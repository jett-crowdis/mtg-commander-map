from unicodedata import decimal
import scipy.sparse
import numpy as np
import pandas as pd
import argparse
import json
import os
import warnings
import re
from collections import defaultdict
import inflect

import companion
import card
import map_classes
import load
import calculate

# ignore warnings
warnings.filterwarnings('ignore')


def load_edhrec_data(data_dir, edhrec_to_scryfall, magic_cards):
    '''Load EDHREC data from decks.csv and cards.csv

    Params:
    -------
    data_dir: str, the path to the folder containing decks.csv and cards.csv
    edhrec_to_scryfall: str, the path to the edhrec_to_scryfall json
    magic_cards: dict, output of load_magic_cards()

    Returns:
    --------
    commander_decks: pandas df, contains information on decks
    commander_deck_dict: dict, contains information on decks (including cards)
    total_seen_cards: set, all cards seen in dataset
    '''

    commander_decks = pd.read_csv(f'{data_dir}/decks.csv',
                                  usecols=['url', 'commander', 'savedate', 'tribe',
                                           'coloridentity', 'commander2', 'theme'],
                                  dtype={'savedate': str})
    print('\tLoaded', len(commander_decks), 'commander decks')

    # we rename some of the columns here. In general, our export format will be relying on IDs (integers) rather than
    # strings, so even though the conversion doesn't happen until later, we rename the columns now.
    rename_dict = {'commander': 'commanderID', 'tribe': 'tribeID', 'coloridentity': 'colorIdentityID',
                   'commander2': 'partnerID', 'theme': 'themeID'}
    commander_decks = commander_decks.rename(columns=rename_dict)

    # sort by url to improve UMAP consistency
    commander_decks = commander_decks.sort_values(
        by='url').reset_index(drop=True)

    # a dictionary to fix cardname typos
    with open(edhrec_to_scryfall) as f:
        edhrec_to_scryfall = json.load(f)

    # fix typos and fill in empty partner, tribe, and themes
    commander_decks[['commanderID', 'partnerID']] = commander_decks[[
        'commanderID', 'partnerID']].replace(edhrec_to_scryfall)
    commander_decks[['partnerID', 'tribeID', 'themeID']] = commander_decks[[
        'partnerID', 'tribeID', 'themeID']].fillna('')

    # combine colors into one string ({W,U,B} > 'WUB')
    # and reorder color identities to WUBRG
    commander_decks['colorIdentityID'] = commander_decks['colorIdentityID'].apply(
        lambda ci: ci[1:-1:2])
    color_order = ['W', 'U', 'B', 'R', 'G']

    def reorder_to_wubrg(ci): return ''.join(
        sorted(ci, key=lambda c: color_order.index(c)))

    commander_decks['colorIdentityID'] = commander_decks['colorIdentityID'].apply(
        reorder_to_wubrg)

    # add cards to the commander decks. We make use of a CommanderDeck class.
    print('\tLoading commanders and deck traits...', end='')
    commander_deck_dict = {}
    total_seen_cards = set()
    for index, row in commander_decks.iterrows():

        if index % 100000 == 0:
            print(str(index // 1000) + 'k', end='...')

        # store commander information
        cdeck = map_classes.CommanderDeck()
        cdeck.url = row['url']

        # set and fix typos of commander and partner
        cdeck.commander = row['commanderID']
        cdeck.partner = row['partnerID']

        # update seen cards with non-empty partners
        total_seen_cards.update(
            [c for c in [cdeck.commander, cdeck.partner] if c != ''])

        # add other metadata
        cdeck.colorIdentity = row['colorIdentityID']
        cdeck.theme = row['themeID']
        cdeck.tribe = row['tribeID']
        cdeck.date = row['savedate']
        cdeck.cards = []

        commander_deck_dict[cdeck.url] = cdeck
    print('done')

    print('\tAdding cards to decklists...', end='')
    with open(f'{data_dir}/cards.csv') as card_file:
        for i, line in enumerate(card_file):
            if i == 0:
                continue

            if i % 1e7 == 0:
                print(f'{int(i//1e6)}M', end='...')

            # each line is a card paired with a url
            line = line.strip('\n')
            bp = line.index(',')
            url, cardname = line[:bp], line[bp + 1:]

            # because EDHREC oddly uses a csv instead json to store card names,
            # we have to handle commas
            if cardname.startswith('"') and cardname.endswith('"'):
                cardname = cardname.strip('"').rstrip('"')

            # handle some misspellings and odd cases
            if cardname not in magic_cards:
                cardname = edhrec_to_scryfall.get(cardname, cardname)

            # if the card name does not match the "name" key in magic_cards, it's a card backside. We
            # overwrite with its frontside. The weird get syntax is needed in case the card is not in Scryfall,
            # as we'd rather show all those cards at once for debugging.
            cardname = magic_cards.get(cardname, {'name': cardname})['name']

            total_seen_cards.add(cardname)
            commander_deck_dict[url].cards.append(cardname)
    print('done')

    # This should be empty. If it's not, update edhrec-to-scryfall.json
    print('\tChecking that all cards and decks are valid...', end='')
    missing_cards = total_seen_cards - set(magic_cards)
    if len(missing_cards):
        raise ValueError(
            f'The following cards are invalid: {missing_cards}. You need to update edhrec-to-scryfall.json to include this card.')

    # we then re-deduplicate decklists, as sometimes the process of mapping
    # backsides to fronts can result in duplicates
    num_duplicates = 0
    dup_cards = set()
    for _, cdeck in commander_deck_dict.items():

        unique_cards = set(cdeck.cards)
        if len(cdeck.cards) != len(unique_cards):
            num_duplicates += 1
            dup_cards.update(
                [c for c in cdeck.cards if cdeck.cards.count(c) > 1])
            cdeck.cards = sorted(unique_cards)

    if num_duplicates > 0:
        print(f're-deduplicated {num_duplicates} decks.')
    else:
        print('done')

    return commander_decks, commander_deck_dict, total_seen_cards


def clean_commander_decks(commander_decks, commander_deck_dict,
                          magic_cards, total_seen_cards, include_commanders):
    '''Clean the EDHREC data. This function does four main things:
    1. Identify and redefine decks with companions
    2. Remove erroneous partners and reorder commander-partner to be alphabetical (except backgrounds)
    3. Remove invalid Meld cards
    4. Adds commanders to the decklist, if specified

    Params:
    -------
    commander_decks: df, dataframe containing commander deck info. Output of load_edhrec_data
    commander_deck_dict: dict, mapping id to CommanderDeck object. Output of load_edhrec_data
    magic_cards: dict, contains info about magic cards. Output of load.load_magic_cards()
    total_seen_cards: set, set of all seen cards in dataset. Output of load_edhrec_data
    include_commanders: bool, whether commanders should be included in the decklists.


    Returns:
    --------
    Objects edited to reflect the cleaned data:
    commander_decks: pandas df, contains information on decks
    commander_deck_dict: dict, contains information on decks (including cards)
    partners: list of partners present in the data, with backgrounds removed
    '''

    ##########################################
    ### Defining and organizing companions ###
    ##########################################

    print('\tDefining and organizing companions...', end='')
    companion_list = []
    for index, (url, cdeck) in enumerate(commander_deck_dict.items()):
        if index % 100000 == 0:
            print(str(index // 1000) + 'k', end='...')

        # calculate from the decklist whether a deck companions a card
        companion_assignment = companion.calculate_companion(
            cdeck, magic_cards)
        cdeck.companion = companion_assignment
        companion_list.append(cdeck.companion)

        # sometimes a companion is designated as a theme when the deck's commander
        # is the companion. That won't do
        if cdeck.commander == cdeck.companion:
            cdeck.theme = ''
            cdeck.companion = ''

        # the same is true for partners, except we let the companion slide
        if cdeck.partner == cdeck.companion:
            cdeck.partner = ''

        # if there is a companion, overwrite the theme if necessary
        elif cdeck.companion != '':
            cdeck.theme = re.split(',| ', cdeck.companion)[
                0].lower() + '-companion'

        # we then remove any commanders, partners, or companions present in the data already (they shouldn't be there)
        cdeck.cards = [cardname for cardname in cdeck.cards if cardname not in [
            cdeck.commander, cdeck.partner, cdeck.companion]]

        # and we add the commanders to the beginning of the list if specified
        if include_commanders:
            if cdeck.companion != '':
                cdeck.cards = [cdeck.companion] + cdeck.cards
            if cdeck.partner != '':
                cdeck.cards = [cdeck.partner] + cdeck.cards
            cdeck.cards = [cdeck.commander] + cdeck.cards

        assert len(cdeck.cards) == len(set(cdeck.cards))

    #################################
    ### Removing invalid partners ###
    #################################

    # First we identify the correct partners. This includes Partner With, Friends Forever, Background, and Choose a Background cards.
    # We do not check the legality of partner pairings.
    partners = [c for c, card_info in magic_cards.items(
    ) if 'Partner' in card_info.get('keywords', [])]
    friends_forever = [c for c, card_info in magic_cards.items(
    ) if '\nFriends forever' in card_info.get('oracle_text', '')]
    backgrounds = [c for c, card_info in magic_cards.items(
    ) if 'Background' in card_info.get('type_line', '')]
    choose_a_background = [c for c, card_info in magic_cards.items(
    ) if '\nChoose a Background' in card_info.get('oracle_text', '')]

    partners = partners + friends_forever + backgrounds + choose_a_background

    # identify decklists with incorrect partners - these are almost always companions
    bad_partners = commander_decks[(~commander_decks['partnerID'].isin(partners)) &
                                   (commander_decks['partnerID'] != '')]

    # remove these bad partners from the commander deck objects
    for url in bad_partners['url']:
        commander_deck_dict[url].partner = ''

    # before we remove these bad partners from the df, use them to calculate how many companion decklists we "recovered"
    bad_partner_set = set(bad_partners['partnerID'])
    recovered_companions = commander_decks[(np.array(companion_list) != '') &
                                           ((~commander_decks['themeID'].str.contains('companion')) &
                                            (~commander_decks['partnerID'].isin(bad_partner_set)))]
    print(f'recovered {len(recovered_companions)} companions')

    # we overwrite all existing themes using the companions
    commander_decks['themeID'] = [
        cdeck.theme for cdeck in commander_deck_dict.values()]
    commander_decks['companionID'] = companion_list

    # next we remove bad partners and companion "themes" (which are now specified by the companion column)
    print('\tRemoving invalid partners and reordering commanders and partners...', end='')

    # remove the bad partners from the commander decks
    commander_decks.loc[bad_partners.index, 'partnerID'] = ''

    # next we rearrange commanders and partners to be alphabetical (this is how EDHREC does it)
    # we do not do this for backgrounds, as we assume those are commander > background
    rearranged_indices = []
    rearranged_data = []

    for index, (url, commander, partner) in enumerate(commander_decks[['url', 'commanderID', 'partnerID']].values):

        if partner == '' or partner in backgrounds:
            continue

        # sort the commander and partner values
        commander, partner = sorted([commander, partner])
        commander_deck_dict[url].commander = commander
        commander_deck_dict[url].partner = partner

        # store for overwriting dataframe
        rearranged_indices.append(index)
        rearranged_data.append([commander, partner])

    rearranged_indices = np.array(rearranged_indices)
    rearranged_data = np.array(rearranged_data)

    # and overwrite all the commander and partner assignments
    commander_decks.loc[rearranged_indices, [
        'commanderID', 'partnerID']] = rearranged_data
    commander_partners = [c + ' + ' + p if p != '' else c for c,
                          p in commander_decks[['commanderID', 'partnerID']].values]
    commander_decks['commander-partnerID'] = commander_partners
    print('done')

    ##################################
    ### Removing invalid med cards ###
    ##################################

    print('\tRemoving invalid meld cards...', end='')

    # First we build which cards are combined meld cards
    combined_meld_cards = {}
    for cardname in total_seen_cards:
        card_info = magic_cards[cardname]
        if card_info.get('layout') == 'meld':
            components = []
            for meld_component in card_info['all_parts']:
                if meld_component['component'] == 'meld_part':
                    components.append(meld_component['name'])
                if meld_component['component'] == 'meld_result':
                    meld_result = meld_component['name']

            if meld_result == cardname:
                combined_meld_cards[cardname] = components

    # next we go through the decklist and fix the card lists
    meld_removed = 0
    for url, cdeck in commander_deck_dict.items():

        # if the combined meld card is in the list, we remove it and check if we need to add the components
        present_combined_melds = [
            cardname for cardname in combined_meld_cards if cardname in cdeck.cards]

        for combined_name in present_combined_melds:

            # check to see which card components are missing from the card list
            components = combined_meld_cards[combined_name]
            c_absent = [c for c in components if c not in cdeck.cards]

            # if one card is missing, add it. If both are present, we don't need to add anything,
            # and if both are missing, we assume the decklist wouldn't have the card
            if len(c_absent) == 1:
                commander_deck_dict[url].cards += c_absent

            # remove the combined meld
            commander_deck_dict[url].cards = [
                c for c in cdeck.cards if c != combined_name]
            meld_removed += 1

    print(f'fixed {meld_removed} meld cards')

    # remove partners that are not in the commander decks and backgrounds
    total_commanders = set(commander_decks['commanderID']).union(
        commander_decks['partnerID'])
    partners = sorted(set(partners) & set(total_commanders) - set(backgrounds))

    return commander_decks, commander_deck_dict, partners


def calculate_prices(commander_decks, commander_deck_dict, include_commanders, magic_cards):
    '''Calculates the price of each decklist, based on the cheapest version of each card.

    Params:
    -------
    commander_decks: df, dataframe containing commander deck info. Output of load_edhrec_data
    commander_deck_dict: dict, mapping id to CommanderDeck object. Output of load_edhrec_data
    include_commanders: bool, whether commanders should be included in the price calculation.
    magic_cards: dict, contains info about magic cards. Output of load.load_magic_cards()

    Returns:
    --------
    Objects edited to reflect updated prices:
    commander_decks: pandas df, contains information on decks
    commander_deck_dict: dict, contains information on decks (including cards)
    '''
    print('\tCalculating prices...', end='')

    # now we add prices
    for i, cdeck in enumerate(commander_deck_dict.values()):

        if i % 100000 == 0:
            print(str(i // 1000) + 'k', end='...')

        # calculate the price based on minimum price of each card
        cdeck.calculate_price(magic_cards, include_commanders)

    # add the price to the dataframe
    commander_decks['price'] = commander_decks['url'].apply(
        lambda url: commander_deck_dict[url].price).astype(int)

    print('done')

    return commander_decks, commander_deck_dict


def filter_commander_decks(commander_decks, commander_deck_dict):
    '''Filters the commander decks, removing duplicates and
    requiring at least one non-commander card. Also adds a deckID column
    to the commander_decks dataframe, corresponding to keys in the
    return cdeck_subset.

    Params:
    -------
    commander_decks: df, dataframe containing commander deck info. Output of load_edhrec_data
    commander_deck_dict: dict, mapping id to CommanderDeck object. Output of load_edhrec_data

    Returns:
    --------
    Objects edited to reflect updated prices:
    commander_decks: pandas df, contains information on decks
    cdeck_subset: dict, mapping deck to CommanderDeck object
    '''

    print('\tFiltering decks...', end='')
    cdeck_subset = []
    cdeck_df_hashes = set()
    cdeck_urls = []

    for cdeck in commander_deck_dict.values():

        # we handle duplicates here
        unique_cards = sorted(list(set(cdeck.cards)))
        deck_hash = hash('-'.join(unique_cards))

        # if we've already seen this exact decklist before, continue
        if deck_hash in cdeck_df_hashes:
            continue
        else:
            cdeck_df_hashes.add(deck_hash)

        # check that the deck contains at least one non-commander card
        num_commanders = 1 + (cdeck.partner != '')
        if len(cdeck.cards) > num_commanders:
            cdeck_subset.append(cdeck)
            cdeck_urls.append(cdeck.url)

    # we subset our commander decks to those that are included.
    commander_decks = commander_decks.set_index(
        'url').loc[cdeck_urls].reset_index()
    commander_decks['deckID'] = range(len(commander_decks))

    # and convert cdeck_subset into a dictionary
    cdeck_subset = dict(zip(range(len(cdeck_urls)), cdeck_subset))

    # we also add this to the cdeck itself
    for deckid, cdeck in cdeck_subset.items():
        cdeck.deckid = deckid

    print(len(cdeck_subset), 'decks after filtering')
    return commander_decks, cdeck_subset


def build_decklist_matrix(cdeck_subset, magic_cards, out_dir):
    '''Builds the sparse matrix of decklists for UMAP input.

    Params:
    -------
    cdeck_subset: dict, dict of n CommanderDeck objects to be included
    magic_cards: dict, output of load_magic_cards
    out_dir: str, output directory

    Returns:
    --------
    card_idx_lookup: dict, column index lookup for cards
    decklist_matrix: (n x m) sparse matrix of decklists
    '''

    print('\tBuilding card indices...', end='')

    # card_idx_lookup is the lookup dictionary for a card's index
    # duplicate cards stores the back of cards and their index
    card_idx_lookup = {}
    duplicate_cards = {}

    # the column index of the card
    index = 0

    total_seen_cards = set()
    for cdeck in cdeck_subset.values():
        total_seen_cards.update(set(cdeck.cards))

    # we go through the cards in alphabetical order to preserve ordering for UMAP
    card_idx_lookup = {c: i for i, c in enumerate(
        sorted(list(total_seen_cards)))}

    print('done')
    print('\tBuilding decklist matrix...', end='')
    decklist_matrix_rows = []
    decklist_matrix_data = []

    for deckid, cdeck in cdeck_subset.items():
        if deckid % 100000 == 0:
            print('{}k...'.format(deckid // 1000), end='')

        # build the sparse entry of the decklist
        row = []
        for cardname in set(cdeck.cards):
            if cardname in card_idx_lookup:
                card_idx = card_idx_lookup[cardname]
            elif cardname in duplicate_cards:
                card_idx = duplicate_cards[cardname]
            else:
                continue

            row.append(card_idx)

        decklist_matrix_rows.append(row)
        decklist_matrix_data.append([1]*len(row))

    # construct sparse matrix from rows and data using lil format
    decklist_matrix = scipy.sparse.lil_matrix(
        (len(decklist_matrix_rows), len(card_idx_lookup)), dtype=np.float32)
    decklist_matrix.rows = np.array(decklist_matrix_rows, dtype=object)
    decklist_matrix.data = np.array(decklist_matrix_data, dtype=object)
    decklist_matrix = decklist_matrix.tocsr()
    print('done')

    # we'll save the sparse decklist matrix
    scipy.sparse.save_npz(
        f'{out_dir}/map_intermediates/sparse-decklists.npz', decklist_matrix)

    with open(f'{out_dir}/map_intermediates/sparse-columns.txt', 'w') as outfile:
        outfile.write('\n'.join(list(card_idx_lookup.keys())))

    return card_idx_lookup, decklist_matrix


def define_grouped_data(commander_map, partners, n_split):
    '''Takes in a dataframe of commander decks and returns the submaps that need to be created.

    Params:
    -------
    commander_map: CommanderMap object
    partners: list of partners present in the data, with backgrounds removed
    n_split: int, how many splits of the data to make

    Returns:
    --------
    grouped_data: df, the dataframe of grouped cards
    '''

    grouped_data = []
    fields = ['commander-partnerID', 'tribeID', 'themeID', 'colorIdentityID']
    for col in fields:
        subdata = commander_map.commander_decks.groupby(col).size().to_frame(
            'count').reset_index().sort_values(by='count', ascending=False)
        subdata['category'] = col
        subdata = subdata.rename(columns={col: 'value'})
        grouped_data.append(subdata)

    grouped_data = pd.concat(grouped_data).reset_index(
        drop=True).sort_values(by='count', ascending=False)
    grouped_data = grouped_data[['category',
                                 'value', 'count']].reset_index(drop=True)

    # we need to drop empty tribes and themes
    empty_tribes_theme = grouped_data[(grouped_data['category'].isin(['tribeID', 'themeID'])) &
                                      (grouped_data['value'] == '')].index
    grouped_data = grouped_data.drop(
        index=empty_tribes_theme).reset_index(drop=True)

    # next we need to handle partners. This is... tedious
    partner_groups = []
    for p in partners:
        all_p_decks = (commander_map.commander_decks['commanderID'] == p) | \
                      (commander_map.commander_decks['partnerID'] == p)

        if sum(all_p_decks):
            partner_groups.append(['partnerID', p, sum(all_p_decks)])

    all_p_decks = pd.DataFrame(partner_groups,  columns=[
                               'category', 'value', 'count'])

    # combine into the dataframe
    grouped_data = pd.concat(
        [grouped_data, all_p_decks]).reset_index(drop=True)
    grouped_data = grouped_data.sort_values(by='count', ascending=False)

    # drop instances where the partner is alone
    grouped_data = grouped_data[~((grouped_data['category'] == 'commander-partnerID') &
                                  (grouped_data['value'].isin(partners)))]

    # sort and drop index
    grouped_data = grouped_data.sort_values(
        by='count', ascending=False).reset_index(drop=True)

    # if we are to split the submap generation process, we need to store which submaps are in each division
    if n_split > 1:

        submap_files = defaultdict(list)
        for index, row in grouped_data.iterrows():
            submap_index = index % n_split
            submap_files[submap_index].append(row)

        for i, submap_f in submap_files.items():
            submap_f = pd.DataFrame(submap_f)
            submap_f.to_csv(
                f'grouped_data_{i}.csv', index=False)

        with open('indices.txt', 'w') as outfile:
            outfile.write('\n'.join([str(i) for i in range(n_split)]))

    else:
        grouped_data.to_csv(f'grouped_data.csv', index=False)

    return grouped_data


def calculate_whole_submap_traits(commander_map, magic_cards, include_commanders, trait_mapping_df,
                                  slug_image_override, out_dir):
    '''Calculates and exports whole submap data--defining traits and cards

    Params:
    -------
    commander_map: CommanderMap object
    magic_cards: dict, cardname to magic properties
    include_commanders: bool, whether or not commanders are in the decklist matrix
    trait_mapping_df: df, a dataframe for mapping traits
    slug_image_override: df, a dataframe of slug and image overrides
    out_dir: str, output directory

    Returns:
    --------
    trait_mapping_slug_images: a dataframe that connects folder names, 
        shown names, and slugs for EDHREC for all traits.
    '''

    # define the data that is going to be used
    fields = ['tribeID', 'themeID', 'commander-partnerID',
              'colorIdentityID']

    # we need to overwrite some of the folder names to match EDHREC slugs, as well as overwrite some of the images and names.
    # We provide a file that does this in most cases
    override_dict = defaultdict(dict)
    for _, row in slug_image_override.iterrows():
        info_dict = {}
        category, internal_slug = row[['category', 'internal_slug']].values
        if row['edhrec_slug'] != '':
            info_dict['edhrec_slug'] = row['edhrec_slug']
        if row['name'] != '':
            info_dict['name'] = row['name']
        if row['image'] != '':
            info_dict['image'] = row['image']
        override_dict[category][internal_slug] = info_dict

    override_dict = dict(override_dict)

    # this variable keeps track of images and nanes for each trait
    trait_images_names = []

    # convert the trait_mapping_df into a lookup dictionary for ids
    trait_mapping = load.build_trait_mapping(trait_mapping_df=trait_mapping_df)

    # for each category, calculating the defining traits and cards
    for cat in fields:
        if not os.path.isdir(out_dir + f'/submaps/{cat}'):
            os.mkdir(out_dir + f'/submaps/{cat}')

        # copy the commander_map object with original references
        commander_submap = commander_map.copy_with_ref(commander_map.decklist_matrix, commander_map.commander_decks,
                                                       commander_map.cdecks)

        # first, we map categories to clusters
        cat_values = commander_submap.commander_decks[cat].value_counts()
        cat_to_cluster = dict(zip(cat_values.index, range(len(cat_values))))
        cluster_to_cat = {v: k for (k, v) in cat_to_cluster.items()}

        print(cat, ':', len(cat_values), 'submaps')

        # we overwrite the commander_map object with new "cluster" assignments
        commander_submap.commander_decks['clusterID'] = commander_submap.commander_decks[cat].apply(
            lambda value: cat_to_cluster[value])

        # now that we've declared clusters, we simply get the cluster defining cards and cluster traits
        drop_categories = [
            cat] if cat != 'commander-partnerID' else [cat, 'coloridentityID']
        commander_submap.get_cluster_traits(drop_categories=drop_categories)

        color_rule = 'ignore' if cat in [
            'tribeID', 'themeID', 'coloridentityID'] else 'superset'
        include_synergy = False if cat == 'coloridentity' else True

        # get the defining cards - will take a while because the empty tribe/themes are very large
        commander_submap.get_cluster_card_counts(
            color_rule=color_rule, include_commanders=include_commanders, verbose=True)
        commander_submap.get_defining_cards(
            include_synergy=include_synergy, n_scope=1000, verbose=True)

        # calculate average decklists - ignore the empty cluster for tribes and themes
        ignore_clusters = [cat_to_cluster['']] if cat in [
            'tribeID', 'themeID'] else []

        commander_submap.calculate_average_decklists(
            ignore_clusters=ignore_clusters, verbose=True)

        # now we export all these files to the output locations
        print('Exporting submap cards, traits, and average decks...', end='')
        for clust in sorted(list(set(commander_submap.cluster_defining_cards['clusterID']))):

            # get the name of the submap from the cluster
            value = cluster_to_cat[clust]

            # skip empty tribes and themes
            if value == '' and cat in ['tribeID', 'themeID']:
                continue

            # now we generate the name of the folder. In all cases except color identity, this should match the EDHREC slug,
            # so that we can link to EDHREC. How this is done depends on the category.
            if cat == 'colorIdentityID':
                out_value = value.lower() if value != '' else 'c'
            elif cat == 'themeID':
                out_value = override_dict['themeID'].get(
                    value, {}).get('edhrec_slug', value).lower()
            elif cat == 'tribeID':
                override_value = override_dict['tribeID'].get(
                    value, {}).get('edhrec_slug', '').lower()
                if override_value != '':
                    out_value = override_value
                else:
                    p = inflect.engine()
                    out_value = p.plural(value).lower()
            elif cat == 'commander-partnerID':
                out_value = card.kebab(value.lower())

            # make this output folder
            output_dir = f'{out_dir}/submaps/{cat}/{out_value}'

            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)

            # fetch json version of the cluster data
            cluster_json = commander_submap.jsonify_map(magic_cards, clusters=[clust],
                                                        trait_mapping=trait_mapping)

            with open(output_dir + '/submap.json', "w") as outjson:
                json.dump(cluster_json, outjson)

            # Here, we reformat theme and tribe names and associated images.
            # The default image is simply the most defining card for that theme or tribe
            if cat in ['themeID', 'tribeID']:

                specific_submap_cards = commander_submap.cluster_defining_cards.query(
                    f'clusterID == {clust}')

                # fetch defaults
                trait_name = calculate.format_tribe_theme_slug(value, cat)
                card_image = specific_submap_cards['card'].iloc[0]

                # get overrides if necessary
                card_image = override_dict[cat].get(
                    value, {}).get('image', card_image)
                trait_name = override_dict[cat].get(
                    value, {}).get('name', trait_name)

                # add this value's data.
                trait_images_names.append(
                    [cat, value, out_value, trait_name, card_image])
        print('done\n')

    # finally, combine in the slug/image override information
    trait_info = pd.DataFrame(trait_images_names, columns=[
                              'category', 'internal_slug', 'folder_slug', 'name', 'image'])
    trait_mapping_slug_images = trait_mapping_df.merge(
        trait_info, on=['category', 'internal_slug'], how='left').fillna('')
    trait_mapping_slug_images = trait_mapping_slug_images.sort_values(by=[
                                                                      'category', 'id'])
    trait_mapping_slug_images = trait_mapping_slug_images[[
        'category', 'id', 'internal_slug', 'folder_slug', 'name', 'image']]

    return trait_mapping_slug_images


def calculate_partner_traits(commander_map, grouped_data, trait_mapping_df,
                             magic_cards, include_commanders,
                             out_dir):
    '''Calculates and exports submap data for decks containing each partner--defining traits and cards.
    There should probably be a better way to do this.

    Params:
    -------
    commander_map: CommanderMap object
    grouped_data: df, a dataframe of ['category', 'value', 'count'] that contains partnerIDs
    trait_mapping_df: a dataframe of trait mappings
    magic_cards: dict, cardname to magic properties
    out_dir: str, output directory

    Returns:
    --------
    partners: the partners that are present in the data
    '''

    # extract partners
    partner_data = grouped_data[grouped_data['category']
                                == 'partnerID'].reset_index()
    partners = partner_data['value'].to_list()

    # convert the trait_mapping_df into a lookup dictionary for ids
    trait_mapping = load.build_trait_mapping(trait_mapping_df=trait_mapping_df)

    # precalculate the noncard data for all decks without a partner
    commander_no_partner = commander_map.copy_with_ref(commander_map.decklist_matrix, commander_map.commander_decks,
                                                       commander_map.cdecks)

    # we do this by declaring decks without a partner as cluster = 0 and decks with a partner as cluster = 1
    partner_indices = commander_no_partner.commander_decks[lambda row: (row['commanderID'].isin(partners)) |
                                                                       (row['partnerID'].isin(partners))].index
    commander_no_partner.commander_decks['clusterID'] = 0
    commander_no_partner.commander_decks.loc[partner_indices, 'clusterID'] = 1

    num_no_partner_decks = len(
        commander_no_partner.commander_decks) - len(partner_indices)
    print(
        f'\tCalculating reference data for {num_no_partner_decks} decks without a partner...', end='')
    _, cluster_noncard_df = commander_no_partner.get_cluster_card_counts(color_rule='ignore',
                                                                         include_commanders=include_commanders, verbose=False)
    # our precalculated noncard counts for cluster = 0 (no partner)
    precomputed_no_partner_noncard_counts = {0: cluster_noncard_df.loc[0, :]}
    print('done')

    # for each partner, calculating the defining traits and cards
    print('\tDefining and exporting submaps for each partner...', end='')
    for index, row in partner_data.iterrows():
        if index % 10 == 0:
            print(index, end='...')
        partner = row['value']

        # copy the commander_map object with original references
        commander_submap = commander_map.copy_with_ref(commander_map.decklist_matrix, commander_map.commander_decks,
                                                       commander_map.cdecks)

        # map the partner assignments to clusters. Designate decks with no partner as 0,
        # decks with any other partner as 1, and decks with the partner in question as 2 (we want 2).
        specific_p_indices = commander_submap.commander_decks[lambda row: (row['commanderID'] == partner) |
                                                                          (row['partnerID'] == partner)].index
        commander_submap.commander_decks['clusterID'] = 0
        commander_submap.commander_decks.loc[partner_indices, 'clusterID'] = 1
        commander_submap.commander_decks.loc[specific_p_indices,
                                             'clusterID'] = 2

        # now that we've declared clusters, we simply get the cluster defining cards and cluster traits
        commander_submap.get_cluster_traits()

        # get the defining cards, passing precomputed data for cluster = 0
        commander_submap.get_cluster_card_counts(color_rule='ignore', include_commanders=include_commanders, verbose=False,
                                                 precomputed_noncard=precomputed_no_partner_noncard_counts)
        commander_submap.get_defining_cards(
            include_synergy=True, n_scope=1000, verbose=False)

        # calculate average decklists, ignoring cluster 0 and 1
        commander_submap.calculate_average_decklists(
            ignore_clusters=(0, 1), verbose=False)

        # fetch the name of the folder
        out_value = card.kebab(partner.lower())

        # make this output folder
        output_dir = f'{out_dir}/submaps/commander-partnerID/{out_value}'
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        # fetch json version of the cluster data - in this case, cluster == 2
        cluster_json = commander_submap.jsonify_map(magic_cards, clusters=[2],
                                                    trait_mapping=trait_mapping)

        with open(output_dir + '/submap.json', "w") as outjson:
            json.dump(cluster_json, outjson)

    print('done')
    return partners


def export_traits(trait_mapping_slug_images, partners, commander_map, out_dir):
    '''Splits the trait mapping csv into individual traits (commanderID, themeID, colorIdentityID, and tribeID).
    Also adds a count column for how many decks correspond to that trait.
    '''

    # we begin by getting the counts for the partners
    partner_counts = []
    for p in partners:
        num_decks = sum((commander_map.commander_decks['commanderID'] == p) |
                        (commander_map.commander_decks['partnerID'] == p))
        partner_counts.append(num_decks)

    # loop through all the categories, extracting their counts and information
    for field in ['tribeID', 'themeID', 'commanderID', 'colorIdentityID']:
        field_mapping_data = trait_mapping_slug_images[trait_mapping_slug_images['category'] == field]

        # get the counts of the category
        field_counts = commander_map.commander_decks[field].value_counts()

        # this count data can be missing partners if they are never solo commanders. Fill this in.
        all_unique_traits = set(
            trait_mapping_df[trait_mapping_df['category'] == field]['internal_slug'])
        field_counts = field_counts.reindex(all_unique_traits, fill_value=0)

        # if we're on commanders, update the commanders with the correct partner counts
        if field == 'commanderID':
            field_counts.loc[partners] = partner_counts

        # convert to a dataframe and merge with the image data
        field_counts = field_counts.to_frame(
            name='count').reset_index().rename(columns={'index': 'internal_slug'})

        field_mapping_data = field_mapping_data.merge(
            field_counts, on=['internal_slug'])

        # drop the first and second columns - they will be specified by file name and row number
        field_mapping_data = field_mapping_data.drop(
            columns=['category', 'id'])

        # for colors and commanders, we don't need the slug and image
        if field not in ['tribeID', 'themeID']:
            field_mapping_data = field_mapping_data.drop(
                columns=['folder_slug', 'name', 'image']).rename(columns={'internal_slug': 'name'})

        # tribes and themes need some slight renaming
        else:
            field_mapping_data = field_mapping_data.drop(
                columns=['internal_slug']).rename(columns={'folder_slug': 'slug'})

        # some pecularities
        if field == 'colorIdentityID':
            field_mapping_data.loc[0, 'name'] = 'C'

        # export the trait mapping!
        field_mapping_data.to_csv(f'{out_dir}/traits/{field}.csv', index=False)

    # now we also export all unique commander-partner pairings and their counts
    commander_partner_combos = commander_map.commander_decks['commander-partnerID'].value_counts(
    )
    commander_partner_combos = commander_partner_combos.to_frame(
        name='count').reset_index().rename(columns={'index': 'name'})

    # subset down to partners
    commander_partner_combos = commander_partner_combos[commander_partner_combos['name'].str.contains(
        '\+')]
    commander_partner_combos.to_csv(
        f'{out_dir}/traits/commander-partnerID.csv', index=False)

    return 0


def export_card_and_deck_files(commander_map, trait_mapping_df, duplicates, include_commanders,
                               out_dir, chunksize=100):
    '''Exports card and deck data

    Params:
    -------
    commander_map: CommanderMap object
    trait_mapping_df: df for mapping traits
    duplicates: list of duplicate cardnames
    include_commanders: whether commanders are included in commander_map.decklist_matrix
    out_dir: str, output directory
    chunksize: how many decks should be put into one json file

    Returns:
    --------
    0
    '''

    cardnames = np.array(list(commander_map.card_idx_lookup.keys()))

    # convert to csc for fast column accesses
    csc_matrix = commander_map.decklist_matrix.tocsc()

    print('\tBuilding card data...', end='')
    deck_play_cards = {}
    for i, c in enumerate(cardnames):
        if i % 2000 == 0:
            print(str(i//1000) + 'k', end='...')
        deck_play_cards[c] = csc_matrix[:, i].indices.astype(str).tolist()

    # if we didn't include commanders when we built the matrix, we include them now
    if not include_commanders:
        for i, cdeck in commander_map.cdecks.items():
            for commander in [cdeck.commander, cdeck.partner, cdeck.companion]:
                if commander != '':
                    deck_play_cards[commander].append(str(i))

    print('\n\tExporting card data...', end='')
    for i, cardname in enumerate(deck_play_cards):
        if i % 2000 == 0:
            print(str(i//1000) + 'k', end='...')
        filename = card.kebab(cardname)
        decks = deck_play_cards[cardname]

        with open(f'{out_dir}/cards/{filename}.csv', 'w') as outfile:
            outfile.write('\n'.join(decks))

    print('\n\tExporting decks...', end='')
    os.mkdir('decks')
    trait_mapping = load.build_trait_mapping(trait_mapping_df=trait_mapping_df)

    # export decklists in chunks of size chunksize
    id_list = list(commander_map.cdecks.keys())
    chunked_ids = [id_list[index:index+chunksize]
                   for index in range(0, len(id_list), chunksize)]

    for i, chunk in enumerate(chunked_ids):
        chunk_name = str(i*chunksize)
        chunk_decks = {}

        # fill in chunk with the deck information
        for deck_id in chunk:
            if deck_id % 100000 == 0:
                print(str(deck_id // 1000) + 'k', end='...')

            cdeck = commander_map.cdecks[deck_id]
            formatted_decklist = cdeck.format_decklist(magic_cards)

            # convert to string for json export
            deck_id = str(deck_id)
            chunk_decks[deck_id] = {'main': formatted_decklist,
                                    'commanderID': str(trait_mapping['commanderID'][cdeck.commander]),
                                    'price': str(int(cdeck.price))}

            for cardname, fieldname in zip([cdeck.partner, cdeck.companion], ['partnerID', 'companionID']):
                if cardname != '':
                    cardname = trait_mapping['commanderID'][cardname]
                    chunk_decks[deck_id][fieldname] = str(cardname)

            # next, we check for duplicate cards
            dups_in_deck = sorted(
                set(duplicates) & set(cdeck.cards))
            if len(dups_in_deck):
                chunk_decks[deck_id]['duplicates'] = dups_in_deck

        # export the chunk
        with open(f'decks/{chunk_name}.json', 'w') as outjson:
            json.dump(chunk_decks, outjson)

    return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Preprocesses the data from EDHREC to make the Lucky Paper Commander Map'
    )

    parser.add_argument('--data_dir', type=str,
                        help='directory containing decks.csv and cards.csv')
    parser.add_argument('--out_dir', type=str, help='output directory')
    parser.add_argument('--duplicates', type=str,
                        help='newline separated text file of cards that can be duplicates')
    parser.add_argument('--edhrec_to_scryfall', type=str,
                        help='json mapping EDHREC names to scryfall names')
    parser.add_argument('--slug_image_override', type=str,
                        help='json mapping EDHREC names to scryfall names')
    parser.add_argument('--n_split', type=int, default=1,
                        help='number of submap files to create.')
    parser.add_argument('--include_commanders',
                        default=False, action='store_true', help='whether commanders should be included in the count matrix.')

    args = parser.parse_args()
    data_dir = args.data_dir
    out_dir = args.out_dir
    duplicates = args.duplicates
    edhrec_to_scryfall = args.edhrec_to_scryfall
    slug_image_override = args.slug_image_override
    include_commanders = args.include_commanders
    n_split = args.n_split

    print(include_commanders)

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    duplicates = [cardname.strip('\n') for cardname in open(duplicates)]
    slug_image_override = pd.read_csv(slug_image_override).fillna('')

    # make intermediate outputs
    for subdir in ['map_intermediates', 'submaps', 'cards', 'traits']:
        if not os.path.isdir(os.path.join(out_dir, subdir)):
            os.mkdir(os.path.join(out_dir, subdir))

    print('\nPreprocessing data for the Commander Map...\n' + '-'*44 + '\n')
    print('Loading magic cards...', end='')
    magic_cards = load.load_magic_cards()
    print(f'loaded {len(magic_cards)} cards')

    print('\nLoading EDHREC data...')
    commander_decks, commander_deck_dict, total_seen_cards = load_edhrec_data(
        data_dir, edhrec_to_scryfall, magic_cards)

    # remove duplicates, fix partners and companions, etc
    commander_decks, commander_deck_dict, partners = clean_commander_decks(commander_decks, commander_deck_dict,
                                                                           magic_cards, total_seen_cards, include_commanders)

    # calculate prices of the decks
    commander_decks, commander_deck_dict = calculate_prices(
        commander_decks, commander_deck_dict, include_commanders, magic_cards)

    # filter out duplicate decks and require at least one non-commander card
    commander_decks, cdeck_subset = filter_commander_decks(
        commander_decks, commander_deck_dict)

    print('\nBuilding the deck, date, and color identity matrices...')
    card_idx_lookup, decklist_matrix = build_decklist_matrix(
        cdeck_subset, magic_cards, out_dir)
    deck_date_idx_lookup, card_date_idx_lookup, date_matrix = load.build_date_matrix(
        commander_decks, magic_cards, card_idx_lookup, out_dir)
    deck_ci_idx_lookup, card_ci_idx_lookup, ci_matrix = load.build_ci_matrix(
        commander_decks, magic_cards, card_idx_lookup, out_dir)

    # build the commander map object
    commander_map = map_classes.CommanderMap(
        decklist_matrix, commander_decks, cdeck_subset)

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

    print('\n')
    # extract the trait mapping
    trait_mapping_df = commander_map.get_trait_mapping()

    # and the websites - adding some keys for site info
    commander_map.extract_deck_sources()

    # export commander decks - this is an intermediate used later
    commander_map.commander_decks.to_csv(
        f'{out_dir}/map_intermediates/commander-decks.csv', index=False)

    print('\nCalculating how many submaps are needed...', end='')
    grouped_data = define_grouped_data(
        commander_map, partners=partners, n_split=n_split)
    print(f'{len(grouped_data)} submaps')

    print('\nCalculating whole submap traits...\n')
    trait_mapping_slug_images = calculate_whole_submap_traits(commander_map, magic_cards, include_commanders, trait_mapping_df,
                                                              slug_image_override, out_dir)

    # export the partner submap data
    print(f'PartnerID : {len(partners)} submaps')
    partners = calculate_partner_traits(commander_map, grouped_data, trait_mapping_df,
                                        magic_cards, include_commanders, out_dir)

    # export the traits
    export_traits(trait_mapping_slug_images, partners, commander_map, out_dir)
    trait_mapping_slug_images.to_csv(
        f'{out_dir}/trait-mapping.csv', index=False)

    # export the card and deck data
    print('\nExporting cards and decks...')
    export_card_and_deck_files(commander_map, trait_mapping_df=trait_mapping_df, duplicates=duplicates,
                               include_commanders=include_commanders, out_dir=out_dir)
    print('done')
