import pandas as pd
import scipy
import requests
import numpy as np
from datetime import datetime
from collections import defaultdict

import calculate
import card
import map_classes


def load_magic_cards():
    '''Loads in magic cards from scryfall oracle-cards, adding in card release dates and minimum price from
    default cards.

    Params:
    -------

    Returns:
    --------
    magic_cards: dict, contains information for each magic card.
    '''

    # we use the oracle cards, which contains just one version for every card.
    resp = requests.get('https://api.scryfall.com/bulk-data/oracle-cards')
    scryfall_download_uri = resp.json()['download_uri']

    magic_cards = card.fetch_cards(
        scryfall_url=scryfall_download_uri, lower=False)

    # we now add in release dates and minimum price
    scryfall_resp = requests.get(
        'https://api.scryfall.com/bulk-data/default-cards')
    scryfall_download_uri = scryfall_resp.json()['download_uri']
    resp = requests.get(scryfall_download_uri)
    default_scryfall_cards = resp.json()

    # add price information
    magic_cards = card.find_price_and_release(
        magic_cards, default_scryfall_cards)

    return magic_cards


def build_date_matrix(commander_decks, magic_cards, card_idx_lookup, out_dir):
    '''Builds an l x k date matrix, where l and k are the number of unique
    deck and card dates, respectively. Entry [i, j] is 1 if date i > date j

    Params:
    -------
    commander_decks: df, dataframe of commander decks
    magic_cards: dict, output of load_magic_cards
    card_idx_lookup: dict, look up of card to decklist matrix index
    out_dir: str, output directory

    Returns:
    --------
    deck_date_idx_lookup: dict, deck index to date matrix row index
    card_date_idx_lookup: dict, cardname to date matrix column index
    date_matrix: numpy array, l x k date matrix
    '''

    print('\tBuilding date matrix...', end='')
    # lookup of card dates, define possible card and deck dates
    card_date_lookup = {cardname: datetime.strptime(
        magic_cards[cardname]['earliest_release'], '%Y-%m-%d') for cardname in card_idx_lookup.keys()}
    deck_dates = sorted(set(commander_decks['savedate']))
    card_dates = sorted(set(card_date_lookup.values()))

    date_matrix = pd.DataFrame(0, index=deck_dates, columns=card_dates)
    for i, deck_dt in enumerate(deck_dates):
        deck_dt = datetime.strptime(deck_dt, '%Y-%m-%d')

        # Define where the matrix is 1
        indices_can_play = np.where(deck_dt >= np.array(card_dates))
        date_matrix.iloc[i, indices_can_play] = 1

    # we export it as a dataframe to preserve row and column names
    date_matrix.index.name = 'deck_date'
    date_matrix.to_csv(f'{out_dir}/map_intermediates/date-matrix.csv')
    date_matrix = date_matrix.values

    # here we define the lookup dictionaries - they take in a deck's index in the dataframe, or a cardname, and they return the index of the appropriate
    # row/column in date_matrix
    deck_date_idx_lookup = dict(zip(deck_dates, range(len(deck_dates))))
    deck_date_idx_lookup = {i: deck_date_idx_lookup[deck_date] for i, deck_date in enumerate(
        commander_decks['savedate'].to_list())}

    # repeat for cards
    card_date_idx_lookup = dict(zip(card_dates, range(len(card_dates))))
    card_date_idx_lookup = {
        cardname: card_date_idx_lookup[card_date_lookup[cardname]] for cardname in card_idx_lookup.keys()}
    row_len, col_len = date_matrix.shape
    print(f'generated a ({row_len} x {col_len}) date matrix')

    return deck_date_idx_lookup, card_date_idx_lookup, date_matrix


def build_ci_matrix(commander_decks, magic_cards, card_idx_lookup, out_dir):
    '''Builds an 32 x 32 color identity matrix. Entry [i, j] is 1 if
    color identity i can play color identity j (i.e. i is a superset of j)

    Params:
    -------
    commander_decks: df, dataframe of commander decks
    magic_cards: dict, output of load_magic_cards
    card_idx_lookup: dict, look up of card to decklist matrix index
    out_dir: str, output directory

    Returns:
    --------
    deck_ci_idx_lookup: dict, deck index to ci matrix row index
    card_ci_idx_lookup: dict, cardname to ci matrix column index
    ci_matrix: numpy array, 32 x 32 ci matrix
    '''

    print('\tBuilding color identity matrix...', end='')

    # generate all 32 color identities
    coloridentities = list(''.join(ci)
                           for ci in calculate.powerset(['W', 'U', 'B', 'R', 'G']))

    # generate our matrix. Element i, j is 1 if color identity i is a
    # subset of color identity j (e.g. R is in UBR)
    ci_matrix = pd.DataFrame(0, index=coloridentities, columns=coloridentities)
    for ci_deck in coloridentities:
        for ci_card in coloridentities:
            if all([c in ci_deck for c in ci_card]):
                ci_matrix.loc[ci_deck, ci_card] = 1

    # we export it as a dataframe to preserve row and column names
    ci_matrix.index.name = 'deck_ci'
    ci_matrix.to_csv(f'{out_dir}/map_intermediates/coloridentity-matrix.csv')
    ci_matrix = ci_matrix.values

    # next we define lookups to go from deck id or cardname to row/column of the ci matrix
    ci_idx_lookup = dict(zip(coloridentities, range(len(coloridentities))))

    deck_ci_idx_lookup = {i: ci_idx_lookup[ci] for i, ci in enumerate(
        commander_decks['colorIdentityID'].to_list())}
    card_ci_idx_lookup = {
        cardname: ci_idx_lookup[magic_cards[cardname]['color_identity']] for cardname in card_idx_lookup.keys()}
    print('done')

    return deck_ci_idx_lookup, card_ci_idx_lookup, ci_matrix


def load_decklists(sparse_decklist_path, sparse_columns_path):
    '''Loads the sparse decklist matrix using the path and colum names

    Params:
    -------
    sparse_decklist_path: str, path to sparse .npz decklist
    sparse_columns_path: str, path to \n delimited file of cards that represent each column

    Returns:
    --------
    decklist_matrix: csr sparse matrixm, containing decklists
    card_idx_lookup: dict, card to column index in decklist_matrix
    '''

    # load decklist matix, convert to lil
    decklist_matrix = scipy.sparse.load_npz(sparse_decklist_path)
    decklist_matrix = decklist_matrix.tocsr()

    # load the columns
    card_idx_lookup = [line.strip() for line in open(sparse_columns_path)]
    card_idx_lookup = dict(zip(card_idx_lookup, range(len(card_idx_lookup))))

    return decklist_matrix, card_idx_lookup


def load_date_matrix(date_matrix_path, commander_decks, card_idx_lookup, magic_cards):
    '''Loads the date matrix data. Returns the values of the matrix, 
    as well as card to index and deck id to index lookups

    Params:
    -------
    date_matrix_path: str, path to array of date matrix values
    commander_decks: df, dataframe of commander decks (with savedate as a key)
    card_idx_lookup: dict, card to idx present in decklist_matrix
    magic_cards: dict, mapping card to card properties (with earliest_release key)

    Returns:
    --------
    date_matrix: numpy array, with unique deck dates as rows and unique card dates as columns (as ints)
    deck_date_idx_lookup: dict, mapping a deck id to its row index in date_matrix
    card_date_idx_lookup: dict, mapping a cardname to its column index in date_matrix

    For example, can see if deck 20 can play Giant Growth based on date using:
        row_idx = deck_date_idx_lookup[20]
        col_idx = card_date_idx_lookup['Giant Growth']
        can_play_gg = date_matrix[row_idx, col_idx] # 1 if deck can play Giant Growth based on date

    The use of a matrix allows for fast lookups with arrays.
    '''

    # load the date matrix, convert to numpy array and build lookup
    date_matrix = pd.read_csv(date_matrix_path).set_index('deck_date')
    card_date_to_idx = dict(
        zip(date_matrix.columns, range(len(date_matrix.columns))))
    deck_date_to_idx = dict(
        zip(date_matrix.index, range(len(date_matrix.index))))
    date_matrix = date_matrix.values

    # use the umap plot and idx lookup to convert to card/date: date matrix index
    card_date_idx_lookup = {
        cardname: card_date_to_idx[magic_cards[cardname]['earliest_release']] for cardname in card_idx_lookup}
    deck_date_idx_lookup = {deck_id: deck_date_to_idx[deck_date] for deck_id, deck_date in zip(
        commander_decks['deckID'].values, commander_decks['savedate'].values)}

    return date_matrix, deck_date_idx_lookup, card_date_idx_lookup


def load_ci_matrix(ci_matrix_path, commander_decks, card_idx_lookup, magic_cards):
    '''Loads the color identity matrix data. Returns the values of the matrix, 
    as well as card to index and deck id to index lookups

    Params:
    -------
    ci_matrix_path: str, path to array of color identity matrix values
    commander_decks: df, dataframe of commander decks (with savedate as a key)
    card_idx_lookup: dict, card to idx present in decklist_matrix
    magic_cards: dict, mapping card to card properties (with earliest_release key)

    Returns:
    --------
    ci_matrix: 32 x 32 numpy array, with color identities as row and columns. 
              i,j = 1 if card ci_i can be played in deck ci_j.
    deck_ci_idx_lookup: dict, mapping a deck id to its color identity index
    card_ci_idx_lookup: dict, mapping a cardname to its color identity index

    For example, can see if deck 20 can play Giant Growth based on color identity using:
        row_idx = deck_ci_idx_lookup[20]
        col_idx = card_ci_idx_lookup['Giant Growth']
        can_play_gg = ci_matrix[row_idx, col_idx] # 1 if deck can play Giant Growth based on color identity.

    The use of a matrix allows for fast lookups with arrays.
    '''

    # repeat for color identity
    ci_matrix = pd.read_csv(ci_matrix_path).fillna('').set_index('deck_ci')
    ci_matrix.columns = ci_matrix.index
    ci_idx_lookup = dict(zip(ci_matrix.index, range(len(ci_matrix.index))))
    ci_matrix = ci_matrix.values

    card_ci_idx_lookup = {
        cardname: ci_idx_lookup[magic_cards[cardname]['color_identity']] for cardname in card_idx_lookup}
    deck_ci_idx_lookup = {deck_id: ci_idx_lookup[deck_ci] for deck_id, deck_ci in zip(
        commander_decks['deckID'].values, commander_decks['colorIdentityID'].values)}

    return ci_matrix, deck_ci_idx_lookup, card_ci_idx_lookup


def load_cdecks(commander_decks, decklist_matrix, card_idx_lookup):
    '''Loads the commander decks as commander deck objects. Sometimes needed
    because code often references a CommanderDeck object for average decklists.

    Params:
    -------
    commander_decks: df, dataframe of commander decks (with savedate as a key)
    decklist_matrix: csr sparse matrixm, containing decklists
    card_idx_lookup: dict, dict, card to idx present in decklist_matrix

    Returns:
    --------
    cdecks: dict, mapping deck id to CommanderDeck object
    '''

    cdecks = {}
    card_list = np.array(list(card_idx_lookup.keys()))
    for i, row in commander_decks.iterrows():

        # store commander information
        cdeck = map_classes.CommanderDeck()
        cdeck.url = row['url']

        # set commander and partner
        cdeck.commander = row['commanderID']
        cdeck.partner = row['partnerID']
        cdeck.companion = row['companionID']
        cdeck.deckid = row['deckID']

        # add other metadata
        cdeck.colorIdentity = row['colorIdentityID']
        cdeck.theme = row['themeID']
        cdeck.tribe = row['tribeID']
        cdeck.date = row['savedate']
        cdeck.price = row['price']

        # add cards
        played_card_idx = decklist_matrix[i, :].indices
        cards = card_list[played_card_idx]

        cdeck.cards = list(cards)

        cdecks[i] = cdeck

    return cdecks


def build_trait_mapping(trait_mapping_path=None, trait_mapping_df=None):
    '''Given a path, builds a trait mapping. If the trait mapping df is passed, use that instead.

    Params:
    -------
    trait_mapping_path or df: path to trait mapping df or the df itself. Columns are ['category', 'id', 'internal_slug',
        'folder_slug', 'name', 'image']. The point of trait mapping is to go from 'internal slug' to 'id' 
        (an integer representation) for data compression.

    Returns:
    --------
    trait_mapping: dict, mapping category > internal slug to its id.

    Example:
        To map the tribe Elves to its integer presentation: 
        trait_mapping['tribeID']['Elves'] # yields X, where X is an integer
    '''

    if trait_mapping_path is not None:
        trait_mapping_df = pd.read_csv(trait_mapping_path, usecols=[
                                       'category', 'internal_slug', 'id']).fillna('')

    elif trait_mapping_df is None:
        raise NotImplementedError(
            'You must pass a trait mapping path or dataframe')

    else:
        trait_mapping_df = trait_mapping_df[[
            'category', 'internal_slug', 'id']]

    # convert the dataframe into a lookup dictionary
    trait_mapping = defaultdict(dict)
    for _, row in trait_mapping_df.iterrows():
        cat, internal_slug, id = row[[
            'category', 'internal_slug', 'id']].values
        trait_mapping[cat][internal_slug] = id

    # convert back to a dict to maintain missing key behavior
    return dict(trait_mapping)
