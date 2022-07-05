import numpy as np
import umap.umap_ as umap
import hdbscan
import scipy
import pandas as pd
import copy
from collections import defaultdict

import card
import calculate


class CommanderDeck:
    '''A class for storing Commander deck characteristics'''

    def __init__(self):

        self.colorIdentity = None
        self.commander = None
        self.partner = None
        self.companion = None
        self.tribe = None
        self.theme = None
        self.cards = None
        self.date = None
        self.url = None
        self.urlhash = None
        self.deckid = None
        self.price = None

    def calculate_price(self, magic_cards, include_commanders=True):
        '''Calculates the price of the deck, based on earliest release

        Params:
        -------
        magic_cards: dict, mapping cardname to properties. Needs 'min_price' key.
        include_commanders: whether commanders should be included in the price calculation.

        Returns:
        --------
        price: float, also stores value in self.price
        '''

        deck_prices = [magic_cards[cardname]['min_price']
                       for cardname in self.cards]

        if include_commanders:
            commanders = [self.commander, self.partner, self.companion]
            deck_prices += [magic_cards[cardname]['min_price']
                            for cardname in commanders if cardname != '']

        price = np.nansum(deck_prices)
        self.price = price

        return price

    def format_decklist(self, magic_cards, include_commanders=False):
        '''Given a decklist and a magic card dictionary, sort the decklist based on
        interpretable magic characteristics. We sort in the following way: creatures > sorceries > instants > artifacts > 
        enchantments > nonbasics > basics, with a space between each category. We sort by mv then alphabetically.
        We add basics to the decklist in accordance with the color breakdown of the decklist.

        Params:
        -------
        magic_cards: dict, mapping cardname to properties. Needs keys like 'mana_cost', 'type_line', 'color_identity'.
        include_commanders: bool, whether commanders should be included in the decklist. If included, the order is
            commander > partner > companion (omitting any that are absent).

        Returns:
        --------
        formatted_decklist: list, list of cards in deck with number in decklist prepended to card name and spaces
            in between type switches.
        '''

        # we begin by looking up the traits for the decklist
        traits = []
        color_pips = defaultdict(int)
        colors = ['W', 'U', 'B', 'R', 'G', 'C']
        for cardname in self.cards:

            # fetch the card info (default given for partner == '')
            card_info = magic_cards.get(cardname, {'mana_cost': ''})

            # store the number of pips the card contributes
            pips = card_info['mana_cost']
            for c in colors:
                color_pips[c] += pips.count(c)

            # we don't store commander and partner for maindeck export
            if cardname in [self.commander, self.partner, self.companion]:
                continue

            # get the most important type (see card.extract_types) and mv
            card_type = card.extract_types(cardname, card_info['type_line'])
            mv = card_info['cmc']

            traits.append([card_type, mv, cardname])

        # sort based on trait, then mv, then alphabetically
        type_order = ['Creature', 'Sorcery', 'Instant',
                      'Artifact', 'Enchantment', 'Planeswalker',
                      'Nonbasic Land', 'Basic Land']
        sorted_decklist = sorted(traits, key=lambda traits: (type_order.index(traits[0]),
                                                             traits[1],
                                                             traits[2]))

        # remove the basics
        sorted_decklist = [c for c in sorted_decklist if c[0] != 'Basic Land']

        # next we add basics. We start by calculating a baseline based on the distribution of pips.
        # companions are the 101st card
        color_pips = np.array(list(color_pips.values()))
        num_basics = 100 - (len(sorted_decklist) + 1 + (self.partner != ''))

        # if there are no color pips, yield an equal supply of basics
        # based on the commander and partner color identities (this is distinct from their mana cost)
        if color_pips.sum() == 0:
            commander_cis = set(magic_cards[self.commander]['color_identity'])
            if self.partner:
                commander_cis.update(
                    magic_cards[self.partner]['color_identity'])

            # add in basics. If the commander is colorless, we use wastes
            color_pips = np.array(
                [1 if c in commander_cis else 0 for c in colors])
            if commander_cis == set():
                color_pips[-1] += 1

        # calculate the basic distribution based on these pips
        basic_dist = np.floor(color_pips/color_pips.sum() * num_basics)

        # then we need to make sure we're at 100 cards. We add missing basics to the most represented basic.
        remaining_basics = num_basics - basic_dist.sum()
        if remaining_basics != 0:
            basic_dist[np.argmax(basic_dist)] += remaining_basics

        # make sure we have 100 cards--add 1 for commanders. Companions are the 101st
        assert sum(basic_dist) + len(sorted_decklist) + \
            1 + (self.partner != '') == 100, 'Decklist not 100 cards'

        # now we convert this to a list we can add to the end of a decklist
        basic_order = ['Plains', 'Island', 'Swamp',
                       'Mountain', 'Forest', 'Wastes']
        basics = [f'{int(count)} {basic}' for count, basic in zip(
            basic_dist, basic_order) if count != 0]

        # next we format the decklist itself
        decklist_formatted = []

        # we add in the maindeck cards, adding spaces whenever a card type changes.
        if len(sorted_decklist) == 0:
            prev_type = ''
        else:
            prev_type = sorted_decklist[0][0]

        for card_traits in sorted_decklist:
            if card_traits[0] == 'Basic Land':
                continue

            if card_traits[0] == prev_type:
                decklist_formatted.append('1 ' + card_traits[-1])
            else:
                decklist_formatted += ['', '1 ' + card_traits[-1]]
            prev_type = card_traits[0]

        decklist_formatted += [''] + basics

        # include commanders if specified
        if include_commanders:
            decklist_formatted = [''] + decklist_formatted
            if self.companion != '':
                decklist_formatted = [
                    '1 ' + self.companion] + decklist_formatted
            if self.partner != '':
                decklist_formatted = ['1 ' + self.partner] + decklist_formatted

            decklist_formatted = ['1 ' + self.commander] + decklist_formatted

        return decklist_formatted


class CommanderMap:

    '''A class for constructing commander maps'''

    def __init__(self, decklist_matrix, commander_deck_df, commander_decks=None):

        # initiate some checks
        if commander_decks is not None:
            if not (decklist_matrix.shape[0] == commander_deck_df.shape[0] == len(commander_decks)):
                raise ValueError(
                    'Decklist matrix, commander df, and list of commander deck objects must have the same number of elements.')
        else:
            if not (decklist_matrix.shape[0] == commander_deck_df.shape[0]):
                raise ValueError(
                    'Decklist matrix and commander df must have the same number of elements.')

        # check that the commander decks and dataframe agree (we don't check decklist_matrix)
        if commander_decks is not None:
            comms_cdeck = [(commander_decks[i].commander, commander_decks[i].partner,
                            commander_decks[i].companion) for i in commander_deck_df['deckID']]
            comms_df = commander_deck_df[[
                'commanderID', 'partnerID', 'companionID']].values
            if not (np.array(comms_cdeck, dtype=object) == comms_df).all():
                raise ValueError(
                    'Different data is contained for commander, partner, and companion in the cdecks and dataframe.')

        # a sparse matrix of decklists and dataframe of commander decks
        self.decklist_matrix = decklist_matrix
        self.commander_decks = commander_deck_df
        self.cdecks = commander_decks

        # the 2d coordinate embedding and embedding for clustering
        self.coordinates = None
        self.cluster_embedding = None

        # cluster labels
        self.cluster_labels = None

        # cluster traits
        self.cluster_traits = None

        # date and color identity matrices, which
        # are used to lookup if a card can be played in a deck
        self.date_matrix = None
        self.ci_matrix = None

        # some lookup tables
        self.card_idx_lookup = None  # key = card, value = column index in decklist_matrix
        # key = deck_id, value = column index in date_matrix
        self.deck_date_idx_lookup = None
        self.card_date_idx_lookup = None  # key = card, value = row index in date_matrix
        # key = deck_id, value = column index in color identity matrix
        self.deck_ci_idx_lookup = None
        self.card_ci_idx_lookup = None  # key = card, value = row index in ci_matrix

        # related to defining cards of each cluster
        self.cluster_card_df = None  # number of decks in each cluster that play each card
        # number of decks in each cluster that could play a card but do not
        self.cluster_noncard_df = None
        self.cluster_defining_cards = None

        # average decklists
        self.average_decklists = None

        # trait mapping df - for mapping traits to integers
        self.trait_mapping_df = None

    def copy_with_ref(self, decklist_matrix, commander_deck_df, commander_decks):
        '''Returns an object with new decklists and commander decks but with copied references

        Params:
        -------
        decklist_matrix: sparse csr matrix, decklist matrix to use
        commander_deck_df: df, dataframe of commander information
        commander_decks: dict, dict mapping deck id to CommanderDeck object

        Returns:
        --------
        new_obj: a Commander Map object with same references as self but new data
        '''

        new_obj = CommanderMap(copy.copy(decklist_matrix), copy.copy(
            commander_deck_df), copy.copy(commander_decks))

        # copy references
        new_obj.date_matrix = copy.copy(self.date_matrix)
        new_obj.ci_matrix = copy.copy(self.ci_matrix)
        new_obj.card_idx_lookup = copy.copy(self.card_idx_lookup)
        new_obj.deck_date_idx_lookup = copy.copy(self.deck_date_idx_lookup)
        new_obj.card_date_idx_lookup = copy.copy(self.card_date_idx_lookup)
        new_obj.deck_ci_idx_lookup = copy.copy(self.deck_ci_idx_lookup)
        new_obj.card_ci_idx_lookup = copy.copy(self.card_ci_idx_lookup)

        return new_obj

    def get_trait_mapping(self):
        '''Creates a mapping of different traits to integers, based on all traits present in self.commander_decks.
        For the trait commanderID, this will include partners and companions (it should really be cardID).

        Returns:
        --------
        trait_mapping_df: df, a dataframe with trait mappings. Columns are 'category', 'internal_slug', 'id'.

        Example:
            commanderID    "Xenagos, God of Revels"   45
        '''

        print('Defining trait mappings...', end='')
        trait_mapping = []
        for field in ['commanderID', 'themeID', 'tribeID', 'colorIdentityID']:

            # get all the unique values. If commander, add partners
            if field == 'colorIdentityID':
                unique_values = list(
                    ''.join(ci) for ci in calculate.powerset(['W', 'U', 'B', 'R', 'G']))

            else:
                unique_values = list(self.commander_decks[field].unique())

                # we add companions and partners to the list. They are almost assuredly already present
                # in the data but we make sure
                if field == 'commanderID':
                    unique_values = unique_values + \
                        list(self.commander_decks['partnerID'].unique()) + \
                        list(
                            set([cdeck.companion for cdeck in self.cdecks.values() if cdeck.companion != '']))

                    unique_values = list(set(unique_values))

                unique_values = sorted(unique_values)

                # remove empty partners, tribes, and themes (there are empty color identities: colorless)
                unique_values = [val for val in unique_values if val != '']

            # creating a dataframe that stores the mapping information. We use `internal_slug` because
            # this refers to how the data is referred to in the EDHREC data
            subtrait_mapping = pd.DataFrame()
            subtrait_mapping['internal_slug'] = unique_values
            subtrait_mapping['id'] = range(len(unique_values))
            subtrait_mapping['category'] = field
            trait_mapping.append(subtrait_mapping)

        # create initial trait mapping. This may be overwritten later.
        trait_mapping_df = pd.concat(trait_mapping)
        trait_mapping_df = trait_mapping_df[[
            'category', 'internal_slug', 'id']]

        print(f'{len(trait_mapping_df)} unique traits')

        self.trait_mapping_df = trait_mapping_df
        return trait_mapping_df

    def extract_deck_sources(self):
        '''Extracts the sources of the decks, compressing URLs.
        Overwrites self.commander_decks to add new fields.

        Returns:
        --------
        0
        '''

        print(f'Calculating deck sources and compressing URLs...', end='')
        self.commander_decks['siteID'] = self.commander_decks['url'].apply(
            calculate.extract_source_from_url)
        self.commander_decks['path'] = self.commander_decks.apply(
            lambda row: calculate.fetch_decklist_ids_from_url(row['url'], row['siteID']), axis=1)

        # then we convert the site to an integer
        unique_sites = sorted(list(set(self.commander_decks['siteID'])))
        site_mapping = dict(zip(unique_sites, range(len(unique_sites))))
        self.commander_decks['siteID'] = self.commander_decks['siteID'].replace(
            site_mapping)
        print(f'found {len(unique_sites)}: {unique_sites}')

        return 0

    def reduce_dimensionality(self, method='UMAP', n_dims=None, coordinates=True, on_embedding=False,
                              **method_kwargs):
        '''Reduce the dimensionality of decklist_matrix to n dimensions

        Params:
        -------
        method: str, the method of dimensionality reduction (UMAP or PCA)
        n_dims: int, number of dimensions for reduction (6 for UMAP, 50 for PCA)
        coordinates: bool, True if the embedding is intended for map coordinates, False
                     if intended for clustering
        on_embedding: bool, if True, embed the existing embedding in self.cluster_embedding. Useful if
                      if you want to do PCA > UMAP for embedding.
        method_kwargs: kwargs to pass to method

        Returns:
        --------
        numpy array of embedded values (n_decks x n_dims)
        '''

        np.random.seed(0)
        random_state = np.random.RandomState(0)

        # some quick checks
        if method not in ['UMAP']:
            raise NotImplementedError('Only UMAP is implemented.')

        if coordinates and n_dims != 2:
            raise ValueError(
                f'If coordinates is True, n_dims must be 2. You put {n_dims}')

        if on_embedding and self.cluster_embedding is None:
            raise ValueError(
                'If on_embedding is True, you must run reduce_dimensionality with coordinates = False first.')

        # a quick check that the number of decklists and commander decks is the same
        assert self.decklist_matrix.shape[0] == self.commander_decks.shape[0]

        # define the embedder - UMAP or PCA. Only UMAP supported currently.
        if method == 'UMAP':
            if n_dims is None:
                n_dims = 6 if len(self.commander_decks) > 10**5 else 4

            if 'metric' not in method_kwargs:
                if on_embedding:
                    method_kwargs['metric'] = 'euclidean'
                else:
                    method_kwargs['metric'] = 'jaccard'

            embedder = umap.UMAP(n_components=n_dims,
                                 random_state=random_state,
                                 verbose=True,
                                 **method_kwargs)

        # define the data that is used for the embedding
        if on_embedding:
            data = self.cluster_embedding
        else:
            data = self.decklist_matrix

        # run the embedding. If too few to embed, we "embed" them ourselves
        try:
            embedding = embedder.fit_transform(data)

        except:
            print('Failed to embed. Using random embedding.')
            embedding = np.random.uniform(size=(len(self.commander_decks), 2))

        # Sometimes this embedding will have nan if some decks are disconnected. We correct
        # this by assigning these decks to their nearest neighbor with that commander
        nan_entries = np.isnan(embedding).any(axis=1)

        if sum(nan_entries) > 0:
            disconnected_idx = np.where(nan_entries)[0]
            connected_idx = np.where(~nan_entries)[0]

            disconnected_assignments = []
            for d in disconnected_idx:
                d_deck = self.cdecks[d]
                decks_with_comm = [i for i, cdeck in self.cdecks.items(
                ) if cdeck.commander == d_deck.commander and i != d]

                if len(decks_with_comm):
                    jaccard_distances = [calculate.jaccard(
                        d_deck.cards, self.cdecks[i].cards) for i in decks_with_comm]
                    smallest_deck_idx = decks_with_comm[np.argmax(
                        jaccard_distances)]

                # if no decks with that commander exist, assign to a random point on the map
                else:
                    smallest_deck_idx = np.random.choice(connected_idx)

                disconnected_assignments.append(smallest_deck_idx)

            # assign to those decklists, plus some noise so its not an exact overlap
            noise = np.random.uniform(0.001, 0.002, size=(
                n_dims,)) * np.random.choice([1, -1], size=(n_dims,))
            embedding[disconnected_idx] = embedding[np.array(
                disconnected_assignments)] + noise

        if coordinates:
            self.coordinates = embedding
            self.commander_decks[['x', 'y']] = embedding.round(6)
        else:
            self.cluster_embedding = embedding

        return embedding

    def cluster_decks(self, method='HDBSCAN', **method_kwargs):
        '''Cluster decks according to a specific method, using self.cluster_embedding. Returns an array of cluster assignments.

        Params:
        -------
        method: str, which method to use. Valid methods include 'HDBSCAN', 'leiden', and 'hierarchical'. Note that some may be extremely slow with certain embeddings and data sizes. Each has their own defaults.
        method_kwargs: kwargs to pass to clustering method.

        Returns:
        --------
        cluster_labels: numpy array, the array of cluster assignments.
        '''
        np.random.seed(0)

        # make sure there is an embedding to cluster
        if self.cluster_embedding is None:
            raise ValueError(
                'Must run reduce_dimensionality(..., coordinates = False) before running cluster_decks')

        if method not in ['HDBSCAN']:
            raise NotImplementedError('Only HDBSCAN implemented.')

        # if there are very few decks, we cluster them ourselves
        if len(self.commander_decks) <= 10:
            print('Too few decks to cluster, assigning all to one cluster.')
            cluster_labels = np.zeros(len(self.commander_decks)).astype(int)
            self.cluster_labels = cluster_labels
            self.commander_decks['clusterID'] = cluster_labels
            return cluster_labels

        # HDBSCAN clustering - Leiden not currently supported
        if method == 'HDBSCAN':
            if 'min_cluster_size' not in method_kwargs:
                method_kwargs['min_cluster_size'] = 15

            if 'min_samples' not in method_kwargs:
                method_kwargs['min_samples'] = method_kwargs['min_cluster_size']

            # define and run the clusterer
            clusterer = hdbscan.HDBSCAN(prediction_data=True, gen_min_span_tree=True, core_dist_n_jobs=1,
                                        **method_kwargs)
            cluster_labels = clusterer.fit_predict(self.cluster_embedding)

            self.cluster_labels = cluster_labels
            self.commander_decks['clusterID'] = cluster_labels
            return cluster_labels
          
#         if method == 'leiden':

#             # pretty hackish here. n_neighbors is not a kwarg to the leiden fitting but we need it here.
#             n_neighbors = method_kwargs.pop('n_neighbors', 15)

#             # Leiden clustering requires a bit more work. It constructs the nearest neighbor graph itself.
#             knn_indices, knn_dists, forest = nearest_neighbors(
#                                                  self.decklist_matrix,
#                                                  n_neighbors=n_neighbors,
#                                                  random_state=random_state,
#                                                  metric='jaccard',
#                                                  metric_kwds={},
#                                                  angular=False,
#                                                  verbose=True)

#             # an empty adjacency matrix
#             X = coo_matrix(([], ([], [])), shape=(self.decklist_matrix.shape[0], 1))

#             # defaults for constructing the fuzzy simplicial set
#             set_op_mix_ratio = method_kwargs.pop('set_op_mix_ratio', 1)
#             local_connectivity = method_kwargs.pop('local_connectivity', 1)

#             # generate adjacency matrix
#             connectivities = fuzzy_simplicial_set(
#                                 X,
#                                 n_neighbors,
#                                 None,
#                                 None,
#                                 knn_indices=knn_indices,
#                                 knn_dists=knn_dists,
#                                 set_op_mix_ratio=set_op_mix_ratio,
#                                 local_connectivity=local_connectivity)

#             connectivities = connectivities[0]
#             connectivities = connectivities.tocsr()

#             # construct the igraph
#             g = cluster.get_igraph_from_adjacency(connectivities, directed=True)

#             # construct the partition
#             partition_type = leidenalg.RBConfigurationVertexPartition
#             weights = np.array(g.es['weight']).astype(np.float64)
#             resolution = method_kwargs.pop('resolution_parameter', 1)

#             set_op_mix_ratio = method_kwargs.pop('set_op_mix_ratio', 1)
#             part = leidenalg.find_partition(g, partition_type, weights = weights,
#                                 n_iterations = -1, seed = 0,
#                                 resolution_parameter = resolution)

#             cluster_labels = np.array(part.membership)
#             self.cluster_labels = cluster_labels
#             self.commander_decks['clusterID'] = cluster_labels
#             return cluster_labels

    def assign_unclustered(self, n_neighbors=50):
        '''If any decks are unassigned, assign them clusters based on nearest neighbors.

        Params:
        -------
        n_neighbors: number of nearest neighbors to examine

        Returns:
        --------
        cluster_labels: numpy array, array of cluster labels
        '''

        if self.cluster_labels is None or self.cluster_embedding is None:
            raise ValueError(
                'No cluster labels or embedding defined. Must run reduce_dimensionality and cluster_decks first.')

        unclustered_deck_count = sum(self.cluster_labels == -1)
        if unclustered_deck_count == 0:
            print('No decks unclustered, returning original assignments.')
            return self.cluster_labels

        # if they are all unclustered, assign them all to a single cluster
        elif unclustered_deck_count == len(self.cluster_labels):
            print('No decks clustered. Assigning all to one cluster.')
            cluster_labels = np.array([0]*len(self.cluster_labels))
            self.cluster_labels = cluster_labels
            self.commander_decks['clusterID'] = self.cluster_labels
            return cluster_labels

        # first, we separate out the clustered from unclustered data
        unclustered = np.where(self.cluster_labels == -1)[0]
        clustered = np.where(self.cluster_labels != -1)[0]

        clustered_embedding = self.cluster_embedding[clustered, :]
        unclustered_embedding = self.cluster_embedding[unclustered, :]

        # if the code is trying to look at more neighbors than there are points, we have to reduce
        if clustered_embedding.shape[0] < n_neighbors + 1:
            print('Fewer clustered decks than n_neighbors, must reduce')
            n_neighbors = clustered_embedding.shape[0] - 1

        # use a kdtree to find the cluster assignments of the nearest n neighbors
        kdtree = scipy.spatial.KDTree(clustered_embedding)
        _, indices = kdtree.query(unclustered_embedding, n_neighbors)

        # umap indices is a n x n_neighbors matrix, with each row representing the index of the n_neighbors closest neighbors.
        # We then convert these into clusters, and take the most common cluster.
        umap_indices = clustered[indices]
        cluster_assignments_neighbors = self.cluster_labels[umap_indices]
        cluster_assignments = np.array(
            [np.bincount(row).argmax() for row in cluster_assignments_neighbors])

        # assign the unclustered points to these clusters
        self.cluster_labels[unclustered] = cluster_assignments
        self.commander_decks['clusterID'] = self.cluster_labels

        return self.cluster_labels

    def get_cluster_traits(self, topn=20, min_perc=1, drop_categories=(), columns=('commander-partnerID',
                           'colorIdentityID', 'themeID', 'tribeID')):
        '''Given a df with a required column 'clusterID' and optional columns
        [commander-partner, tribe, theme, colorIdentity], return a dataframe
        giving the characteristics of that cluster. If submap_type is given,
        that type will not be included in the dataframe.

        Params:
        -------
        topn: int, number of traits to show for each category
        min_perc: int, traits with representation below min_perc will not be shown
        drop_categories: iterable, if any categories should be excluded
        columns: iterable, which categories should be examined

        Returns:
        --------
        cluster_traits: df, a dataframe depicting the most represented traits for each cluster
        '''

        # some checks
        if 'clusterID' not in self.commander_decks.columns:
            raise NotImplementedError('Must run cluster_decks first.')

        # create a copy
        func_df = self.commander_decks.copy()

        missing_cols = set(columns) - set(func_df.columns)
        if len(missing_cols):
            raise ValueError(f'Missing the following columns: {missing_cols}')

        # drop the categories as specified
        if drop_categories:
            columns = [col for col in columns if col not in drop_categories]

        # get number of clusters
        num_clusters = len(set(func_df['clusterID']))

        # now we do the grouping
        output_df = []
        for col in columns:
            groups = func_df.groupby(['clusterID', col]).size().to_frame(
                'percent').reset_index()

            # if there is one cluster, have to do something different
            if num_clusters == 1:
                groups['percent'] = (
                    groups['percent'] / sum(groups['percent']) * 100).round(0).astype(int).values
            else:
                groups['percent'] = groups.groupby(['clusterID']).apply(
                    lambda x: (x['percent']/sum(x['percent']) * 100)).round(0).astype(int).values

            groups = groups.sort_values(
                by=['clusterID', 'percent'], ascending=[True, False])

            # we drop fields if they're less than min_perc
            groups = groups[groups['percent'] >= min_perc]

            # keep at maximum only the top n commanders for each cluster
            groups = groups.groupby('clusterID').head(
                topn).reset_index(drop=True)
            groups = groups.rename(columns={col: 'value'})
            groups['category'] = col

            output_df.append(
                groups[['clusterID', 'category', 'value', 'percent']])

        # combine data
        output_df = pd.concat(output_df).reset_index(drop=True)

        self.cluster_traits = output_df
        return output_df

    def get_cluster_card_counts(self, color_rule, include_commanders,
                                chunksize=1000, precomputed_noncard=None, verbose=False):
        '''Given a dataframe with cluster assignments, a sparse decklist matrix, a date matrix, a color identity matrix with appropriate index lookups, calculate and return A) the number of decks in each cluster that play each card, and B) the number of decks in each cluster that COULD play the card but do not.

        Params:
        -------
        color_rule: str, dictates what decks are included in a count. 
            'ignore' means to ignore color identity, not restricting deck counts at all
            'superset' (or anything else) means to restrict counts by color identity
        include_commanders: a boolean for whether commanders should be included in counts
        chunksize: int, size of chunks for determining cluster noncounts. 
            Required for large datasets to prevent memory issues.
        precomputed_noncard: dict, a dict of precomputed Pandas series of number of decks that could play a card but do not.
            Key should be the cluster, and value should be the Series with index of cards and value of decks.

        Returns:
        --------
        cluster_card_df: df, contains the number of decks that play each card for each cluster
        cluster_noncard_df: df, contains the number of decks that COULD play each card for each cluster but do not.
        '''

        # some checks
        for ref in [self.date_matrix, self.ci_matrix, self.card_idx_lookup, self.deck_date_idx_lookup, self.card_date_idx_lookup, self.deck_ci_idx_lookup, self.card_ci_idx_lookup]:
            if ref is None:
                raise ValueError(
                    'All reference lookups must be defined. Some are not.')

        if 'clusterID' not in self.commander_decks.columns:
            raise NotImplementedError('Must run cluster_decks first.')

        if precomputed_noncard:
            for clust, noncard_clust in precomputed_noncard.items():
                if list(noncard_clust.index) != list(self.card_idx_lookup.keys()):
                    raise ValueError(
                        f'There are some card mismatches between the precomputed noncard df and the cards in card_idx for cluster {clust}. Check that these are equivalent.')
        else:
            precomputed_noncard = {}

        # first we define a n x m matrix of card counts for each cluster
        cluster_card_df = np.zeros(
            shape=(self.commander_decks['clusterID'].max() + 1, len(self.card_idx_lookup)))

        if verbose:
            print('Building the cluster counts...\n\tConstructing the number of decks that play each card in each cluster. On cluster...', end='')

        # loop through each cluster to determine the card counts
        clusters = sorted(list(set(self.commander_decks['clusterID'])))
        breaks = len(clusters) // 20 if len(clusters) >= 20 else 1
        for clust in clusters:
            if (clust % breaks == 0 or clust == -1) and verbose:
                print(clust, end=', ')

            # extract out the cluster and add in the number of decks that play each card
            cluster_plot_df = self.commander_decks[lambda row: row['clusterID'] == clust]
            cluster_card_counts = self.decklist_matrix[cluster_plot_df.index, :].sum(
                axis=0)
            cluster_card_df[clust, :] = cluster_card_counts

            # remove commanders and partners if needed. First tally the commanders and partners in the cluster.
            if include_commanders:
                commander_partners = cluster_plot_df['commanderID'].value_counts(
                )
                commander_partners = commander_partners.add(
                    cluster_plot_df['partnerID'].value_counts(), fill_value=0)
                commander_partners = commander_partners.add(
                    cluster_plot_df['companionID'].value_counts(), fill_value=0)

                # we drop partners that are absent
                commander_partners = commander_partners.drop(
                    index=[''], errors='ignore')

                # lookup the commander and partner indices and subtract them
                commander_partner_indices = [
                    self.card_idx_lookup[cardname] for cardname in commander_partners.index]
                cluster_card_df[clust,
                                commander_partner_indices] -= commander_partners.values

        # make the cluster card df - the number of decks in each cluster that play each card.
        cluster_card_df = pd.DataFrame(cluster_card_df, index=clusters,
                                       columns=list(self.card_idx_lookup.keys()))

        if verbose:
            print('done')

        # next, we cross reference by date (and optionally by color identity) to determine
        # the number of decks that COULD POSSIBLY play each card.

        # construct the empty dataframe
        cluster_noncard_df = np.zeros(
            shape=(self.commander_decks['clusterID'].max() + 1, len(self.card_idx_lookup)))

        if verbose:
            print('\tConstructing number of decks that COULD play each card, controlling for date and optionally color identity. On cluster...', end='')

        # lookup the date and color identity of each card in question
        card_dates = np.array([self.card_date_idx_lookup[cardname]
                               for cardname in self.card_idx_lookup.keys()]).reshape(1, -1)
        card_cis = np.array([self.card_ci_idx_lookup[cardname]
                             for cardname in self.card_idx_lookup.keys()]).reshape(1, -1)

        # loop through each cluster, determining the number of decks that COULD play each card
        for clust in clusters:
            if (clust % breaks == 0 or clust == -1) and verbose:
                print(clust, end=', ')

            # first we check if we precomputed anything for this cluster
            if clust in precomputed_noncard.keys():
                cluster_noncard_df[clust,
                                   :] = precomputed_noncard[clust].values
                continue

            # otherwise, compute the cluster's data
            cluster_plot_df = self.commander_decks[lambda row: row['clusterID'] == clust]

            # to prevent subsetting into the date and ci matrix with very large amounts of data,
            # we split the deck dates and color identity lists into chunks. Slower, but lower risk of memory errors
            deck_dates = np.array([self.deck_date_idx_lookup[deck_id]
                                   for deck_id in cluster_plot_df['deckID']]).reshape(-1, 1)

            deck_date_chunks = np.array_split(
                deck_dates, deck_dates.shape[0] // chunksize + 1)

            if color_rule != 'ignore':
                deck_cis = np.array([self.deck_ci_idx_lookup[deck_id]
                                     for deck_id in cluster_plot_df['deckID']]).reshape(-1, 1)
                deck_ci_chunks = np.array_split(
                    deck_cis, deck_cis.shape[0] // chunksize + 1)

            # loop through chunks, tallying how many decks could play a card but do not
            total_can_play = np.zeros(shape=(len(self.card_idx_lookup)))
            for i, date_chunk in enumerate(deck_date_chunks):
                date_chunk_play = self.date_matrix[date_chunk, card_dates]

                if color_rule != 'ignore':
                    ci_chunk = deck_ci_chunks[i]
                    ci_chunk_play = self.ci_matrix[ci_chunk, card_cis]

                # if we ignore color identity, we declare that the decks can always play each card based on ci
                else:
                    ci_chunk_play = np.ones(shape=date_chunk_play.shape)

                # enforce that the deck can play it by date AND color identity
                total_can_play += np.logical_and(date_chunk_play,
                                                 ci_chunk_play).sum(axis=0)

            cluster_noncard_df[clust, :] = total_can_play

        # finally, we subtract out the number that do play from the number that can play to get the number that don't.
        # sometimes, due to errors in release dates, we can end up with negative numbers here. We clip these to 0.
        cluster_noncard_df = (cluster_noncard_df -
                              cluster_card_df).clip(lower=0)

        if verbose:
            print('done')

        self.cluster_card_df = cluster_card_df
        self.cluster_noncard_df = cluster_noncard_df

        return cluster_card_df, cluster_noncard_df

    def get_defining_cards(self, include_synergy=True, n_scope=200, verbose=False):
        '''Extract cards that define each cluster relative to all others.

        Params:
        -------
        include_synergy: whether synergy should be calculated. Synergy is never included 
            if there is only one cluster
        n_scope: how many cards to calculate synergy for (top 200 by default)

        Returns:
        --------
        defining_cards: df, a combined dataframe of cluster defining cards
        '''

        # some checks
        if 'clusterID' not in self.commander_decks.columns:
            raise NotImplementedError('Must run cluster_decks first.')

        if self.cluster_card_df is None:
            raise NotImplementedError(
                'Must run get_cluster_card_counts first.')

        # we don't include synergy if there's a single cluster
        if len(set(self.commander_decks['clusterID'])) == 1:
            include_synergy = False

        if verbose:
            print('\tCalculating card synergies for clusters. On cluster...', end='')

        # calculate the play rate of each card in each cluster
        play_rate_df = (
            self.cluster_card_df/(self.cluster_card_df + self.cluster_noncard_df)).fillna(0)

        if include_synergy:

            # calculate the play rate of each card in each OTHER cluster
            play_other_cluster = self.cluster_card_df.sum(
                axis=0) - self.cluster_card_df

            # the number of decks that could play each card in each other cluster but do not:
            nonplay_other_cluster = self.cluster_noncard_df.sum(
                axis=0) - self.cluster_noncard_df

            # the play rate in other clusters. We assume 0/0 = 0.
            other_play_rate = (
                play_other_cluster / (nonplay_other_cluster + play_other_cluster)).fillna(0)

            # calculate cluster synergies!
            synergies = play_rate_df - other_play_rate

        # now we loop through each cluster and calculate the synergy of its top n cards
        clusters = sorted(list(set(self.commander_decks['clusterID'])))
        combined_output_df = []
        for clust in clusters:
            if clust % 100 == 0 and verbose:
                print(clust, end=', ')

            cluster_play_rates = play_rate_df.loc[clust]

            # get top n cards
            topn_cards = cluster_play_rates.sort_values(
                ascending=False).head(n_scope).index

            # fill in dataframe
            output_df = pd.DataFrame()
            output_df['card'] = topn_cards
            output_df.insert(0, 'clusterID', clust)

            # fill in play rate
            output_df['play_rate'] = cluster_play_rates.loc[topn_cards].round(
                2).values

            # fill in synergies if necessary
            if include_synergy:
                output_df['synergy'] = synergies.loc[clust,
                                                     topn_cards].round(2).values
                output_df = output_df.sort_values(
                    by='synergy', ascending=False)
            else:
                output_df = output_df.sort_values(
                    by='play_rate', ascending=False)

            # add to dataframe
            combined_output_df.append(output_df)

        # combine all the cluster results
        combined_output_df = pd.concat(combined_output_df)

        if verbose:
            print('done')

        self.cluster_defining_cards = combined_output_df
        return combined_output_df

    def calculate_average_decklists(self, ignore_clusters=(), verbose=False):
        '''Returns average decklists for each cluster. The "average" deck is the one with the highest average synergy.

        Params:
        -------
        verbose: bool, whether output should be verbose
        ignore_clusters: iterable, whether certain clusters should be ignored. Used for skipping large clusters
            we don't care about.

        Returns:
        --------
        average_decklists: dict, a dictionary mapping cluster to average deck
        '''

        if self.cluster_defining_cards is None:
            raise NotImplementedError('Must run get_defining_cards first.')

        if verbose:
            print('Calculating average decklists. On cluster...', end='')

        # column index to cardname
        idx_to_card_lookup = {v: k for k, v in self.card_idx_lookup.items()}

        # clusters
        clusters = sorted(list(set(self.commander_decks['clusterID'])))

        # loop through all the clusters and calculate their average decks
        average_decklists = {}
        for clust in clusters:

            # ignore some clusters (us)
            if clust in ignore_clusters:
                continue

            if clust % 100 == 0 and verbose:
                print(clust, end=', ')

            # get the cluster data, defining cards, and decklists
            clust_data = self.commander_decks[self.commander_decks['clusterID'] == clust]
            clust_defining_cards = self.cluster_defining_cards[
                self.cluster_defining_cards['clusterID'] == clust]
            clust_decklists = self.decklist_matrix[clust_data.index]

            # calculate synergies and the size of the list. If synergies are absent, we use play rate
            if 'synergy' in clust_defining_cards.columns:
                val_lookup = dict(
                    zip(clust_defining_cards['card'], clust_defining_cards['synergy']))
            else:
                val_lookup = dict(
                    zip(clust_defining_cards['card'], clust_defining_cards['play_rate']))

            # for each decklist in the cluster, calculate the average synergy of that decklist.
            # cards without a synergy score get 0.
            scores, list_size = [], []
            for decklist in clust_decklists.tolil().rows:
                cardnames = [idx_to_card_lookup[idx] for idx in decklist]
                avg_score = np.mean([val_lookup.get(c, 0) for c in cardnames])
                scores.append(avg_score)
                list_size.append(len(cardnames))

            # find the decklist with the highest synergy that is between 20 and 80 percentiles of the list length
            # this is done to enforce a normal basic land count (otherwise a deck with 1 high synergy card would win here)
            score_df = pd.DataFrame(
                [scores, list_size], index=['score', 'size']).T
            score_df_with_lands = score_df[score_df['size'].between(
                *np.percentile(score_df['size'], [20, 80]))]

            # sometimes this df can be empty if the number of decks is very small,
            # in which case just keep all the decks
            if len(score_df_with_lands) == 0:
                score_df_with_lands = score_df.copy()

            # identify the decklist with the highest mean synergy and get its decklist and info
            max_synergy_idx = score_df_with_lands['score'].idxmax()
            deck_id = clust_data.iloc[max_synergy_idx]['deckID']
            average_decklists[clust] = deck_id

        if verbose:
            print('done')

        # the deck id of the average decklists (for each cluster)
        self.average_decklists = average_decklists

        return average_decklists

    def jsonify_map(self, magic_cards, clusters=None, trait_mapping=None):
        '''Converts map data of specific clusters to an exportable json. Note
        that the export format differs if one cluster is provided versus many

        Params:
        -------
        clusters: iterable, which clusters to export to a json. If None, export all.
        magic_cards: dict, used to look up card traits
        trait_mapping: dict, a mapping of category value to integer (for data compression purposes)

        Returns:
        --------
        map_json: dict, containing json data
        '''

        if clusters is None:
            clusters = sorted(set(self.cluster_labels))

        # obtain and subset data
        defining_cards = self.cluster_defining_cards[self.cluster_defining_cards['clusterID'].isin(
            clusters)]
        traits = self.cluster_traits[self.cluster_traits['clusterID'].isin(
            clusters)]
        commander_decks = self.commander_decks[self.commander_decks['clusterID'].isin(
            clusters)]

        # some reference values
        rename_dict = {'commander-partnerID': 'commanderID'}
        basics = ['Plains', 'Island', 'Swamp', 'Mountain', 'Forest']

        # drop basics from cluster defining cards
        defining_cards = defining_cards[~defining_cards['card'].isin(basics)]

        # drop empty traits
        traits = traits[~((traits['category'].isin(['tribeID', 'themeID'])) &
                          (traits['value'] == ''))]

        clusters = sorted(set(traits['clusterID']))

        # preextract average prices for faster lookup
        average_prices = commander_decks[(np.isfinite(commander_decks['price']))].groupby([
            'clusterID'])['price'].mean()

        # get the data from each cluster
        json_data = []
        for clust in clusters:

            # make the cluster json data, specifying cluster number if greater than 1.
            cluster_json_data = defaultdict(list)
            if len(clusters) > 1:
                cluster_json_data['clusterID'] = clust

            cluster_defining_cards = defining_cards[defining_cards['clusterID'] == clust].drop(
                columns=['clusterID'])
            cluster_traits = traits[traits['clusterID']
                                    == clust].drop(columns=['clusterID'])
            cluster_average_price = average_prices.loc[clust]
            cluster_average_deck_id = self.average_decklists[clust]

            # add in the cluster traits
            for _, row in cluster_traits.iterrows():
                category, value, percent = row.values
                category = rename_dict.get(category, category)

                # Convert value to int if a trait_mapping is given, handling commander-partner
                # e.g. "Thrasios + Tymna" > "342 + 431"
                if trait_mapping:
                    if category == 'commanderID' and ' + ' in value:
                        c_p_vals = value.split(' + ')
                        c_p_vals = [trait_mapping['commanderID'][cardname]
                                    for cardname in c_p_vals]
                        value = str(c_p_vals[0]) + '+' + str(c_p_vals[1])
                    else:
                        value = str(trait_mapping[category][value])

                cluster_json_data[category].append([value, percent])

            # add in the defining cards
            for _, row in cluster_defining_cards.iterrows():
                cluster_json_data['definingCards'].append(list(row.values))

            # add in the average price
            cluster_json_data['averagePrice'] = int(cluster_average_price)

            # add in the average deckID
            cluster_json_data['averageDeck'] = str(cluster_average_deck_id)

            # finally, add the cluster data
            json_data.append(dict(cluster_json_data))

        # if there is only one cluster, we simply return that cluster
        if len(json_data) == 1:
            json_data = json_data[0]

        return json_data
