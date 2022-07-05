import itertools
import inflect
import re


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))


def jaccard(list_1, list_2):
    return len(set(list_1) & set(list_2)) / len(set(list_1).union(set(list_2)))
    
def get_parameters(ndecks):

    if ndecks < 300:
        nneighbors = 4
        minclustersize = 4

    elif ndecks < 500:
        nneighbors = 6
        minclustersize = 6

    elif ndecks < 1000:
        nneighbors = 8
        minclustersize = 6

    elif ndecks < 2000:
        nneighbors = 10
        minclustersize = 6

    elif ndecks < 5000:
        nneighbors = 12
        minclustersize = 6

    elif ndecks < 20000:
        nneighbors = 15
        minclustersize = 8

    else:
        nneighbors = 25
        minclustersize = 12

    return nneighbors, minclustersize


def format_tribe_theme_slug(string, category):
    '''Formats a slug string for display on the website'''

    if category == 'themeID':
        string = string.replace('-', ' ')
        string = ' '.join([c.capitalize() for c in string.split(' ')])
        return string

    elif category == 'tribeID':
        p = inflect.engine()
        return p.plural(string)

    else:
        raise NotImplementedError(
            f'category must tribe or theme, is {category}')


def replace_traits_with_ints(commander_decks, trait_mapping):
    '''Uses a trait mapping to replace the commander, partner, tribe, theme, and colorIdentity
    with ints'''

    commander_decks = commander_decks.copy()
    for field in ['colorIdentityID', 'commanderID', 'partnerID', 'themeID', 'tribeID']:

        # get the mapping. The mapping for the partner and companion is commander
        replace_dict = trait_mapping[field] if field != 'partnerID' else trait_mapping['commanderID']

        # we utilize a lambda to avoid empty partners, tribes, and themes
        # but still throw an error on a missing key. Color identities have an empty key (colorless)
        if field != 'colorIdentityID':
            def mapping_func(
                value): return replace_dict[value] if value != '' else ''
        else:
            def mapping_func(value): return replace_dict[value]

        commander_decks[field] = commander_decks[field].astype(
            str).apply(mapping_func)

    return commander_decks


def extract_source_from_url(url):

    # we remove www which occurs in mtggoldfish data, and get the site from https://site.com/....
    source = url.replace('www.', '').split('/')[2].split('.')[0]
    return source


def fetch_decklist_ids_from_url(url, source):

    # the structure for deckstats is https://deckstats.net/decks/user_id/deck_id(-title).
    # we extract user_id/deck_id
    if source == 'deckstats':
        match = re.search(
            'https://deckstats.net/decks/([0-9]*/[0-9]*)\-.*', url)
        url_id = match.group(1)

    # moxfield is https://moxfield.com/decks/deck_id(#XXX). We extract deck_id
    elif source == 'moxfield':
        match = re.search('https://moxfield.com/decks/([^#]*)', url)
        url_id = match.group(1)

    # moxfield is http://www.mtggoldfish.com/deck/deck_id(#XXX). We extract deck_id
    elif source == 'mtggoldfish':
        match = re.search('http://www.mtggoldfish.com/deck/([^#]*)', url)
        url_id = match.group(1)

    # aetherhub is http://aetherhub.com/Deck/Public/deck_id(#XXX). We extract deck_id
    elif source == 'aetherhub':
        match = re.search('http://aetherhub.com/Deck/Public/([^#]*)', url)
        url_id = match.group(1)

    # archidekt is https://archidekt.com/decks/1923808(#XXX). We extract deck_id
    elif source == 'archidekt':
        match = re.search('https://archidekt.com/decks/([^#]*)', url)
        url_id = match.group(1)

    else:
        raise NotImplementedError(f'unrecognized source {source}')

    return url_id
