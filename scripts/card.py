import requests
import json
import string
from collections import defaultdict
import numpy as np
from datetime import datetime, timedelta
import re
import pydash

def fetch_cards(scryfall_url, replace_json=None, lower=False):
    '''Fetches the default cards from Scryfall, renames to match naming if 
    given a replacement json'
    
    Params:
    -------
    scryfall_url: str, scryfall url to lookup
    replace_json: path to json file of cardname mappings
    lower: bool, whether cardnames should be lowered

    Returns:
    --------
    magic_cards: dict, a dictionary mapping card name to card info from scryfall
    '''
    
    if replace_json:
        rename_dict = json.load(open(replace_json))
    else:
        rename_dict = {}

    resp = requests.get(scryfall_url)
    json_data = resp.json()

    magic_cards = {}
    for card_data in json_data:
        
        # skip if its a token
        if card_data['layout'] == 'token': 
            continue
        
        if lower:
            name = card_data['name'].lower()
        else:
            name = card_data['name']
        
        magic_cards[name] = card_data
        
        # if a transform/flip card, only take front half name
        if card_data['layout'] in ['flip', 'transform', 'modal_dfc', 'adventure']:
            left_name, right_name = name.split(' // ')

            for subname in [left_name, right_name]:
                magic_cards[subname] = card_data['card_faces'][0]
                magic_cards[subname]['cmc'] = card_data['cmc']
                magic_cards[subname]['color_identity'] = card_data['color_identity']
                magic_cards[subname]['original_name'] = name
            
                # rename if appropriate. We try to handle both cases here to prevent having to be case
                # specific in the rename json
                if subname in rename_dict:
                    correct_spelling = rename_dict[subname]
                    magic_cards[correct_spelling] = magic_cards[subname]

        # rename some cards
        if name in rename_dict or name.lower() in rename_dict:
            correct_spelling = rename_dict[name.lower()]
            
            if not lower:
                correct_spelling = string.capwords(correct_spelling.lower())

            magic_cards[correct_spelling] = magic_cards[name]
            
    # some cards are not tokens
    for cardname in ['Llanowar Elves', 'Ajani\'s Pridemate', 'Cloud Sprite', 'Storm Crow', 'Goldmeadow Harrier', 
                     'Kobolds of Kher Keep', 'Festering Goblin', 'Metallic Sliver', 'Spark Elemental', 'Storm Crow']:
        magic_cards[cardname]['layout'] = 'normal'
        magic_cards[cardname]['set_type'] = 'normal'
        
    # we also edit the color identities here
    sort_ci = lambda ci: ''.join(sorted(ci, key = lambda c: ['W', 'U', 'B', 'R', 'G'].index(c)))
    for key in magic_cards:
        card_ci = magic_cards[key]['color_identity']
        magic_cards[key]['color_identity'] = sort_ci(card_ci)

    return magic_cards

def find_price_and_release(magic_cards, default_card_json):
    '''Takes in the output of fetch_cards (magic_cards) and uses the default card json
    from scryfall to determine the lowest price of each card.
    
    Params:
    -------
    magic_cards: dict, output of card.fetch_cards()
    default_card_json: json file from scryfall default cards
    '''
    
    default_scryfall_cardnames = set([jdata['name'] for jdata in default_card_json])
    
    # first, we generate a lookup for prices based on the default cards
    price_lookup = defaultdict(list)
    release_lookup = defaultdict(list)
    
    # we define which fields are acceptable for prices. We use eur only when no other option is available
    price_fields = ['usd', 'usd_foil', 'usd_etched', 'eur', 'eur_foil']
    
    # we loop through all magic cards, storing the minimum price and release date of each
    for card_json in default_card_json:
        cardname = card_json['name']
        
        # get the minimum price from all printings
        min_price = np.inf
        for field in price_fields:
            price = card_json.get('prices', {field: np.nan})[field]
            
            # only use euro if no other usd currency is listed
            if price == price and price is not None and 'eur' in field:
                price = 1.13 * float(price)

            if price is not None and float(price) < min_price:
                min_price = float(price)
        
        # sometimes prices are correct, in which case store np.nan (we use nansum later)
        if price != np.inf:
            price_lookup[cardname].append(float(min_price))
        else:
            price_lookup[cardname].append(np.nan)
        
        # next we identify the release date. If the card was spoiled, use that date.
        if 'preview' in card_json:
            release_date = card_json['preview']['previewed_at']
            release_date = datetime.strptime(release_date, '%Y-%m-%d')
        
        # Otherwise, use the actual release date minus two weeks (to account for near spoilers)
        else: 
            release_date = card_json.get('released_at', '')
            release_date = datetime.strptime(release_date, '%Y-%m-%d') - timedelta(14)

        release_lookup[cardname].append(release_date)

    min_price_lookup = {cardname: (min(prices) if min(prices) != np.inf else np.nan) for cardname, prices in price_lookup.items()}
    earliest_release_lookup = {cardname: datetime.strftime(min(release_dates), '%Y-%m-%d') for cardname, release_dates in release_lookup.items()}
    
    # we then use this lookup to add to our oracle cards
    for cardname in magic_cards:
    
        # scryfall stores the full name of 'flip', 'transform', 'modal_dfc', 'adventure' cards. Since edhrec
        # doesn't, we don't when defining the lookups. So we need the original name back.
        if cardname not in default_scryfall_cardnames:
            lookup_name = magic_cards[cardname]['original_name']

        else:
            lookup_name = cardname

        magic_cards[cardname]['min_price'] = min_price_lookup.get(lookup_name, np.nan)
        magic_cards[cardname]['earliest_release'] = earliest_release_lookup.get(lookup_name, '')
        
    return magic_cards

# def extract_mana_cost(mana_value)
def extract_types(name, type_line):
    '''Extracts all the card type given a type line'''

    possible_card_types = ['Land', 'Creature', 'Sorcery', 'Instant', 
                                  'Artifact', 'Enchantment', 'Planeswalker']
    
    # the first type out of the list above determines where the card lies
    main_card_type = [c_type for c_type in possible_card_types if c_type in type_line][0]

    # determine basic or nonbasic
    basics = ['Mountain', 'Forest', 'Island', 'Plains', 'Swamp', 'Wastes']
    basics += ['Snow-Covered ' + bas for bas in basics]
    if main_card_type == 'Land':
        if name in basics:
            main_card_type = 'Basic Land'
        else:
            main_card_type = 'Nonbasic Land'

    return main_card_type

def kebab(string):
    '''Converts a string into its kebab form, following Javascript'''
    
    string = pydash.deburr(string)
    string = string.replace("'", "")
    words = re.split('[^a-zA-Z0-9]', string)
    words = [w.lower() for w in words if w]
    return '-'.join(words)