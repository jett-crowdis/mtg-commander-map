import re

# these functions take in a card and return if it can be played in a deck that companions the companion


def gyruda(cardname, magic_cards):
    card_info = magic_cards[cardname]
    return card_info['cmc'] % 2 == 0


def jegantha(cardname, magic_cards):
    card_info = magic_cards[cardname]

    # if there are card faces, only the front matters
    if 'card_faces' in card_info:
        mc = card_info['card_faces'][0]['mana_cost']
    else:
        mc = card_info['mana_cost']

    symbols_in_mc = [mc for mc in re.split("[^0-9A-Z\/]", mc) if mc != '']
    return len(symbols_in_mc) == len(set(symbols_in_mc))


def kaheera(cardname, magic_cards):
    card_info = magic_cards[cardname]
    creature_types = ['Cat', 'Elemental', 'Nightmare', 'Dinosaur', 'Beast']
    type_line = card_info['type_line']
    if 'Creature' in type_line:
        return any([c_type in type_line for c_type in creature_types])
    else:
        return 1


def keruga(cardname, magic_cards):
    card_info = magic_cards[cardname]
    return 'Land' in card_info['type_line'] or card_info['cmc'] >= 3


def lurrus(cardname, magic_cards):
    card_info = magic_cards[cardname]
    perms = ['Land', 'Creature', 'Enchantment', 'Artifact', 'Planeswalker']

    # check if it's a permanent
    if any([p in card_info['type_line'] for p in perms]):
        return card_info['cmc'] <= 2
    else:
        return 1


def obosh(cardname, magic_cards):
    card_info = magic_cards[cardname]
    if 'Land' in card_info['type_line']:
        return 1
    else:
        return card_info['cmc'] % 2 == 1

# zirda's is very annoying


def zirda(cardname, magic_cards):

    card_info = magic_cards[cardname]

    # we first hardcode a number of abilities that imply an activated ability
    activated_keywords = ['equip', 'cycling', 'transfigure', 'unearth', 'levelup', 'outlast', 'crew', 'ninjutsu', 'commanderninjutsu', 'transmute',
                          'forceast', 'auraswap', 'reinforce', 'scavenge', 'embalm', 'eternalize', 'fortify']

    # then we check if the card is a permanent
    perms = ['Land', 'Creature', 'Enchantment', 'Artifact', 'Planeswalker']

    # if it isn't, we simply return 1 (all nonpermanents are playable in zirda)
    if not any([p in card_info['type_line'] for p in perms]):
        return 1

    # if it is, we need to check if it has an activated ability. We follow scryfall here
    else:

        # some keywords always imply an activated ability
        card_kw = [kw.lower() for kw in card_info.get('keywords')]
        if any([kw in card_kw for kw in activated_keywords]):
            return 1

        # if that doesn't work, we extract all the lines of the card
        lines = []

        # handle dfcs
        if 'card_faces' in card_info:
            lines = [line for face in card_info['card_faces']
                     for line in face['oracle_text'].split('\n')]

        else:
            lines = card_info['oracle_text'].split('\n')

        # apply a regex designed to detect activated abilities
        matches = [re.search('^[^"]+:.+$', l) for l in lines]
        return any(matches)

# umori is a little bit trickier. Its restriction applies to the whole deck


def umori(card_list, magic_cards):

    nonlands = [
        c for c in card_list if 'Land' not in magic_cards[c]['type_line']]
    possible_card_types = ['Artifact', 'Creature', 'Land',
                           'Enchantment', 'Planeswalker', 'Instant', 'Sorcery']

    # we go through each card in the nonland card list. If at any point
    # a card doesn't share a type with a previous card, we simply return 0
    shared_types = possible_card_types
    for cardname in nonlands:
        card_info = magic_cards[cardname]
        shared_types = [
            card_type for card_type in shared_types if card_type in card_info['type_line']]
        if len(shared_types) == 0:
            return 0

    return 1


def playable_in_companion(companion, magic_cards, cardname=None, card_list=None):

    if cardname is None and card_list is None:
        raise NotImplementedError('Must define one of cardname or card_list')

    if card_list is not None and companion != 'Umori, the Collector':
        raise NotImplementedError('Only card lists allowed for Umori')

    if companion == 'Gyruda, Doom of Depths':
        return gyruda(cardname, magic_cards)

    elif companion == 'Jegantha, the Wellspring':
        return jegantha(cardname, magic_cards)

    elif companion == 'Kaheera, the Orphanguard':
        return kaheera(cardname, magic_cards)

    elif companion == 'Keruga, the Macrosage':
        return keruga(cardname, magic_cards)

    elif companion == 'Lurrus of the Dream-Den':
        return lurrus(cardname, magic_cards)

    elif companion == 'Obosh, the Preypiercer':
        return obosh(cardname, magic_cards)

    elif companion == 'Zirda, the Dawnwaker':
        return zirda(cardname, magic_cards)

    elif companion == 'Umori, the Collector':
        return umori(card_list, magic_cards)

    else:
        raise NotImplementedError(f'Unrecognized companion {companion}')


def calculate_companion(cdeck, magic_cards):
    '''Calculates the companion associated with a commander deck

    Params:
    -------
    cdeck: CommanderDeck object
    magic_cards: dict mapping cardname to properties
    '''

    companions = ['Gyruda, Doom of Depths', 'Jegantha, the Wellspring', 'Kaheera, the Orphanguard', 'Keruga, the Macrosage',
                  'Lurrus of the Dream-Den', 'Obosh, the Preypiercer', 'Umori, the Collector', 'Zirda, the Dawnwaker']

    # check each companion
    deck_companions = []
    for comp in companions:
        copied_card_list = cdeck.cards.copy()

        # if the commander is the companion, it's not a companioned deck.
        # If not, the commander still needs to match the companion requirement
        if comp == cdeck.commander:
            continue
        else:
            copied_card_list.append(cdeck.commander)

        # if the partner is a companion, just register it as such. If it's not
        # but nonempty, add it to the card list
        if cdeck.partner == comp:
            deck_companions.append(comp)
            continue
        elif cdeck.partner != '':
            copied_card_list.append(cdeck.partner)

        # if the theme involves that companion, register it
        if cdeck.theme == re.split(',| ', comp)[0].lower() + '-companion':
            deck_companions.append(comp)
            continue

        # if the deck can't play the companion based on color identity, continue
        if not all([c in cdeck.colorIdentity for c in magic_cards[comp]['color_identity']]):
            continue

        # next we check if the companion is in the deck. If so, we check that the deck could companion the card
        if comp in cdeck.cards:

            # next we verify whether the decklist can play the companion. Buckle up
            # first we get all the original names of cards
            copied_card_list = [magic_cards[cardname].get(
                'original_name', cardname) for cardname in copied_card_list]

            # next we check if all the cards satisfy the deck's restriction
            if comp == 'Umori, the Collector':
                all_companionable = playable_in_companion(
                    comp, magic_cards, card_list=copied_card_list)
            else:
                companionable_cards = [playable_in_companion(
                    comp, magic_cards, cardname=c) for c in copied_card_list]
                all_companionable = all(companionable_cards)

            if all_companionable:
                deck_companions.append(comp)

    # we take the first companion
    c_assignment = deck_companions[0] if len(deck_companions) else ''
    return c_assignment
