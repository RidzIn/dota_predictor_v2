import json

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from rich import print as pp


def read_heroes_prior():
    """
    Read hero priorities from 'parser/heroes_prior.txt' and return a dictionary
    with hero names as keys and their respective priorities as values.

    Returns:
        dict: A dictionary containing hero names as keys and their priorities as values.

    Raises:
        FileNotFoundError: If the 'parser/heroes_prior.txt' file does not exist.
        json.JSONDecodeError: If there's an issue decoding JSON data from the file.
    """
    try:
        hero_prior_dict = {}
        with open("heroes_prior.txt", "r") as f:
            lines = f.readlines()

        for i in range(len(lines)):
            lines[i] = lines[i].strip()

        for line in lines:
            hero, prior_json = line.split(" | ")
            try:
                hero_prior_dict[hero] = json.loads(prior_json)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON data for hero '{hero}': {str(e)}")

        return hero_prior_dict

    except FileNotFoundError:
        raise FileNotFoundError("heroes_prior.txt' file does not exist.")


def reshape_pick(pick):
    """
    Reshape the provided "pick" list based on hero priorities and create a new dictionary.

    Args:
        pick (list): A list containing hero names in the order they were picked.

    Returns:
        dict: A dictionary with hero positions (1 to 5) as keys and their corresponding names as values.

    Note:
        The hero priorities are obtained from the 'read_heroes_prior' function.
    """

    hero_prior = read_heroes_prior()

    priors_list = []
    hero_dict = {}
    counter = 1
    for i in pick:
        hero_dict[counter] = i
        counter += 1
        priors_list.append(hero_prior[i])
    transposed = list(zip(*priors_list))
    transposed = [list(row) for row in transposed]
    new_hero_dict = {}

    for i in range(len(transposed)):
        counter = 1
        for k in transposed[i]:
            if 1 == k:
                new_hero_dict[1] = hero_dict.get(counter)
                for n in transposed:
                    n[counter - 1] = 0
                break
            counter += 1
        try:
            if new_hero_dict[1] is not None:
                break
        except KeyError:
            continue

    for i in range(len(transposed)):
        counter = 1
        for k in transposed[i]:
            if 2 == k:
                new_hero_dict[2] = hero_dict.get(counter)
                for n in transposed:
                    n[counter - 1] = 0
                break
            counter += 1
        try:
            if new_hero_dict[2] is not None:
                break
        except KeyError:
            continue

    for i in range(len(transposed)):
        counter = 1
        for k in transposed[i]:
            if 3 == k:
                new_hero_dict[3] = hero_dict.get(counter)
                for n in transposed:
                    n[counter - 1] = 0
                break
            counter += 1
        try:
            if new_hero_dict[3] is not None:
                break
        except KeyError:
            continue

    for i in range(len(transposed)):
        counter = 1
        for k in transposed[i]:
            if 4 == k:
                new_hero_dict[4] = hero_dict.get(counter)
                for n in transposed:
                    n[counter - 1] = 0
                break
            counter += 1
        try:
            if new_hero_dict[4] is not None:
                break
        except KeyError:
            continue
    for i in range(len(transposed)):
        counter = 1
        for k in transposed[i]:
            if 5 == k:
                new_hero_dict[5] = hero_dict.get(counter)
                for n in transposed:
                    n[counter - 1] = 0
                break
            counter += 1
        try:
            if new_hero_dict[5] is not None:
                break
        except KeyError:
            continue

    return new_hero_dict



def get_feature_vec(winrates: dict, pick_1: list, pick_2: list) -> list:
    """
    Compute the complete feature vector for two Dota 2 picks based on their win rates and synergies.

    Parameters:
        winrates (dict): A dictionary of win rates of every hero against and with each other hero.
        pick_1 (list): The list of the heroes.
        pick_2 (list): The list of the second heroes.

    Returns:
        list: Features for the two picks.(len:80)

    The feature vector is computed by concatenating the following four types of features:
        - Synergy features for pick_1, pick_2 (computed using the `get_synergy_features` function).
        - Duel features for heroes in pick_1 and pick_2 (computed using the `get_duel_features` function).
    """
    pick_1_synergy_features = get_synergy_features(winrates, pick_1)
    pick_2_synergy_features = get_synergy_features(winrates, pick_2)
    pick_1_duel_features, pick_2_duel_features = get_duel_features(
        winrates, pick_1, pick_2
    )

    return (
        pick_1_synergy_features
        + pick_1_duel_features
        + pick_2_synergy_features
        + pick_2_duel_features
    )


def get_synergy_features(winrates: dict, pick: list) -> list:
    """
    Compute the synergy features for a pick based on their win rates with each others.

    Parameters:
        winrates (dict): A dictionary of win rates for heroes with each other.
        pick (list): A list of heroes.

    Returns:
       list: Synergy features for the given heroes.(len:15)

    The synergy features are computed by iterating over each pair of heroes in the list and
    appending their "with" win rate to the feature vector.

    """
    pick_copy = pick[::]
    synergy_features = []
    for h1 in pick:
        for h2 in pick_copy:
            synergy_features.append(winrates[h1][h2]["with_winrate"])
        del pick_copy[0]
    return synergy_features


def get_duel_features(winrates: dict, pick_1: list, pick_2: list) -> tuple:
    """
    Compute the duel features for two lists of Dota 2 heroes based on their win rates.

    Parameters:
        winrates (dict): A dictionary of win rates for pairs of heroes.
        pick_1 (list): A list of hero names for the first pick.
        pick_2 (list): A list of hero names for the second pick.

    Returns:
        tuple: Two lists of duel features, one for each team.(len:50)

    The duel features are computed by iterating over all pairs of heroes from the two picks and
    appending their "against" win rate to the feature vector.

    """
    duel_features1, duel_features2 = [], []
    for h1 in pick_1:
        for h2 in pick_2:
            against_winrate = winrates[h1][h2].get("against_winrate", 0)
            duel_features1.append(against_winrate)
            duel_features2.append(1 - against_winrate)
    return duel_features1, duel_features2



predictor = TabularPredictor.load('AutogluonModels/production_v2')

winrates = pd.read_json('updated_winrates.json')

model_to_use = ['KNeighborsUnif_BAG_L1', 'RandomForest_r16_BAG_L1', 'LightGBMLarge_BAG_L1', 'XGBoost_r194_BAG_L1']

# for model_name in model_to_use:
#     model_pred = round(predictor.predict_proba(X_test_df, model=model_name)[1].iloc[0], 2)
#     print("Prediction from %s model: %s" % (model_name, model_pred))


def predict_v2(dire_pick, radiant_pick):

    result_dict = {'dire': dire_pick, 'radiant': radiant_pick}

    print(dire_pick)
    dire_pick = list(reshape_pick(dire_pick).values())
    radiant_pick = list(reshape_pick(radiant_pick).values())
    print(dire_pick)
    arr = np.array(get_feature_vec(winrates, pick_1=dire_pick,
                                   pick_2=radiant_pick))

    arr = arr.reshape(1, -1)

    Features_df = pd.DataFrame(arr, columns=[f'Feature_{i + 1}' for i in range(80)])

    for model_name in model_to_use:
        model_pred = round(predictor.predict_proba(Features_df, model=model_name)[1].iloc[0], 2)
        # print("Prediction from %s model: %s" % (model_name, model_pred))
        result_dict[model_name] = model_pred
    result_dict['dire_pick'] = dire_pick
    result_dict['radiant_pick'] = radiant_pick
    return result_dict


pick_1=['Troll Warlord', 'Windranger', 'Slardar', 'Muerta', 'Batrider']
pick_2=['Juggernaut', 'Leshrac', 'Legion Commander', 'Gyrocopter', 'Crystal Maiden']


pp(predict_v2(dire_pick=pick_1, radiant_pick=pick_2))

