import json

import joblib
import numpy as np
import pandas as pd
import requests
from autogluon.tabular import TabularPredictor
from rich import print as pp


hero_prior = pd.read_excel('data/heroes_prior.xlsx')


def pos_reshape(pick):
    filtered_df = hero_prior[hero_prior['hero'].isin(pick)]

    ordered_heroes = [None] * 5

    # Dictionary to store potential heroes for each position
    position_candidates = {}

    # Collecting candidates for each position
    for pos in range(1, 6):
        pos_col = f'pos_{pos}'
        sorted_heroes = filtered_df.sort_values(by=pos_col, ascending=False)['hero'].tolist()
        position_candidates[pos] = sorted_heroes

    # Set to track chosen heroes
    chosen_heroes = set()

    # Assigning heroes to each position
    for pos in range(1, 6):
        candidates = position_candidates[pos]
        for hero in candidates:
            if hero not in chosen_heroes:
                ordered_heroes[pos - 1] = hero
                chosen_heroes.add(hero)
                break

    # Fill any remaining positions with available heroes
    all_heroes = filtered_df['hero'].tolist()
    for i in range(len(ordered_heroes)):
        if ordered_heroes[i] is None:
            for hero in all_heroes:
                if hero not in chosen_heroes:
                    ordered_heroes[i] = hero
                    chosen_heroes.add(hero)
                    break

    return ordered_heroes


def get_feature_vec(winrates, dire: list, radiant: list) -> list:
    """
    Compute the complete feature vector for two Dota 2 picks based on their win rates and synergies.

    Parameters:
        winrates (dict): A dictionary of win rates of every hero against and with each other hero.
        dire (list): The list of the heroes.
        radiant (list): The list of the second heroes.

    Returns:
        list: Features for the two picks.(len:80)

    The feature vector is computed by concatenating the following four types of features:
        - Synergy features for pick_1, pick_2 (computed using the `get_synergy_features` function).
        - Duel features for heroes in pick_1 and pick_2 (computed using the `get_duel_features` function).
    """
    dire_synergy_features = get_synergy_features(winrates, dire)
    radiant_synergy_features = get_synergy_features(winrates, radiant)
    dire_duel_features, radiant_duel_features = get_duel_features(
        winrates, dire, radiant
    )

    return (
            dire_synergy_features
            + dire_duel_features
            + radiant_synergy_features
            + radiant_duel_features
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


def get_duel_features(winrates, pick_1: list, pick_2: list) -> tuple:
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


predictor = TabularPredictor.load('AutogluonModels/production')

winrates = pd.read_json('data/winrates.json')

model_to_use = ['ExtraTrees_r49_BAG_L1', 'RandomForest_r16_BAG_L1']

rf_feedback = pd.read_json('data/rf_stat.json')

ext_feedback = pd.read_json('data/extra_tree_stat.json')

EXT_PREDICTED_MEAN = 0.54
EXT_UNPREDICTED_MEAN = 0.46


def predict_dict(dire_pick, radiant_pick):
    from feedbacks import get_feedback_prediction

    result_dict = {}

    dire_pick = pos_reshape(dire_pick)
    radiant_pick = pos_reshape(radiant_pick)

    rf_f = get_feedback_prediction(winrates, rf_feedback, dire_pick, radiant_pick, 'RandomForest_r16_BAG_L1')
    ext_f = get_feedback_prediction(winrates, ext_feedback, dire_pick, radiant_pick, 'ExtraTrees_r49_BAG_L1')

    if rf_f['predicted_side'] == 'dire' and ext_f['predicted_side'] == 'dire':
        result_dict['predicted_side'] = 'dire'
        result_dict['predicted_pick'] = dire_pick
    elif rf_f['predicted_side'] == 'radiant' and ext_f['predicted_side'] == 'radiant':
        result_dict['predicted_side'] = 'radiant'
        result_dict['predicted_pick'] = radiant_pick
    else:
        raise ValueError("Unpredicted")

    result_dict['predicted_winrate'] = (rf_f['predicted_winrate'] + ext_f['predicted_winrate']) / 2
    result_dict['unpredicted_winrate'] = (rf_f['unpredicted_winrate'] + ext_f['unpredicted_winrate']) / 2
    result_dict['model_pred'] = (rf_f['model_pred'] + ext_f['model_pred']) / 2
    print(result_dict)
    return result_dict


def predict(dire_pick, radiant_pick):
    pred_dict = predict_dict(dire_pick, radiant_pick)

    if pred_dict['predicted_winrate'] >= 0.53 and pred_dict['model_pred'] >= 0.55 and pred_dict['unpredicted_winrate'] <= 0.47:
        return {'pick': pred_dict['predicted_pick'], 'side': pred_dict['predicted_side']}
    else:
        return {'pick': 'unpredicted', 'side': 'unpredicted'}



def get_meta_prediction(pick_1, pick_2):
    """Parse data from OpenDota API and calculate win probability based on recent matches played on this
    heroes by non-professional players"""
    avg_winrates = {}
    for hero in pick_1:
        temp_df = get_hero_matchups(hero, pick_2)
        avg_winrates[hero] = temp_df["winrate"].sum() / 5
    team_1_win_prob = round(sum(avg_winrates.values()) / 5, 3)
    return {"dire": team_1_win_prob, "radiant": 1 - team_1_win_prob}


def get_hero_matchups(hero_name, pick):
    with open('data/heroes_decoder.json', 'r') as file:
        heroes_id_names = json.load(file)
    for key, value in heroes_id_names.items():
        if value == hero_name:
            hero_key = key
            break

    response = requests.get(f"https://api.opendota.com/api/heroes/{hero_key}/matchups")

    data = json.loads(response.text)
    temp_df = pd.DataFrame(data)
    temp_df["winrate"] = round(temp_df["wins"] / temp_df["games_played"], 2)

    temp_df["name"] = [heroes_id_names[str(i)] for i in temp_df["hero_id"]]
    temp_df = temp_df[temp_df["name"].isin(pick)]
    return temp_df


def pick_parse(pick_str: str):
    """From csv file parse heroes"""
    pick_str = pick_str[1:-1]
    pick_str = pick_str.replace("'", "")
    temp: list = pick_str.split(", ")
    if 'Natures Prophet' in temp:
        print(temp)
        temp.remove('Natures Prophet')
        temp.append("Natures's Prophet")
        print(temp)
    return temp


def get_features_from_dataframe(dataframe, winrates):
    features = [get_feature_vec(winrates, dataframe.iloc[i]['dire_pick'], dataframe.iloc[i]['radiant_pick']) for i in
                range(len(dataframe))]

    df = pd.DataFrame(features, columns=[f'Feature_{i + 1}' for i in range(80)])

    df['Label'] = dataframe['Label']
    return df

