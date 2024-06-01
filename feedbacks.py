import json
from collections import Counter

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from tqdm import tqdm

from utils import get_feature_vec

winrates = pd.read_json('data/winrates.json')

predictor = TabularPredictor.load('AutogluonModels/production')


def get_model_raw_info(df, winrates, model, min_threshold=0.48, max_threshold=0.52):
    """
    Evaluates the performance of a model on a test dataset and returns a dictionary of hero picks and predictions.

    Args:
        df (pandas.DataFrame): A test dataset with columns 'dire_pick', 'radiant_pick', and 'Label'.
        winrates (dict): A dictionary of hero winrates.
        model: A trained model for making predictions.
        min_threshold (float): The minimum threshold for a model's win probability prediction to be considered 'sure'.
        max_threshold (float): The maximum threshold for a model's win probability prediction to be considered 'sure'.

    Returns:
        dict: A dictionary containing lists of hero picks and predictions for the test dataset.
        The dictionary has the following keys:
        - predicted_win_heroes: A list of heroes predicted to win.
        - predicted_lose_heroes: A list of heroes predicted to lose.
        - unpredicted_win_heroes: A list of heroes that were not predicted to win but did.
        - unpredicted_lose_heroes: A list of heroes that were not predicted to lose but did.
    """
    sure = 0
    unsure = 0
    prediction_list = []

    predicted_win_heroes = []
    predicted_lose_heroes = []

    unpredicted_win_heroes = []
    unpredicted_lose_heroes = []

    correct = 0
    incorrect = 0

    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        dire_pick = row["dire_pick"]
        radiant_pick = row["radiant_pick"]
        label = row["Label"]
        arr = np.array(get_feature_vec(winrates, dire=dire_pick,
                                       radiant=radiant_pick)).reshape(1, -1)

        Features_df = pd.DataFrame(arr, columns=[f'Feature_{i + 1}' for i in range(80)])
        pred = round(predictor.predict_proba(Features_df, model=model)[1].iloc[0], 2)

        if pred >= max_threshold or pred <= min_threshold:
            sure += 1

            if pred >= max_threshold and label == 1:
                correct += 1
                predicted_win_heroes.extend(radiant_pick)
                unpredicted_lose_heroes.extend(dire_pick)
                prediction_list.append(1)

            elif pred <= min_threshold and label == 0:
                correct += 1
                predicted_win_heroes.extend(dire_pick)
                unpredicted_lose_heroes.extend(radiant_pick)
                prediction_list.append(1)

            elif pred >= max_threshold and label == 0:
                incorrect += 1
                predicted_lose_heroes.extend(radiant_pick)
                unpredicted_win_heroes.extend(dire_pick)
                prediction_list.append(-1)

            elif pred <= min_threshold and label == 1:
                incorrect += 1
                predicted_lose_heroes.extend(dire_pick)
                unpredicted_win_heroes.extend(radiant_pick)
                prediction_list.append(-1)
        else:
            unsure += 1
            prediction_list.append(0)

    return {
        "predicted_win_heroes": predicted_win_heroes,
        "predicted_lose_heroes": predicted_lose_heroes,
        "unpredicted_win_heroes": unpredicted_win_heroes,
        "unpredicted_lose_heroes": unpredicted_lose_heroes,
    }


def get_model_lists(pred_dict):
    counters = {}
    for key in pred_dict:
        counters[key] = Counter(pred_dict[key])
    return (
        counters["predicted_win_heroes"],
        counters["predicted_lose_heroes"],
        counters["unpredicted_win_heroes"],
        counters["unpredicted_lose_heroes"],
    )


def get_model_stat_dict(df, winrates, model, min_threshold=0.48, max_threshold=0.52):
    model_raw_info = get_model_raw_info(
        df, winrates, model, min_threshold, max_threshold
    )

    model_counts = get_model_lists(model_raw_info)

    return model_stat_dict(*model_counts)


def model_stat_dict(
    predicted_win_heroes,
    predicted_lose_heroes,
    unpredicted_win_heroes,
    unpredicted_lose_heroes,
):
    heroes = set(predicted_win_heroes.keys()) | set(predicted_lose_heroes.keys())

    prediction_stat_dict = {}
    for hero in heroes:
        predicted_wins = predicted_win_heroes.get(hero, 0)
        predicted_loses = predicted_lose_heroes.get(hero, 0)
        unpredicted_wins = unpredicted_win_heroes.get(hero, 0)
        unpredicted_loses = unpredicted_lose_heroes.get(hero, 0)

        total_predicted = predicted_wins + predicted_loses
        total_unpredicted = unpredicted_wins + unpredicted_loses

        predicted_winrate = round(predicted_wins / (total_predicted + 0.0001), 2)
        unpredicted_winrate = round(unpredicted_wins / (total_unpredicted + 0.0001), 2)

        prediction_stat_dict[hero] = {
            "predicted_wins": predicted_wins,
            "predicted_loses": predicted_loses,
            "predicted_winrate": predicted_winrate,
            "unpredicted_wins": unpredicted_wins,
            "unpredicted_loses": unpredicted_loses,
            "unpredicted_winrate": unpredicted_winrate,
        }

    return prediction_stat_dict


def show_mean_winrates(prediction_stat_dict):
    temp = get_mean_winrates(prediction_stat_dict)
    print(f"\tMean predicted winrate: {temp[0]}")
    print(f"\tMean unpredicted winrate: {temp[1]}")


def get_mean_winrates(prediction_stat_dict):
    winrates = {"predicted_winrate": [], "unpredicted_winrate": []}
    for hero_stats in prediction_stat_dict.values():
        winrates["predicted_winrate"].append(hero_stats["predicted_winrate"])
        winrates["unpredicted_winrate"].append(hero_stats["unpredicted_winrate"])
    predicted_winrate_mean = sum(winrates["predicted_winrate"]) / len(
        winrates["predicted_winrate"]
    )
    unpredicted_winrate_mean = sum(winrates["unpredicted_winrate"]) / len(
        winrates["unpredicted_winrate"]
    )
    return round(predicted_winrate_mean, 2), round(unpredicted_winrate_mean, 2)


def save_model_stat(df, winrates, model, file_name, min_threshold=0.48, max_threshold=0.52):
    temp = get_model_stat_dict(
        df, winrates, model, min_threshold, max_threshold
    )
    show_mean_winrates(temp)
    with open(f"{file_name}.json", "w") as outfile:
        json.dump(temp, outfile)


def update_models_feedback(df, winrates, model_1='ExtraTrees_r49_BAG_L1', model_2='RandomForest_r16_BAG_L1'):
    print("EXT model:")
    save_model_stat(
        df,
        winrates,
        model_1,
        "extra_tree_stat",
        min_threshold=0.45,
        max_threshold=0.55,
    )

    print("RF model:")
    save_model_stat(
        df,
        winrates,
        model_2,
        "rf_stat",
        min_threshold=0.45,
        max_threshold=0.55,
    )


# tier_2 = pd.read_pickle('data/tier2.pkl')
#
#
# update_models_feedback(tier_2, winrates)
