import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from utils import get_feature_vec, get_hero_matchups, pos_reshape

predictor = TabularPredictor.load('AutogluonModels/production')
winrates = pd.read_json('data/winrates.json')
rf_feedback = pd.read_json('data/feedback/rf_stat.json')
ext_feedback = pd.read_json('data/feedback/extra_tree_stat.json')


def get_row_prediction(winrates, dire_pick, radiant_pick, model):
    """Return model's prediction without any additional checkers"""

    arr = np.array(get_feature_vec(winrates, dire=dire_pick, radiant=radiant_pick)).reshape(1, -1)
    Features_df = pd.DataFrame(arr, columns=[f'Feature_{i + 1}' for i in range(80)])
    prediction = predictor.predict_proba(Features_df, model=model)

    return {'dire': prediction[0].iloc[0], 'radiant': prediction[1].iloc[0]}


def get_feedback_prediction(winrates, feedback, dire_pick, radiant_pick, model):
    prediction = get_row_prediction(winrates, dire_pick, radiant_pick, model)

    if prediction["dire"] > prediction["radiant"]:
        predicted_pick = dire_pick
        unpredicted_pick = radiant_pick
        predicted_side = 'dire'
        unpredicted_side = 'radiant',
        model_pred = prediction['dire']
    else:
        predicted_pick = radiant_pick
        unpredicted_pick = dire_pick
        predicted_side = 'radiant'
        unpredicted_side = 'dire'
        model_pred = prediction['radiant']

    predicted_winrate = sum(feedback[hero]["predicted_winrate"] for hero in predicted_pick) / 5
    unpredicted_winrate = sum(feedback[hero]["unpredicted_winrate"] for hero in unpredicted_pick) / 5

    return {
        'predicted_winrate': round(predicted_winrate, 3),
        'unpredicted_winrate': round(unpredicted_winrate, 3),
        'predicted_side': predicted_side,
        'unpredicted_side': unpredicted_side,
        'model_pred': model_pred,
    }


def get_meta_prediction(pick_1, pick_2):
    """Parse data from OpenDota API and calculate win probability based on recent matches played on this
    heroes by non-professional players"""
    avg_winrates = {}
    for hero in pick_1:
        temp_df = get_hero_matchups(hero, pick_2)
        avg_winrates[hero] = temp_df["winrate"].sum() / 5
    team_1_win_prob = round(sum(avg_winrates.values()) / 5, 3)
    return {"dire": team_1_win_prob, "radiant": 1 - team_1_win_prob}


def predict_dict(dire_pick, radiant_pick):
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
        raise ValueError("Models selected 2 different teams, reload the page and use another match")

    result_dict['predicted_winrate'] = (rf_f['predicted_winrate'] + ext_f['predicted_winrate']) / 2
    result_dict['unpredicted_winrate'] = (rf_f['unpredicted_winrate'] + ext_f['unpredicted_winrate']) / 2
    result_dict['model_pred'] = (rf_f['model_pred'] + ext_f['model_pred']) / 2

    return result_dict


def predict(dire_pick, radiant_pick):
    pred_dict = predict_dict(dire_pick, radiant_pick)

    if pred_dict['predicted_winrate'] >= 0.53 and pred_dict['model_pred'] >= 0.55 and pred_dict['unpredicted_winrate'] <= 0.47:
        return {'is_predicted': True, 'info': pred_dict}
    else:
        return {'is_predicted': False, 'info': pred_dict}
