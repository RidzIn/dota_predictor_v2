import json

import requests
import streamlit as st

from utils import predict_v2, calculate_prob_v2, calculate_prob_v1, get_meta_prediction

f = open("heroes_decoder.json")

heroes_id_names = json.load(f)


def read_heroes(file_name="heroes.txt"):
    """
    Take txt file of heroes and return set object
    """
    hero_set = set()
    with open(file_name, "r") as file:
        for line in file:
            hero_set.add(line.strip())
    return hero_set



def get_match_picks(match_id):
    response = requests.get(f'https://api.opendota.com/api/matches/{match_id}')

    radiant_picks = [pick["hero_id"] for pick in response.json()['picks_bans'] if pick["is_pick"] and pick["team"] == 0]
    dire_picks = [pick["hero_id"] for pick in response.json()['picks_bans'] if pick["is_pick"] and pick["team"] == 1]

    radiant_picks_decode = []
    dire_picks_decode = []
    for id in radiant_picks:
        for key, value in heroes_id_names.items():
            if int(key) == int(id):
                radiant_picks_decode.append(value)
                break

    for id in dire_picks:
        for key, value in heroes_id_names.items():
            if int(key) == int(id):
                dire_picks_decode.append(value)
                break

    return {"dire": dire_picks_decode, 'radiant': radiant_picks_decode,
            "dire_team": response.json()['dire_team']['name'], 'radiant_team': response.json()['radiant_team']['name']}


def display_results(dire_team, dire_pick, radiant_team, radiant_pick, probs_v2, pred):
    col1, col2 = st.columns(2)

    with col1:
        st.title(dire_team)
        st.write('**dire**')
        st.write(dire_pick)

    with col2:
        st.title(radiant_team)
        st.write('**radiant**')
        st.write(radiant_pick)

    st.write('----')

    col1, col2 = st.columns(2)

    with col1:
        st.header(f"Win probability ML")
        st.metric("", probs_v2['dire'])

    with col2:
        st.header(f"Win probability ML")
        st.metric("", probs_v2['radiant'])


    st.write('----')

    col1, col2 = st.columns(2)

    with col1:
        st.header(f"Winrates summary")
        st.metric("", pred['dire_simple'])

    with col2:
        st.header(f"Winrates summary")
        st.metric("", pred['radiant_simple'])

    st.write('----')

    col1, col2 = st.columns(2)

    temp = get_meta_prediction(dire_pick, radiant_pick)
    with col1:
        st.header("Meta")
        st.metric('', temp['dire'])
    with col2:
        st.header("Meta")
        st.metric("", temp['radiant'])

    st.write('----')


tab1, tab2 = st.tabs(["Link", 'Test Yourself'])

with tab1:
    match_id = st.number_input(label="Put match id")

    if st.button("Predict", key=2):
        temp_dict = get_match_picks(int(match_id))

        pred = predict_v2(temp_dict["dire"], temp_dict["radiant"])
        pred['dire_team'] = temp_dict['dire_team']
        pred['radiant_team'] = temp_dict['radiant_team']

        calibrated = calculate_prob_v1(pred)

        display_results(pred['dire_team'], pred['dire_pick'], pred['radiant_team'], pred['radiant_pick'], calibrated, pred)

with tab2:
    heroes = read_heroes()
    """
    ## \tSELECT HEROES FOR DIRE TEAM
    """

    dire_1, dire_2, dire_3, dire_4, dire_5 = st.columns(5)

    with dire_1:
        d1 = st.selectbox("Dire Position 1", heroes, index=None)

    with dire_2:
        d2 = st.selectbox("Dire Position 2", heroes, index=None)

    with dire_3:
        d3 = st.selectbox("Dire Position 3", heroes, index=None)

    with dire_4:
        d4 = st.selectbox("Dire Position 4", heroes, index=None)

    with dire_5:
        d5 = st.selectbox("Dire Position 5", heroes, index=None)

    """
    ## \tSELECT HEROES FOR RADIANT TEAM 
    """

    radiant_1, radiant_2, radiant_3, radiant_4, radiant_5 = st.columns(5)

    with radiant_1:
        r1 = st.selectbox("Radiant Position 1", heroes, index=None)

    with radiant_2:
        r2 = st.selectbox("Radiant Position 2", heroes, index=None)

    with radiant_3:
        r3 = st.selectbox("Radiant Position 3", heroes, index=None)

    with radiant_4:
        r4 = st.selectbox("Radiant Position 4", heroes, index=None)

    with radiant_5:
        r5 = st.selectbox("Radiant Position 5", heroes, index=None)

    if st.button("Predict", key=1):
        dire_pick = [d1, d2, d3, d4, d5]
        radiant_pick = [r1, r2, r3, r4, r5]

        pred = predict_v2(dire_pick, radiant_pick)

        probs_v2 = calculate_prob_v2(pred)

        display_results('Dire', dire_pick, 'Radiant', radiant_pick, probs_v2, pred)
