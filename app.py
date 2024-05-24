import json

import requests
import streamlit as st

from utils import predict_v2

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


tab1, tab2 = st.tabs(["Link", 'Test Yourself'])

with tab1:
    match_id = st.number_input(label="Put match id")

    if st.button("Predict", key=2):
        temp_dict = get_match_picks(int(match_id))

        pred = predict_v2(temp_dict["dire"], temp_dict["radiant"])
        pred['dire_team'] = temp_dict['dire_team']
        pred['radiant_team'] = temp_dict['radiant_team']

        col1, col2 = st.columns(2)

        with col1:
            st.title(pred['dire_team'])

            st.write(pred['dire_pick'])

            st.write('----')
            col1, col3 = st.columns(2)
            with col1:
                st.write('<h3>KNeighbors</h3>', unsafe_allow_html=True)
                st.metric('', round(1-pred['KNeighborsUnif_BAG_L1'], 2))
            with col3:
                st.write('<h3>RandomForest</h3>', unsafe_allow_html=True)
                st.metric('', round(1-pred['RandomForest_r16_BAG_L1'], 2))

            col1, col3 = st.columns(2)
            with col1:
                st.write('<h3>LightGBM</h3>', unsafe_allow_html=True)
                st.metric('', round(1-pred['LightGBMLarge_BAG_L1'], 2))
            with col3:
                st.write('<h3>XGBoost</h3>', unsafe_allow_html=True)
                st.metric('', round(1-pred['XGBoost_r194_BAG_L1'], 2))

        with col2:
            st.title(pred['radiant_team'])

            st.write(pred['radiant_pick'])
            st.write('----')
            col1, col3 = st.columns(2)
            with col1:
                st.write('<h3>KNeighbors</h3>', unsafe_allow_html=True)
                st.metric('', pred['KNeighborsUnif_BAG_L1'])
            with col3:
                st.write('<h3>RandomForest</h3>', unsafe_allow_html=True)
                st.metric('', pred['RandomForest_r16_BAG_L1'])

            col1, col3 = st.columns(2)
            with col1:
                st.write('<h3>LightGBM</h3>', unsafe_allow_html=True)
                st.metric('', pred['LightGBMLarge_BAG_L1'])
            with col3:
                st.write('<h3>XGBoost</h3>', unsafe_allow_html=True)
                st.metric('', pred['XGBoost_r194_BAG_L1'])

with tab2:
        heroes = read_heroes()
        """
        ## \tSELECT HEROES FOR DIRE TEAM
        """

        dire_1, dire_2, dire_3, dire_4, dire_5 = st.columns(5)

        with dire_1:
            d1 = st.selectbox("Dire Position 1", heroes)

        with dire_2:
            d2 = st.selectbox("Dire Position 2", heroes)

        with dire_3:
            d3 = st.selectbox("Dire Position 3", heroes)

        with dire_4:
            d4 = st.selectbox("Dire Position 4", heroes)

        with dire_5:
            d5 = st.selectbox("Dire Position 5", heroes)

        """
        ## \tSELECT HEROES FOR RADIANT TEAM 
        """

        radiant_1, radiant_2, radiant_3, radiant_4, radiant_5 = st.columns(5)

        with radiant_1:
            r1 = st.selectbox("Radiant Position 1", heroes)

        with radiant_2:
            r2 = st.selectbox("Radiant Position 2", heroes)

        with radiant_3:
            r3 = st.selectbox("Radiant Position 3", heroes)

        with radiant_4:
            r4 = st.selectbox("Radiant Position 4", heroes)

        with radiant_5:
            r5 = st.selectbox("Radiant Position 5", heroes)

        if st.button("Predict", key=1):
            dire_pick = [d1, d2, d3, d4, d5]
            radiant_pick = [r1, r2, r3, r4, r5]

            pred = predict_v2(dire_pick, radiant_pick)

            col1, col2 = st.columns(2)

            with col1:
                st.title('Dire')

                st.write(pred['dire_pick'])

                st.write('----')
                col1, col3 = st.columns(2)
                with col1:
                    st.write('<h3>KNeighbors</h3>', unsafe_allow_html=True)
                    st.metric('', round(1 - pred['KNeighborsUnif_BAG_L1'], 2))
                with col3:
                    st.write('<h3>RandomForest</h3>', unsafe_allow_html=True)
                    st.metric('', round(1 - pred['RandomForest_r16_BAG_L1'], 2))

                col1, col3 = st.columns(2)
                with col1:
                    st.write('<h3>LightGBM</h3>', unsafe_allow_html=True)
                    st.metric('', round(1 - pred['LightGBMLarge_BAG_L1'], 2))
                with col3:
                    st.write('<h3>XGBoost</h3>', unsafe_allow_html=True)
                    st.metric('', round(1 - pred['XGBoost_r194_BAG_L1'], 2))

            with col2:
                st.title('radiant')

                st.write(pred['radiant_pick'])
                st.write('----')
                col1, col3 = st.columns(2)
                with col1:
                    st.write('<h3>KNeighbors</h3>', unsafe_allow_html=True)
                    st.metric('', pred['KNeighborsUnif_BAG_L1'])
                with col3:
                    st.write('<h3>RandomForest</h3>', unsafe_allow_html=True)
                    st.metric('', pred['RandomForest_r16_BAG_L1'])

                col1, col3 = st.columns(2)
                with col1:
                    st.write('<h3>LightGBM</h3>', unsafe_allow_html=True)
                    st.metric('', pred['LightGBMLarge_BAG_L1'])
                with col3:
                    st.write('<h3>XGBoost</h3>', unsafe_allow_html=True)
                    st.metric('', pred['XGBoost_r194_BAG_L1'])

