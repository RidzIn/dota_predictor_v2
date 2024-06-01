import streamlit as st
from predict import predict_dict, get_meta_prediction, predict
from utils import get_match_picks, read_heroes


def display_results(dire_pick, radiant_pick, pred):
    col1, col2 = st.columns(2)
    if 'error' in pred['info'].keys():
        st.header('unpredicted')
        st.write('----')
    else:
        with col1:
            if pred['is_predicted']:
                if 'team' in pred['info'].keys():
                    st.header(pred['info']['team'])
                st.write(pred['info']['predicted_side'])
                st.write(pred['info']['predicted_pick'])
            else:
                st.header('unpredicted')

        with col2:
            st.write(pred['info'])
        st.write('----')

    col1, col2 = st.columns(2)

    meta = get_meta_prediction(dire_pick, radiant_pick)
    with col1:
        st.header("Dire Meta")
        st.metric('', meta['dire'])
    with col2:
        st.header("Radiant Meta")
        st.metric("", meta['radiant'])

    st.write('----')


tab1, tab2 = st.tabs(["Link", 'Test Yourself'])

with tab1:
    match_id = st.number_input(label="Put match id")

    if st.button("Predict", key=2):
        temp_dict = get_match_picks(int(match_id))

        pred = predict(temp_dict["dire"], temp_dict["radiant"])
        if pred['info']['predicted_side'] == 'dire':
            pred['info']['team'] = temp_dict['dire_team']

        if pred['info']['predicted_side'] == 'radiant':
            pred['info']['team'] = temp_dict['radiant_team']

        display_results(temp_dict["dire"], temp_dict["radiant"], pred)

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

        pred = predict(dire_pick, radiant_pick)

        display_results(dire_pick, radiant_pick, pred)
