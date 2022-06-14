import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from lwb_smr.map_image import GetMapImage
from lwb_smr.params import predict_paths_dict
from lwb_smr.predict import PredictRoof
from PIL import Image
# Code to loaded images, make prediction

# raw_data/data_samples/train_examples
# train_test = Image.open('raw_data/data_samples/train_examples/austin1.tif')
# mask_test = Image.open('raw_data/data_samples/train_examples/austin2.tif')
# layover_test = Image.open('raw_data/data_samples/train_examples/austin3.tif')

train_test = Image.open('/Users/jackhousego/code/hughharford/lwb_smr/raw_data/data_samples/train_examples/austin1.tif')
mask_test = Image.open('/Users/jackhousego/code/hughharford/lwb_smr/raw_data/data_samples/gt_examples/austin1.tif')
layover_test = Image.open('/Users/jackhousego/code/hughharford/lwb_smr/raw_data/data_samples/input_with_mask.jpg')

with st.sidebar:
    selected = option_menu(
        menu_title='Main Menu',
        options=['Home Page','Post Code', 'Preselected', 'About the project','Contact']
    )


if selected == 'Home Page':
    st.title('Solar My Roof')

    st.markdown('''
    ## User Guide
    - Select page from menu on left hand side
    - Post code: enter a postcode you want to map roofs
    - Preselected: select a location from list
    - About the project: background to project, method to implement and information of Convolutional Neural Network
    - Contact: contact information of team
    ''')
    col1, col2 = st.columns(2)
    col1.subheader('Raw RGB Image')
    with col1:
        st.image(train_test)
        st.markdown('''
                    ## Description
                    Given a aerial image of an urban environment, the model will predict
                    the locations of building roofs and create a predicted mask image.
                    This mask image will then be overlaid onto the original photo to
                    highlight the effectiveness of the prediction.
                    ''')

    col2.subheader('Roof Prediction')
    with col2:
        st.image(mask_test)
        st.markdown('''## Prediction laid over raw image''')
        st.image(layover_test)

# ----------------------------------------------
# Logic to deal with post code and display image:
# ----------------------------------------------

if selected == 'Post Code':
    st.markdown('''
                # Enter postcode
                ''')
    post_code = st.text_input(label='Postcode:', max_chars=8)

    # print('TYPE POST CODE IS: ', type(post_code))

    if st.button('Predict for postcode'):
        gif_runner = st.image('/Users/jackhousego/code/hughharford/lwb_smr/raw_data/data_samples/ezgif.com-gif-maker.gif')
        # end_execution = st.button('End
        map = GetMapImage(post_code)
        im_name = map.get_map() # gets image name

        # --------------
        # CALL PREDICT.PY HERE
        # --------------
        # im_predict_path = PredictRoof().new_function()

        colp1, colp2 = st.columns(2)
        gif_runner.empty()
        colp1.subheader('Postcode RGB Image')
        with colp1:
            st.image(f"{predict_paths_dict['input_image']+im_name}")
            map.remove_saved_file()

        colp2.subheader('Predcited Mask Image')
        with colp2:
            # st.image(im_predict_path)
            st.markdown(''' Feature to be added ''')

    test_dict = {'Binary IoU loss': [0.65],
                 'AuC': [0.76],
                 'Accuracy':[0.89]}
    test_df = pd.DataFrame.from_dict(test_dict)
    st.markdown('''Metrics''')
    st.table(test_df)




# ----------------------------------------------
# Preloaded images:
# ----------------------------------------------
if selected == 'Preselected':
    st.markdown('''
                # Preselected location
                ''')
    st.markdown('''
                # WORK IN PROGRESS
                ''')
    option = st.selectbox(
        'Please select a preloaded example location',
        ('Le Wagon London - E2 8DY', 'Finnieston, Glasgow - G3 8LX')
    )

    image_dict_path = {
        'Le Wagon London - E2 8DY':'/Users/jackhousego/code/hughharford/lwb_smr/raw_data/data_samples/lewagon_london.jpg',
        'Finnieston, Glasgow - G3 8LX': '/Users/jackhousego/code/hughharford/lwb_smr/raw_data/data_samples/glasgow_test.png'
    }
    st.write('You selected: ', option)

    if st.button('Show me the roofs'):
        col3, col4 = st.columns(2)
        col3.subheader('Raw RGB Image')
        with col3:
            option_image = Image.open(image_dict_path[option])
            st.image(option_image)
        col4.subheader('Laid over image')
        with col4:
            st.image(layover_test)



# ----------------------------------------------
# Logic to deal with post code and display image:
# ----------------------------------------------



if selected == 'About the project':

    st.markdown('''
    ### About this project
    lwb solar my roof is a project developed at Le Wagon Data Science bootcamp.
    Solar Panal installation is a complex task, yet provides a great opportunity
    to illeviate energy demands and polution. It is desirable to be able to assess
    large areas of roof tops in order to determine applicaple roofs for solar panel
    instalation.
    This project was developed to help automate the solar installation process by
    identifying suitable roof space using Deep Learning and satellite imagery.
    It is with the hope that this work could have other uses for individuals, councils
    or installers.
    This page contains the final model developed by the team. Custom locations can be
    requested and the model will predict the locations of the roofs, giving an estimation
    of the roof area.
    ### INRIA
    ### technical stuff ...
    - model , method ,
    ''')
