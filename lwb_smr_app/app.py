from matplotlib.pyplot import get
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from lwb_smr.map_image import GetMapImage
from lwb_smr.params import predict_paths_dict
from lwb_smr.predict import PredictRoof
from PIL import Image
import os

from lwb_smr.map_image import GetMapImage
from lwb_smr.params import predict_paths_dict, prediction_path_dict
from lwb_smr.solar_my_roof import SolarMyRoof
# Code to loaded images, make prediction

### st.write(os. getcwd())

# raw_data/data_samples/train_examples

# train_test = Image.open('raw_data/data_samples/train_examples/austin1.tif')
# mask_test = Image.open('raw_data/data_samples/train_examples/austin2.tif')
# layover_test = Image.open('raw_data/data_samples/train_examples/austin3.tif')

# train_test = Image.open('/Users/jackhousego/code/hughharford/lwb_smr/raw_data/data_samples/train_examples/austin1.tif')
# mask_test = Image.open('/Users/jackhousego/code/hughharford/lwb_smr/raw_data/data_samples/gt_examples/austin1.tif')
# layover_test = Image.open('/Users/jackhousego/code/hughharford/lwb_smr/raw_data/data_samples/input_with_mask.jpg')


# ABOUT THESE PATHS:
#                  download the following file and then unzip in your lwb_smr/data folder:
#          GSUTIL command:
#                      gsutil cp gs://lwb-solar-my-roof/data/demo_files.zip demo_files.zip

# IMAGE NEED:
#           1) INRIA Austin3 our full prediction
# using Austin 3, see naming already specified here:

train_test = Image.open(f"{prediction_path_dict['all_files_here']}_sample_INRIA_austin3_RGB.jpg") # got this, just MASSIVE 75mb
mask_test = Image.open(f"{prediction_path_dict['all_files_here']}_sample_INRIA_austin3_mask.tif") # got this one
layover_test = Image.open(f"{prediction_path_dict['all_files_here']}_sample_INRIA_austin3_our_prediction.tif") # copy of mask, our prediction is AMAZING!

with st.sidebar:
    selected = option_menu(
        menu_title='Main Menu',
        options=['Solar My Roof!','Home Page', 'Preselected', 'About the project','Contact']
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

if selected == 'Solar My Roof!':
    st.markdown('''
                # Solar My Roof!
                ''')
    st.session_state.top_level = False

    post_code = st.text_input(label='Enter Postcode:', max_chars=8)

    smr = SolarMyRoof()

    if st.session_state.top_level == False:

        if "load_state" not in st.session_state:
            st.session_state.load_state = False
        # print('TYPE POST CODE IS: ', type(post_code))

        if st.button('Predict for postcode') or st.session_state.load_state:
            st.session_state.load_state = True
        # if len(post_code) > 4:
            # DONE >>> NEED CHANGE: we should show the predicted area RGB
            # DONE >>> NEED CHANGE: the gif should be for 'where the prediction will appear'
            # DONE >>> NEED CHANGE: gif to right hand side of RGB

            # end_execution = st.button('End
            st.write("**BEGINNING PREDICTION:**  Getting satellite image from Google Earth API, please wait...")
            # map = GetMapImage(post_code)
            # im_path_and_filename = map.get_map() # gets image name, and writes it to a file
            colp1, colp2 = st.columns(2)
            # st.write('Loading satellite image into neural network model..')

            colp1.subheader('Postcode RGB Image')
            colp2.subheader('Predicted Mask Image')

            with colp1:
                gif_loading_google = st.image(f"{prediction_path_dict['model_path']}google_load.gif")
                map = GetMapImage(post_code)
                im_path_and_filename = map.get_map() # gets image name, and writes it to a file
                gif_loading_google.empty()
                st.image(f"{im_path_and_filename}") # should already be at this path: prediction_path_dict['all_files_here']+
            #     map.remove_saved_file() # only once all neatly working

            with colp2:
                gif_runner = st.image(f"{prediction_path_dict['model_path']}loading-buffering.gif")

                #st.markdown(''' WORKING ON IT! ''')

                # LOCATION is:
                # --------------
                # CALL PREDICT.PY HERE
                # --------------

                # smr = SolarMyRoof()
                smr.load_and_ready(im_path_and_filename)

            # with colp2:
                smr.predict()
                y_pred_path_and_filename = smr.output_completed_mask()

                gif_runner.empty()
                # st.spinner(text="In progress...")
                st.image(y_pred_path_and_filename)
                # st.write(y_pred_path_and_filename[1])

                # st.markdown(''' DONE, but not loading prediction, yet ''')
            st.session_state.load_state = False

        st.session_state.top_level = True

    # if st.session_state.top_level == False:

    # if "load_roof" not in st.session_state:
    #     st.session_state.load_roof = False

    st.write('**List of roof areas:**')
    # tra = smr.get_total_roof_area()
    # st.write(f"Total area of all roofs: {tra} m^2")

    roof_df = smr.get_custom_roof_area()
    st.write(roof_df)

    # roof_number = st.text_input(label='Roof number:', max_chars=3)
    # roof_button = st.button('Get roof area')

    # if roof_button: # or st.session_state.load_roof:
    #     roof_area = smr.get_custom_roof_area(roof_number)
    #     st.write(f"Roof {roof_number} area = {roof_area: .2f}m^2")

        # st.session_state.load_roof = True

        # st.session_state.top_level = True
    # test_dict = {'Binary IoU loss': [0.65],
    #              'AuC': [0.76],
    #              'Accuracy':[0.89]}
    # test_df = pd.DataFrame.from_dict(test_dict)
    # st.markdown('''Metrics''')
    # st.table(test_df)




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
        ('Le Wagon London - E2 8DY') # include next line once NEED IMAGE completed
        #, 'Finnieston, Glasgow - G3 8LX')
    )

    image_dict_path = {
        'Le Wagon London - E2 8DY':f"{prediction_path_dict['all_files_here']}_sample_lewagon_london.jpg",
        # 'Finnieston, Glasgow - G3 8LX': f"{prediction_path_dict['all_files_here']}glasgow_test.png"
        # NEED IMAGE: glasgow image into the all files here path: glasgow_test.png
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
    inria_url = 'https://project.inria.fr/aerialimagelabeling/'
    st.markdown('''
    ## About this project
    Solar My Roof is a project developed at Le Wagon Data Science bootcamp.
    Solar Panel installation is a complex task yet provides a great opportunity
    to alleviate energy demands and provides a renewable energy source.

    It may be desirable to be able to assess large areas of urban environments to determine
    suitable roofs for solar panel installations. This project was developed to
    help automate the installation processes by identifying suitable roof space
    using Deep Learning and satellite imagery. This would not only be able to
    highlight specific roofs but be able to make an estimation of the roof
    surface area.

    It is with the hope that this work could have other uses for
    individuals, councils, or installers. To tackle this problem, we made use of
    the [INRIA dataset](%s) to provide a full labelled dataset to train a
    network.

    ''' % inria_url)

    st.markdown('''
    ### The Team
    The team consists of four Le Wagon Data Science students, who worked
    closely together over a two week sprint period to source data, train several
    Convolutional Neural Networks (CNN) and deploy a final model to a Python package.
    ''')
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.image(f"{prediction_path_dict['all_files_here']}lw-logo.png")
    # with col_a2:
    #     st.image(f"{prediction_path_dict['all_files_here']}coding_le_wagon.jpeg")
    # NEED IMAGE: in path: coding_le_wagon.jpeg, then uncomment 2 lines above

    st.markdown("""
    ### Technical Approach

    At the core, this problem is an Image Semantic Segmentation problem where for
    each pixel of an image we wish to predict whether it is part of a roof (1) or not (0).
    As the pixels that are located around each other tell us information regarding this
    classification it is important to use a suitable network. In addition to this, a
    full labelled dataset was required inorder to train a suitable model able
    to classify satellite imagery.

    #### Inria Aerial Image Labelling Dataset

    The dataset used for this project was the Inria Aerial Image Labelling Dataset.
    Within this dataset, a total of 810km^2 land coverage is provided (405km2 for
    training and testing). All images were provided with a pixel resolution of 0.3m.

    The images are taken from several cities across the world, including Chicago, San
    Francisco, Lienz and Vienna.This provides a dataset with a large variation in
    building styles and architecture. The goal behind the original dataset was the
    aim to create a model that can generalise its prediction across various regions.

    Within these separate areas there is also photos containing highly densely
    populated areas (e.g., San Francisco's financial district) and sparsely populated
    areas such as towns in Austria.

    A full description on the dataset and further reading can be found on their
    [website here](%s).

    """ % inria_url)

    col_a3, col_a4 = st.columns(2)
    col_a3.subheader('Inria Raw RGB Image')
    with col_a3:
        st.image(train_test)
    col_a4.subheader('Inria Training Mask')
    with col_a4:
        st.image(mask_test)


    st.markdown('''
    ## Model
    The model deployed in this Python package is a U-Net model enhanced with
    transfer learning from the VGG16 Neural Network. This pretrained network was
    trained on over one million images from the ImageNet database.
    ''')

    unet_url = 'https://arxiv.org/abs/1505.04597'
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('')
    with c2:
        # NEED IMAGE: in path: unet_model.png
        st.image(f"{prediction_path_dict['all_files_here']}unet_model.png", use_column_width=True)
        st.markdown('Figure: [U-Net Model](%s)' % unet_url)
    with c3:
        st.markdown('')

    st.markdown('''
    The team also investigated alterative models to use for transfer learning.
    Specifically large attention was placed on the ResNet50 model. Ultimately,
    due to time constraints and an initial better performance from the VGG16 it
    was decided to continue with the latter.

    It is the team’s objective to keep working on this project post bootcamp
    where more time can be spent on model selection and optimisation.

    While training the model, several iterations were ran using common techniques
    such as data augmentation (to provide a richer dataset with the hope of developing
    a model that better generalises) and several loss functions.

    To assess the performance of the model both the Intersection over Union
    (IoU) and accuracy where used, as outline by the Inria challenge.

    ''')

if selected == 'Contact':
    hugh = 'https://www.linkedin.com/in/hugh-harford/'
    josh = 'https://www.linkedin.com/in/joshua-katzenberg'
    amed = 'https://www.linkedin.com/in/ahmed-abbood'
    jack = 'https://www.linkedin.com/in/jack-h-79470222a'
    st.markdown('''# Contact Information''')
    st.markdown('''## LinkedIn''')
    st.markdown('[Hugh Harford](%s)' % hugh)
    st.markdown('[Jack Housego](%s)' % jack)
    st.markdown('[Josh Katzenberg](%s)' % josh)
    st.markdown('[Amed Abbood](%s)' % amed)
