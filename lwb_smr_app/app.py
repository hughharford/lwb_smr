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

train_test = Image.open(f"{prediction_path_dict['all_files_here']}austin1.tif")
mask_test = Image.open(f"{prediction_path_dict['all_files_here']}austin2.tif")
layover_test = Image.open(f"{prediction_path_dict['all_files_here']}austin3.tif")

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

        st.image('/Users/jackhousego/code/hughharford/lwb_smr/raw_data/data_samples/contour_mask_03_numbered (1).png')
        st.image('/Users/jackhousego/code/hughharford/lwb_smr/raw_data/data_samples/contour_mask_03_numbered (2).png')

    col2.subheader('Roof Prediction')
    with col2:
        st.image(mask_test)
        st.markdown('''## Prediction laid over raw image''')
        st.image(layover_test)
        st.image('/Users/jackhousego/code/hughharford/lwb_smr/raw_data/data_samples/contour_mask_03_numbered.png')

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
        im_path_and_filename = map.get_map() # gets image name, and writes it to a file
        # LOCATION is:
        # --------------
        # CALL PREDICT.PY HERE
        # --------------
        smr = SolarMyRoof()
        smr.load_and_ready(im_path_and_filename)
        smr.predict()

        colp1, colp2 = st.columns(2)
        gif_runner.empty()
        colp1.subheader('Postcode RGB Image')

        # DON'T USE remove_saved_file() for now, just to ensure it is saving!
        # with colp1:
        #     st.image(f"{predict_paths_dict['input_image']+im_path_and_filename}")
        #     map.remove_saved_file()

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
        st.image('/Users/jackhousego/code/hughharford/lwb_smr/raw_data/data_samples/le_wagon_logo.png')
    with col_a2:
        st.image('/Users/jackhousego/code/hughharford/lwb_smr/raw_data/data_samples/coding_le_wagon.jpeg')

    st.markdown('''
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
    populated areas (e.g., San Francisco’s financial district) and sparsely populated
    areas such as towns in Austria.

    A full description on the dataset and further reading can be found on their
    [website here](%s).

    ''' % inria_url)
    col_a3, col_a4 = st.columns(2)
    col_a3.subheader('Inria Raw RGB Image')
    with col_a3:
        st.image(train_test)
    col_a4.subheader('Inria Trianing Mask')
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
        st.image('/Users/jackhousego/code/hughharford/lwb_smr/raw_data/data_samples/u_net_model.png', use_column_width=True)
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
