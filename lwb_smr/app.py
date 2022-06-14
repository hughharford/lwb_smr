import streamlit as st
from PIL import Image
# Code to loaded images, make prediction

# raw_data/data_samples/train_examples
train_test = Image.open('raw_data/data_samples/train_examples/austin1.tif')
mask_test = Image.open('raw_data/data_samples/train_examples/austin2.tif')
layover_test = Image.open('raw_data/data_samples/train_examples/austin3.tif')

'''
# lwb solar my roof
'''
st.markdown('''

## User Guide
- Enter post code below
-
''')
## Enter post box BOX HERE
post_code = st.text_input(label='Enter postcode', max_chars=8)
# ----------------------------------------------
# Logic to deal with post code and display image:
# ----------------------------------------------



col1, col2 = st.columns(2)
col1.subheader('Raw RGB Image')
with col1:
    st.image(train_test)
    st.markdown('''
                ## Description
                A very fancy and well thoughout description of the
                images will be placed here. It will be written beeautifully
                and fully explain what the images are showing, it will be great.
                ''')
# st.expander('Expander')
# with st.expander('Expand'):
#     st.write('Juicy deets')
col2.subheader('Roof Prediction')
with col2:
    st.image(mask_test)
    st.markdown('''## Prediction laid over raw image''')
    st.image(layover_test)

# ----------------------------------------------
# Logic to deal with post code and display image:
# ----------------------------------------------





st.markdown('''

raw-image -------- roof predicted
some description -------- overlay
## About this project

## technical stuff ...
- model , method ,

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

An example of pre-selected locations can be found at in the following section.
## INRIA



## Dropdown selection of preloaded images to predict



''')
