import streamlit as st
from PIL import Image
# Code to loaded images, make prediction


train_test = Image.open('/Users/jackhousego/code/hughharford/lwb_smr/raw_data/data_samples/train_examples/austin1.tif')
mask_test = Image.open('/Users/jackhousego/code/hughharford/lwb_smr/raw_data/data_samples/gt_examples/austin1.tif')
layover_test = Image.open('/Users/jackhousego/code/hughharford/lwb_smr/raw_data/data_samples/input_with_mask.jpg')

'''
# lwb solar my roof
'''
st.markdown('''

## User Guide
- Enter post code below
- Or used selected locations in following section
- Website will source a raw RGB image and make a prediction on roof
- Prediction will be laid over the raw image as demonstrated below
''')

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
            # Enter postcode
            ''')
post_code = st.text_input(label='Postcode:', max_chars=8)
st.markdown('''
            # Preselected location
            ''')
st.markdown('''
            # WORK IN PROGRESS
            ''')


# ----------------------------------------------
# Preloaded images:
# ----------------------------------------------
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
