# Project Description: lwb_solarmyroof
- This project was our keystone project from #900 batch Le Wagon data-science bootcamp, June-Apr 2022
  - A team of 4 worked on this for 10 days, fairly intensively, and a 10 minute presentation was made
  - The project was lead by Hugh Harford (with huge help from Jack, Josh and Ahmed) and included a live demo.
  - Usage of the SolarMyRoof functionality involves these steps (steps done internally only in brackets):
  - - Load website
    - Enter your postcode
    - (Run model on high spec GCP Vertex machine, and get prediction)
    - See prediction of solar panel roofspace and calculations of number of solar panels
- Data Source:
- - Inria Aerial Imaging Dataset: https://project.inria.fr/aerialimagelabeling/
- Type of analysis:
- - Neural Network (16 million parameter UNet with VGG-16 transfer learning) run on a GCP Vertex machine
  - The model was designed, implemented, and trained by the team on the Vertex instance

# NOTE BENE
- - This will run a local website, using Streamlit
- - What will NOT work is the loading the historical trained model file (.h5)
- - This is unavoidable (for now)...


# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

# Install

Ensure your ssh public key is setup etc...

Clone and install:
```bash
git clone git@github.com:hughharford/lwb_solarmyroof.git
cd lwb_solarmyroof
pip install -r requirements.txt
make clean install test                # install and test
```

# Run the website locally, once installed:

```bash
streamlit run lwb_smr/app.py
```
