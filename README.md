# Data analysis
- Document here the project: lwb_solarmyroof
- Description:
- - This project was our keystone project from #900 batch Le Wagon data-science bootcamp, June-Apr 2022
  - A team of 4 worked on this for 10 days, fairly intensively, and a 10 minute presentation was made
  - The presentation was lead by Hugh Harford and included a live demo.
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

Please document the project the better you can.

# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for lwb_solarmyroof in gitlab.com/{group}.
If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/lwb_solarmyroof`
- Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "lwb_solarmyroof"
git remote add origin git@github.com:{group}/lwb_solarmyroof.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
lwb_solarmyroof-run
```

# Install

Go to `https://github.com/{group}/lwb_solarmyroof` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/lwb_solarmyroof.git
cd lwb_solarmyroof
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
lwb_solarmyroof-run
```
