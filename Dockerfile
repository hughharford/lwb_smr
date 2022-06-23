FROM python:3.8.13-buster

# CREDENTIALS for GCP:
COPY le-wagon-bootcamp-347615_gcp_google_cloud.json /credentials.json


# copy and install requirements
COPY requirements.txt     /requirements.txt
#We will install all the packages we need
RUN pip install --upgrade pip
RUN git clone https://github.com/hughharford/lwb_smr.git

# copy files across to the new container
COPY env.py     lwb_smr/env.py
COPY .streamlit/secrets.toml     lwb_smr/.streamlit/secrets.toml
COPY lwb_smr/data/demo_files        lwb_smr/lwb_smr/data/demo_files
# OR USE GCP... let's see

RUN cd lwb_smr && pip install -e .

RUN pip install matplotlib
RUN apt-get update

# to cover dependency issues
#RUN apt-get install ffmpeg libsm6 libxext6  -y

# USE THESE INSTEAD:
# RUN apt-get update && apt-get install -y opencv-python-headless
RUN pip install opencv-python-headless

# Launch our app with streamlit
# use when ready...
# CMD streamlit run lwb_smr_app/app.py --host 0.0.0.0 --port $PORT

#CMD "/bin/bash"
# CMD streamlit run lwb_smr_app/app.py --server.address 0.0.0.0 --port $PORT
## didn't work with --server.address ...

CMD "/bin/bash"
# streamlit run lwb_smr_app/app.py

# RUN THESE COMMANDS IN TERMINAL:
# export DOCKER_IMAGE_NAME=smr:latest
# echo $DOCKER_IMAGE_NAME

# export PROJECT_ID=le-wagon-bootcamp-347615
# OR
# export PROJECT_ID=lwb-solar-my-roof                                                                                                              [🐍 lewagon]
# echo $PROJECT_ID

# BUILD with name
# sudo docker build . smr:latest
# sudo docker build -t eu.gcr.io/$PROJECT_ID/$DOCKER_IMAGE_NAME .

# TO RUN AND LOOK ABOUT INSIDE:
# sudo docker run -it  eu.gcr.io/lwb-solar-my-roof/smr:latest



# from MAKEFILE:
#streamlit run lwb_smr_app/app.py --server.port 8000

# run with
# sudo docker run -e PORT=8501 -p 8501:8501 smr:latest
# sudo docker run -e PORT=8501 -p 8501:8501 smr:latest
# sudo docker run -e PORT=8501 -p 8501:8501 eu.gcr.io/lwb-solar-my-roof/smr:latest


# ➜  TFM_PredictInProd git:(master) ✗ export DOCKER_IMAGE_NAME=smr:latest                                                                                                                       [🐍 lewagon]
# ➜  TFM_PredictInProd git:(master) ✗ echo $DOCKER_IMAGE_NAME                                                                                                                                   [🐍 lewagon]
# smr:latest
# ➜  TFM_PredictInProd git:(master) ✗ export PROJECT_ID=lwb-solar-my-roof                                                                                                                [🐍 lewagon]
# ➜  TFM_PredictInProd git:(master) ✗ echo $PROJECT_ID                                                                                                                                          [🐍 lewagon]
# lwb-solar-my-roof
# ➜  TFM_PredictInProd git:(master) ✗ docker build -t eu.gcr.io/$PROJECT_ID/$DOCKER_IMAGE_NAME .                                                                                                [🐍 lewagon]
# Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Post "http://%2Fvar%2Frun%2Fdocker.sock/v1.24/build?buildargs=%7B%7D&cachefrom=%5B%5D&cgroupparent=&cpuperiod=0&cpuquota=0&cpusetcpus=&cpusetmems=&cpushares=0&dockerfile=Dockerfile&labels=%7B%7D&memory=0&memswap=0&networkmode=default&rm=1&shmsize=0&t=eu.gcr.io%2Fle-wagon-bootcamp-347615%2Ffastapi%3A01&target=&ulimits=null&version=1": dial unix /var/run/docker.sock: connect: permission denied
# ➜  TFM_PredictInProd git:(master) ✗ sudo docker build -t eu.gcr.io/$PROJECT_ID/$DOCKER_IMAGE_NAME .                                                                                           [🐍 lewagon]
# [sudo] password for hsth:
#  --->
#     ...
#     ...
# Successfully built 1ae3280620db
# Successfully tagged eu.gcr.io/lwb-solar-my-roof/smr:latest

# RUN updated GCP container:
# docker run -e PORT=8000 -p 8000:8000 eu.gcr.io/lwb-solar-my-roof/smr:latest

# push to GCP
# sudo docker push eu.gcr.io/lwb-solar-my-roof/smr:latest

# DEPLOY ON GCP
# gcloud run deploy --image eu.gcr.io/lwb-solar-my-roof/smr:latest --platform managed --region europe-west1

    # eu.gcr.io/lwb-solar-my-roof/smr:latest \

# BUILD with name
# sudo docker build . smr:latest

# # DEPLOY WITH CREDENTIALS:
# gcloud run deploy \
#     --image smr:latest \
#     --platform managed \
#     --region europe-west1 \
#     --set-env-vars "GOOGLE_APPLICATION_CREDENTIALS=/credentials.json"
