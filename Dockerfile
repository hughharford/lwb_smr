FROM python:3.8.13-buster

# CREDENTIALS for GCP:
COPY /home/hsth/HSTH_most_TP/LeWagon_TP/CREDENTIALS/le-wagon-bootcamp-347615-e902c9dccea0__INC_RENAME_gcp_google_cloud.json /credentials.json


# copy and install requirements
COPY requirements.txt     /requirements.txt
#We will install all the packages we need
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# copy files across to the new container
COPY model.joblib         /model.joblib
# OR USE GCP... let's see

COPY api/fast.py          /api/fast.py
COPY TaxiFareModel        TaxiFareModel
COPY predict.py           /predict.py



# Launch with uvicorn our FastAPI app
CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT

# run with
# sudo docker run -e PORT=8000 -p 8000:8000 fastapi:01


# ➜  TFM_PredictInProd git:(master) ✗ export DOCKER_IMAGE_NAME=fastapi:01                                                                                                                       [🐍 lewagon]
# ➜  TFM_PredictInProd git:(master) ✗ echo $DOCKER_IMAGE_NAME                                                                                                                                   [🐍 lewagon]
# fastapi:01
# ➜  TFM_PredictInProd git:(master) ✗ export PROJECT_ID=le-wagon-bootcamp-347615                                                                                                                [🐍 lewagon]
# ➜  TFM_PredictInProd git:(master) ✗ echo $PROJECT_ID                                                                                                                                          [🐍 lewagon]
# le-wagon-bootcamp-347615
# ➜  TFM_PredictInProd git:(master) ✗ docker build -t eu.gcr.io/$PROJECT_ID/$DOCKER_IMAGE_NAME .                                                                                                [🐍 lewagon]
# Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Post "http://%2Fvar%2Frun%2Fdocker.sock/v1.24/build?buildargs=%7B%7D&cachefrom=%5B%5D&cgroupparent=&cpuperiod=0&cpuquota=0&cpusetcpus=&cpusetmems=&cpushares=0&dockerfile=Dockerfile&labels=%7B%7D&memory=0&memswap=0&networkmode=default&rm=1&shmsize=0&t=eu.gcr.io%2Fle-wagon-bootcamp-347615%2Ffastapi%3A01&target=&ulimits=null&version=1": dial unix /var/run/docker.sock: connect: permission denied
# ➜  TFM_PredictInProd git:(master) ✗ sudo docker build -t eu.gcr.io/$PROJECT_ID/$DOCKER_IMAGE_NAME .                                                                                           [🐍 lewagon]
# [sudo] password for hsth:
#  --->
#     ...
#     ...
# Successfully built 1ae3280620db
# Successfully tagged eu.gcr.io/le-wagon-bootcamp-347615/fastapi:01

# RUN updated GCP container:
# docker run -e PORT=8000 -p 8000:8000 eu.gcr.io/le-wagon-bootcamp-347615/fastapi:01

# push to GCP
# sudo docker push eu.gcr.io/le-wagon-bootcamp-347615/fastapi:01

# DEPLOY ON GCP
# gcloud run deploy --image eu.gcr.io/le-wagon-bootcamp-347615/fastapi:01 --platform managed --region europe-west1

# DEPLOY WITH CREDENTIALS:
gcloud run deploy \
    --image eu.gcr.io/le-wagon-bootcamp-347615/fastapi:01 \
    --platform managed \
    --region europe-west1 \
    --set-env-vars "GOOGLE_APPLICATION_CREDENTIALS=/credentials.json"
