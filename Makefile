# ----------------------------------
#          INSTALL & TEST
# ----------------------------------


PROJECT_ID=lwb-solar-my-roof
# le-wagon-bootcamp-347615
## cannot set PROJECT (with set_project) to:
#
IMAGE=tbc___XXX
REGION=europe-west1
MULTI_REGION=eu.gcr.io
BUCKET_NAME=lwb-solar-my-roof

# BUCKET FOLDERS
BUCKET_FOLDER=data
BUCKET_TRAIN_DATA_FOLDER=train
BUCKET_TEST_DATA_FOLDER=test
BUCKET_GROUNDTRUTH_DATA_FOLDER=gt




# BUILD AND RUN LOCALLY
build_image:
	docker build -t ${MULTI_REGION}/${PROJECT}/${IMAGE} .

run_image:
	docker run -e PORT=8000 -p 8080:8000 ${MULTI_REGION}/${PROJECT}/${IMAGE}

# API COMMANDS
run_api:
	uvicorn api.fast:app --reload

# GCP PROJECT SETUP COMMANDS

see_gcloud_config:
	@gcloud config list

set_project:
	@gcloud config set project ${PROJECT_ID}

# create_bucket:
# 		@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}

## file paths
SOME_FILE = /media/hsth/hh_Extr_SSD/AllData/100G/OTHER_LARGE/lwb_SMR_data/NEW2-AerialImageDataset.zip
BUCKET_TRAIN_FILE_NAME=$(shell basename ${AERIAL_SH_FILE})

# sample file paths
SAMPLE_FOLDER_FILE = /home/hsth/code/hughharford/lwb_solarmyroof/lwb_solarmyroof/data_samples/data_samples.zip
SAMPLE_FOLDER_FILE_NAME=$(shell basename ${SAMPLE_FOLDER_FILE})

# train data
# LOCAL_TRAIN_DATA_PATH="/home/hsth/code/hughharford/TaxiFareModel/raw_data/test.csv"
# BUCKET_TRAIN_FILE_NAME=$(shell basename ${LOCAL_TEST_DATA_PATH})

# test data
#LOCAL_TEST_DATA_PATH="/home/hsth/code/hughharford/TaxiFareModel/raw_data/test.csv"
#BUCKET_TEST_FILE_NAME=$(shell basename ${LOCAL_TEST_DATA_PATH})

# DATA UPLOAD

upload_sample_data:
		@gsutil cp ${SAMPLE_FOLDER_FILE} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_TRAIN_DATA_FOLDER}/${SAMPLE_FOLDER_FILE_NAME}


upload_train_data:
		@gsutil cp ${AERIAL_SH_FILE} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_TRAIN_DATA_FOLDER}/${BUCKET_TRAIN_FILE_NAME}

upload_gt_data:
		@gsutil cp ${LOCAL_GT_DATA_PATH} gs://${BUCKET_NAME}/${BUCKET_GROUNDTRUTH_DATA_FOLDER}/${BUCKET_TRAIN_DATA_FOLDER}/${BUCKET_GT_FILE_NAME}

upload_test_data:
		@gsutil cp ${LOCAL_TEST_DATA_PATH} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_TEST_DATA_FOLDER}/${BUCKET_TEST_FILE_NAME}


# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* lwb_solarmyroof/*.py

black:
	@black scripts/* lwb_solarmyroof/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr lwb_solarmyroof-*.dist-info
	@rm -fr lwb_solarmyroof.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)
