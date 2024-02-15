# Madhukant ML Project Sample

This repo contains the code for
- reading csv file
- cleaning dataframe
- do feature extraction on dataframe
- train the model
- save the model
- evaluate the model
- visualize the model

All the codes have it's corresponding test scripts as well. Code coverage can be seen on console as well as on html file.


## Run Commands:

### Setup Environment

#### Install
```bash
python -m virtualenv venv
```

#### Activate
```bash
venv/Scripts/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the code
```bash
python scripts/main.py
```

### Run the Test cases
```bash
pytest .\tests --cov=src --cov-report=term --cov-report=html
```
