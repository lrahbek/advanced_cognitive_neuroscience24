python -m venv env
. env/bin/activate
python -m pip install ipykernel
python -m ipykernel install --user --name=env
#source ./env/bin/activate
python -m pip install -r requirements.txt
#pip install -r requirements.txt
#deactivate
echo Done!