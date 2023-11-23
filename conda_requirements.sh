conda install -c pytorch faiss-gpu
pip install torch==1.13.1+cu116  --extra-index-url https://download.pytorch.org/whl/cu116
pip install transformers
pip install ipykernel
pip install scikit-learn
pip install boto3
pip install pandas==1.4.3
python -m ipykernel install --user --name base --display-name "CondaBase"