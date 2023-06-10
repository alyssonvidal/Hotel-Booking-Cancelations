
dvc.init

This is a folder on Google Driver: 1JoXvKsC6O6fzfa7xrL114yf2u-arrVs2

dvc remote add -d myremote gdrive://1JoXvKsC6O6fzfa7xrL114yf2u-arrVs2
dvc remote add --default myremote gdrive://1JoXvKsC6O6fzfa7xrL114yf2u-arrVs2
dvc remote add myremote gdrive://1JoXvKsC6O6fzfa7xrL114yf2u-arrVs2

dvc remote modify myremote gdrive_client_id 'client-id'
dvc remote modify myremote gdrive_client_secret 'client-secret'

dvc add data reports models


git rm -r --cached 'data'
git commit -m "stop tracking data"

dvc add data/data_raw/data_raw.csv
dvc add data/data_processed/data_processed.csv
dvc add reports/plots/confusion_matrix.png
dvc add reports/metrics/metrics.json