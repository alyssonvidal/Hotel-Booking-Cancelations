
dvc.init

dvc remote add -d myremote gdrive://1JoXvKsC6O6fzfa7xrL114yf2u-YpykVY

dvc remote add --default myremote gdrive://1JoXvKsC6O6fzfa7xrL114yf2u-YpykVY

dvc remote add myremote gdrive://1JoXvKsC6O6fzfa7xrL114yf2u-YpykVY

dvc remote modify myremote gdrive_client_id 'client-id'

dvc remote modify myremote gdrive_client_secret 'client-secret'