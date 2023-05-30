#!/bin/bash

chmod +x "$0"

ROOT_DIR="/home/alysson/projects/Hotel-Booking-Cancelations"

# Mover o arquivo zip para o diretório data_raw
mv "$ROOT_DIR/hotel_bookings.csv.zip" "$ROOT_DIR/data/data_raw"

# Descompactar o arquivo zip
unzip "$ROOT_DIR/data/data_raw/hotel_bookings.csv.zip" -d "$ROOT_DIR/data/data_raw"

# Excluir o arquivo zip
rm "$ROOT_DIR/data/data_raw/hotel_bookings.csv.zip"

# Renomear o arquivo csv
mv "$ROOT_DIR/data/data_raw/hotel_bookings.csv" "$ROOT_DIR/data/data_raw/data_raw.csv"