#!/bin/bash

# Output folder
OUTDIR="./ERA5_forcing"
mkdir -p $OUTDIR
cd $OUTDIR

# Base URL
BASEURL="https://ns5001k.web.sigma2.no/ROBINSON_DIRECTORIES/VAHIDREZA/FNO/ECMWF"

# Year to download
YEAR=1983

# Loop over all months
for YEAR in {1983..2024}; do
    for MONTH in {01..12}; do

        FILENAME="fno_ERA5forcing_y${YEAR}m${MONTH}.nc"
        URL="${BASEURL}/${FILENAME}"

        echo "Downloading ${FILENAME} ..."

        # wget with resume and explicit output filename
        wget -c -O "${FILENAME}" "${URL}" --no-check-certificate

        # Check if download was successful
        if [ $? -ne 0 ]; then
            echo "Warning: Failed to download ${FILENAME}"
        else
            echo "Downloaded ${FILENAME} successfully."
        fi

    done
done