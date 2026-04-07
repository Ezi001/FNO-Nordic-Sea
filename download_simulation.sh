#!/bin/bash

# Output folder
OUTDIR="./data"
mkdir -p $OUTDIR
cd $OUTDIR

# Base URL
BASEURL="https://ns5001k.web.sigma2.no/ROBINSON_DIRECTORIES/VAHIDREZA/FNO"

# Year to download
YEAR=1983

# Loop over all months

for YEAR in {1980..1986}; do

        FILENAME="NAA10KM_1h_${YEAR}0101_${YEAR}1231_ssh.nc"
        FILENAME2="NAA10KM_1h_${YEAR}0101_${YEAR}1231_ubar.nc"
        FILENAME3="NAA10KM_1h_${YEAR}0101_${YEAR}1231_vbar.nc"

        URL="${BASEURL}/${FILENAME}"
        URL2="${BASEURL}/${FILENAME2}"
        URL3="${BASEURL}/${FILENAME3}"

        echo "Downloading ${FILENAME} ${FILENAME2} ${FILENAME3}..."

        # wget with resume and explicit output filename
        wget -c -O "${FILENAME}" "${URL}" --no-check-certificate
        wget -c -O "${FILENAME2}" "${URL2}" --no-check-certificate
        wget -c -O "${FILENAME3}" "${URL3}" --no-check-certificate

        # Check if download was successful
        if [ $? -ne 0 ]; then
            echo "Warning: Failed to download ${FILENAME}"
        else
            echo "Downloaded ${FILENAME} successfully."
        fi

    done
done