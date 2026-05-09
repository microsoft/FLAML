#!/bin/bash

# Base URL for the downloads
base_url="https://synapsemldatascience.z13.web.core.windows.net/releases/flaml/conda-build/"

# Function to download files using wget
download_file() {
    local version=$1
    local url="${base_url}${version}"
    local download_dir="noarch"

    if [[ $version == "linux-64"* ]]; then
        download_dir="linux-64"
    fi

    echo "Downloading $url to $download_dir directory..."
    wget -P "$download_dir" "$url"
}

# Read each line from the file and download files
while IFS= read -r version; do echo "Downloading version $version ..."; download_file "$version"; echo "Download of version $version complete."; done < "versions_in_blob.txt"
