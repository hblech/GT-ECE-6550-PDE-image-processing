#!/bin/bash
new_dir=$1
mkdir -p "$1"/lr_images "$1"/hr_images
mv experiences.txt "$1"
mv lr_images/* "$1"/lr_images
mv hr_images/* "$1"/hr_images
