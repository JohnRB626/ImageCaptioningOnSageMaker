#!/bin/bash

# Downloads COCO2017 image splits and annotations to specified s3 bucket

# Usage: get_coco.sh <bucket_name>
#    - bucket_name: S3 URI

splits=("val2017" "train2017" "test2017")

for split in "${splits[@]}"; do
    # Get data
    wget http://images.cocodataset.org/zips/$split.zip
    unzip -q $split.zip
    rm $split.zip
    aws s3 sync $split $1/$split --quiet
done

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/image_info_test2017.zip

unzip annotations_trainval2017.zip
unzip image_info_test2017.zip

rm annotations_trainval2017.zip
rm image_info_test2017.zip

aws s3 sync annotations $1/annotations