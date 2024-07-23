!/bin/bash

# Directory containing the sets
test_dir="test"

# Python script to process the files
python_script1="preprocess.py"
python_script2="postprocess.py"
python_script3="slic_lungwise.py"

set=$1
PREPROCESS_OUTPUT_PATH=$2
POSTPROCESS_OUTPUT_PATH=$3
SLIC_OUTPUT_PATH=$4
  
if [[ -d "$set" ]]; then
    count=$(expr $count + 1)

    echo "Processing set: $set"
    
    for file in "$set"/*; do
        if [[ -f "$file" ]]; then
            scan_name=$(basename "$file" .npz)
            echo "Processing file: $file with $python_script1"
            python3 $python_script1 "$file" "$PREPROCESS_OUTPUT_PATH"

            if [[ -f "lungmask/$set/$scan_name.nii" ]]; then
                echo "Processing file: $file with $python_script2"
                python3 $python_script2 "lungmask/$set/$scan_name.nii" "$POSTPROCESS_OUTPUT_PATH"
            else
                echo "lungmask/$set/$scan_name.nii Not Found" 
            fi
            
            
            if [[ -f "$POSTPROCESS_OUTPUT_PATH" ]]; then
                echo "Found Post Process: $POSTPROCESS_OUTPUT_PATH"
                echo "Processing file: $file with $python_script3"
                python3 $python_script3 "$PREPROCESS_OUTPUT_PATH" "$POSTPROCESS_OUTPUT_PATH" "$SLIC_OUTPUT_PATH/$scan_name"
            else
                echo "$POSTPROCESS_OUTPUT_PATH Not Found"
            fi
        else
            echo "$file is not a regular file, skipping."
        fi
    done
else
    echo "$set is not a directory, skipping."
fi

echo "Processing completed."
