'''
Author: Jacob Wren
Date: 10/05/2022
Description: Extract data from portable executable files within a directory.
'''


def extract_data(directory, label):
    import os
    from ember import features

    extractor = features.PEFeatureExtractor(2)
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        file_data = open(file_path, "rb").read()
        features = str(extractor.raw_features(file_data))
        # Inserting class label using a string replace
        features = features.replace("'histogram'", "'label': " + str(label) + ", 'histogram'")
        # Replacing apostrophes with speech marks to avoid formatting errors
        features = features.replace("'", '"')
        # Write data to file
        data_dir = 'D:/ember-master/ember-master/ember2018/data.jsonl'
        with open(data_dir, 'a') as outfile:
            outfile.write(str(features) + '\n')
        outfile.close()

malware_dir = 'D:/ember-master/ember-master/mal_files'
benign_dir = 'D:/ember-master/ember-master/beni_files'
extract_data(malware_dir, 1)
extract_data(benign_dir, 0)

