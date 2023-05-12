import os
import fnmatch
import pandas as pd

# Read the CSV file
df = pd.read_csv("subrat/Data/Clean_train_data_encd.csv")

images = []
for root, dirs, files in os.walk("subrat/Data/solo_train"):
    for file in files:
        images.append(os.path.join(root, file))
        print(images)
# exit()

# Get the list of image names in the folder
# images = os.listdir("subrat/Data/solo_train")

# Using str.endswith()
images = [file for file in images if file.endswith(".tif")]


images = [file for file in images if fnmatch.fnmatch(file, "*.tif")]
# Filter the rows that have the same image name as in images
filtered_df = df[df['Name'].isin([os.path.basename(file) for file in images])]
# Save the filtered_df to a new CSV file
filtered_df.to_csv("subrat/Data/filtered_320_solo_train.csv")
# print(filtered_df)
# exit()
