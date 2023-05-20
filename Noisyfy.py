train_root_dir = './subrat/Data/train_image/'
train_csv_file = './subrat/Data/filtered_320_solo_train.csv'



df = pd.read_csv('./subrat/Data/filtered_320_solo_train.csv')
print(df)


#The dx library currently enables DEX media type visualization of pandas DataFrames e.g. individual calls to dx.display()
list_unique = df.label.unique().tolist()
print(list_unique)
# exit()
for element in list_unique:
    print(f'the number of {element} is {len(df.loc[df.label == int(element)])}')

# Plot histogram
df['label'].hist(bins=10)
plt.title('Histogram of org_label')
plt.xlabel('org_label')
plt.ylabel('Count')
plt.show()
# Save plot as PNG file
# plt.savefig('subrat/Final_model/org_label_histo.png')


#  symetric noise of 20% between the below classes
Normal = df.index[df.label == 0].tolist()
Invasive = df.index[df.label== 3].tolist()
print(len(Normal))
# exit()

#symetric noise of 10% between the below classes
Benign = df.index[df.label == 1].tolist()
InSitu = df.index[df.label == 2].tolist()
# print(len(df['0']))
# exit()

df_Noisy = df
print("Initialisation done for df_Noisy")

# select 20% of data from col1 column
Normal_noisy = int(len(Normal) * 0.2)
print(Normal_noisy)
# exit()
Norm_noisy_id = random.sample(Normal,Normal_noisy) # sample randomly from 20% of the list
Norm_noisy_id
# exit()

for idx in Norm_noisy_id:
    df_Noisy.at[idx, 'label'] = '3'
print("20% of the Normal label has been converted to Invasive")


Invasive_noisy = int(len(Invasive) * 0.2)
print(Invasive_noisy)
# exit()
Inv_noisy_id = random.sample(Invasive,Invasive_noisy) # sample randomly from 20% of the list
Inv_noisy_id
# exit()

for idx in Inv_noisy_id:
    df_Noisy.at[idx, 'label'] = '0'
print("20% of the Invasive label has been converted to Normal")


Benign_noisy = int(len(Benign) * 0.2)
print(Benign_noisy)
# exit()
Ben_noisy_id = random.sample(Benign,Benign_noisy) # sample randomly from 20% of the list
Ben_noisy_id
# exit()

for idx in Ben_noisy_id:
    df_Noisy.at[idx, 'label'] = '2'
print("20% of the Invasive label has been converted to Normal")



InSitu_noisy = int(len(InSitu) * 0.2)
print(InSitu_noisy)
# exit()
Ins_noisy_id = random.sample(InSitu,InSitu_noisy) # sample randomly from 20% of the list
Ins_noisy_id
# exit()

for idx in Ins_noisy_id:
    df_Noisy.at[idx, 'label'] = '1'
print("20% of the Invasive label has been converted to Normal")


df_Noisy.to_csv('subrat/Final_Supervised_Results/noisy_bach_pf.csv', index=False)

