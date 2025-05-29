import pandas as pd
import os
dir="/Users/simaypay/Downloads/Results"


df = pd.read_csv('/Users/simaypay/Downloads/Results/Dataset_BaddhaKonasana_Angles.csv')
df['Label'] = 'Butterfly'
os.remove("/Users/simaypay/Downloads/Results/Dataset_BaddhaKonasana_Angles.csv")
df.to_csv('/Users/simaypay/Downloads/Results/Butterfly.csv', index=False)

df = pd.read_csv('/Users/simaypay/Downloads/Results/Dataset_Downward_Dog_Angles.csv')
df['Label'] = 'Downward Dog'
os.remove("/Users/simaypay/Downloads/Results/Dataset_Downward_Dog_Angles.csv")
df.to_csv('/Users/simaypay/Downloads/Results/Downward_Dog.csv', index=False)

df = pd.read_csv('/Users/simaypay/Downloads/Results/Dataset_Natarajasana_Angles.csv')
df['Label'] = 'Dancer'
os.remove("/Users/simaypay/Downloads/Results/Dataset_Natarajasana_Angles.csv")
df.to_csv('/Users/simaypay/Downloads/Results/Dancer.csv', index=False)

df = pd.read_csv('/Users/simaypay/Downloads/Results/Dataset_Triangle_Angles.csv')
df['Label'] = 'Triangle'
os.remove("/Users/simaypay/Downloads/Results/Dataset_Triangle_Angles.csv")
df.to_csv('/Users/simaypay/Downloads/Results/Triangle.csv', index=False)

df = pd.read_csv('/Users/simaypay/Downloads/Results/Dataset_UtkataKonasana_Angles.csv')
df['Label'] = 'Goddess'
os.remove("/Users/simaypay/Downloads/Results/Dataset_UtkataKonasana_Angles.csv")
df.to_csv('/Users/simaypay/Downloads/Results/Goddess.csv', index=False)

df = pd.read_csv('/Users/simaypay/Downloads/Results/Dataset_Veerabhadrasana_Angles.csv')
df['Label'] = 'Warrior'
os.remove("/Users/simaypay/Downloads/Results/Dataset_Veerabhadrasana_Angles.csv")
df.to_csv('/Users/simaypay/Downloads/Results/Warrior.csv', index=False)



allcsv=[pd.read_csv(os.path.join(dir, f)) for f in os.listdir(dir)]

merged_df = pd.concat(allcsv, ignore_index=True)

merged_df.to_csv("final_dataset.csv")


table = pd.read_csv("/Users/simaypay/Downloads/Yoga_Classifier-main/final_dataset.csv")


table.drop(table.columns[:2], axis=1, inplace=True)

# Save changes back to CSV (overwrite original file)
table.to_csv("/Users/simaypay/Downloads/Yoga_Classifier-main/final_dataset.csv", index=False)
