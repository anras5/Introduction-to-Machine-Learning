import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 8)

general_df = pd.read_csv('./test/general.csv')
prenatal_df = pd.read_csv('./test/prenatal.csv')
sports_df = pd.read_csv('./test/sports.csv')

prenatal_df.rename({'HOSPITAL': 'hospital', 'Sex': 'gender'}, axis=1, inplace=True)
sports_df.rename({'Hospital': 'hospital', 'Male/female': 'gender'}, axis=1, inplace=True)

merged_df = pd.concat([general_df, prenatal_df, sports_df], ignore_index=True)
merged_df.drop('Unnamed: 0', axis=1, inplace=True)
merged_df.dropna(axis=0, how='all', inplace=True)

merged_df['gender'].fillna('f', inplace=True)
merged_df['gender'] = merged_df['gender'].apply(lambda x: 'f' if x == 'female' or x == 'woman' or x == 'f' else 'm')
for col in ['bmi', 'diagnosis', 'blood_test', 'ecg', 'ultrasound', 'mri', 'xray', 'children', 'months']:
    merged_df[col] = merged_df[col].fillna(0)

# 1
answers = [merged_df['hospital'].value_counts().index[0]]

# 2
general_length = len(merged_df[merged_df['hospital'] == 'general'])
general_stomach_length = len(merged_df[(merged_df['hospital'] == 'general') & (merged_df['diagnosis'] == 'stomach')])
answers.append(round(general_stomach_length / general_length, 3))

# 3
sports_length = len(merged_df[merged_df['hospital'] == 'sports'])
sports_dislocation_length = len(merged_df[(merged_df['hospital'] == 'sports') & (merged_df['diagnosis'] == 'dislocation')])
answers.append(round(sports_dislocation_length / sports_length, 3))

# 4
median_age_general = merged_df[merged_df['hospital'] == 'general']['age'].median()
median_age_sports = merged_df[merged_df['hospital'] == 'sports']['age'].median()
answers.append(median_age_general - median_age_sports)

# 5
blood_test_taken = merged_df[merged_df['blood_test'] == 't'][['hospital']]
hospital_most_tests = blood_test_taken.value_counts().index[0]
number_most_tests = blood_test_taken.value_counts().max()
answers.append(f'{hospital_most_tests[0]}, {number_most_tests} blood tests')

#
# for i, ans in enumerate(answers):
#     print(f'The answer to the {i+1}st question is {ans}')

# 1
sns.histplot(data=merged_df, x='age')
plt.show()

# 2
data_pie = merged_df['diagnosis'].value_counts()
plt.pie(data_pie, labels=data_pie.index, colors=sns.color_palette('pastel'), autopct='%.0f%%')
plt.show()

# 3
sns.violinplot(data=merged_df, x='height', hue='hospital')
plt.show()


print(f'The answer to the 1st question: 15-35')
print(f'The answer to the 2st question: pregnancy')
print(f'The answer to the 3st question: It\'s because there is a lot of kids')