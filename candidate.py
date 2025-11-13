import pandas as pd

data = pd.DataFrame([
    ['Sunny','Warm','Normal','Strong','Warm','Same','Yes'],
    ['Sunny','Warm','High','Strong','Warm','Same','Yes'],
    ['Rainy','Cold','High','Strong','Warm','Change','No'],
    ['Sunny','Warm','High','Strong','Cool','Change','Yes']
], columns=['Sky','Temp','Humidity','Wind','Water','Forecast','EnjoySport'])

X = data.iloc[:, :-1]
y = data.iloc[:, -1]


specific = X.iloc[0].copy()
general = [['?' for _ in range(len(specific))] for _ in range(len(specific))]

for i, val in enumerate(y):
    if val == 'Yes':
        for j in range(len(specific)):
            if X.iloc[i, j] != specific.iloc[j]:
                specific.iloc[j] = '?'
    else:
        for j in range(len(specific)):
            if X.iloc[i, j] != specific.iloc[j]:
                general[j][j] = specific.iloc[j]


general = [g for g in general if g != ['?'] * len(specific)]

print("Final Specific Hypothesis:", list(specific.values))
print("Final General Hypotheses:", general)