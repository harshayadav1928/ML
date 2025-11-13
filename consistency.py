data = [
 ['Sunny','Warm','Normal','Strong','Warm','Same','Yes'],
 ['Sunny','Warm','High','Strong','Warm','Same','Yes'],
 ['Rainy','Cold','High','Strong','Warm','Change','No'],
 ['Sunny','Warm','High','Strong','Cool','Change','Yes']
]
h = ['Sunny','Warm','?','Strong','?','?']

for i,x in enumerate(data):
    print(f"Example {i+1}: {all(a=='?' or a==b for a,b in zip(h,x[:-1]))}")