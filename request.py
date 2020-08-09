import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Alpha':-2.569518, 'Beta':162.1685, 'Gamma':-11.618951})

print(r.json())