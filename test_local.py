import requests

url = "http://localhost:9696/classify_waste"

data = {'url': 'https://images2.imgbox.com/c5/38/QmwPh8M6_o.jpg'}

response = requests.post(url, json=data).json()
print(response)