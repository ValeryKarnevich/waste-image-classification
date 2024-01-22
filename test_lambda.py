import requests

url = 'https://3ybj5ymr0c.execute-api.us-east-2.amazonaws.com/test/classify'

data = {'url': 'https://images2.imgbox.com/c5/38/QmwPh8M6_o.jpg'}

result = requests.post(url, json=data).json()
print(result)
