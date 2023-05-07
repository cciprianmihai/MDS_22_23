import requests

# SERVICE 1
data = {'vector': [5, 3, 1, 4, 2]}
response = requests.post('http://localhost:5000/sort', json=data)
if response.ok:
    result = response.json()
    sorted_vector = result.get('sorted_vector')
    print('Sorted vector:', sorted_vector)
else:
    print('Error:', response.text)

# SERVICE 2
data = {'vector': [5, 3, 1, 4, 2]}
response = requests.post('http://localhost:5001/sort', json=data)
if response.ok:
    result = response.json()
    sorted_vector = result.get('sorted_vector')
    print('Sorted vector:', sorted_vector)
else:
    print('Error:', response.text)

