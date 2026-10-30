import requests

url = "http://api.open-notify.org/iss-now.json" 

a = requests.get(url)
aa = a.json()
print(aa)