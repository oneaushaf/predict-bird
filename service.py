import requests, json

def make_request(endpoint : str, body: dict):
    try:
        response = requests.post(endpoint, json=body)
        if response.status_code == 200:
            return 
        else:
            with open('./../models/temp/report.json', 'w') as json_file:
                json.dump(body, json_file)
    except Exception as e:
        with open('./../models/temp/report.json', 'w') as json_file:
                json.dump(body, json_file)