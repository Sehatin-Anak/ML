## Install library in Virtual Environment

```sh
python -m venv venv
source venv\Scripts\activate
pip install -r requirements.txt
```

## test API in localhost

```sh
 curl -X GET "http://localhost:8080/generate_json/{age_group}" -H "accept: application/json"
```

using 1 / 2 / 3 for age group
