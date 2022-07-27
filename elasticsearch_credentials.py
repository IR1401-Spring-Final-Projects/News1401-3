import argparse
import json

CREDENTIALS_PATH = 'UI/webui/mir/Elasticsearch_Credentials.json'


parser = argparse.ArgumentParser(description='Set cedentials for Elasticsearch')
parser.add_argument('--username', type=str, help="Username for Elasticsearch", required=True)
parser.add_argument('--password', type=str, help="Passwrod for Elasticsearch", required=True)

args = parser.parse_args()

with open(CREDENTIALS_PATH, 'w') as file:
    username_password = {
    "USERNAME": args.username,
    "PASSWORD": args.password
    }
    json.dump(username_password, file, indent=4)