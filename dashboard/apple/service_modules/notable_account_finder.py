

import json

with open('apple/service_modules/id_to_screenname_mapper.json','r') as f:
	mapper = json.load(f)

mapper = {int(k):v for k,v in mapper.items()}


def convert_to_notable_account_format(list_of_ids):
	set_of_ids = set(list_of_ids)
	return [mapper[x] for x in set_of_ids if x in mapper]
