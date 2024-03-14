import json, os

def save_config_file(config, path):
	''' Save the configuration file of the model into run directory '''

	# Create directory if doesn't exist yet
	os.makedirs(path, exist_ok=True)

	# Save config as json
	with open(path + '/config.json', 'w+') as f:
	    json.dump(config, f, indent=4)