import yaml
import json

with open("config_hits_track_v2_noise.yaml", "r") as f:
    yaml_data = yaml.safe_load(f)

with open("config_hits_track_v2_noise.json", "w") as f:
    json.dump(yaml_data, f, indent=2)
