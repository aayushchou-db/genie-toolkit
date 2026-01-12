## Installation

After cloning the repo run 
- ```uv sync```
- ```source .venv/bin/activate```

TODO: Add better installation and setup instructions

## Usage

- 'genie init' starts a new local genie config. On the first run, it will prompt you to provide a sql warehouse id corresponding to where you want your genie and a local databricks profile for auth. Can use the databricks-cli to create one. It will store your profile name and warehouse id in a local .env file.
- 'genie create' creates a genie space if it hasn't been created already
- 'genie pull' pulls config from the genie space
- 'genie push' pushes local config to the genie space
