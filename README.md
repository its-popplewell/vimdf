# VIMDF Project as of Nov 12, 2026
This is a copy of the `era5engine` that is working.
I have wrapped up most of the setup and running logic into script files which I used Claude to generate.
I have verified the code and tested it.

## Running Directions
It should work by just:
- running `./setup.sh`
- updating the box and time and whatnot in `vimdf.py` 
- running `./submit <ncar-project-id>`
- running `./finish.sh` after all the shards are complete
