# !/bin/sh

# GLOBAL Args
# Reset
Color_Off='[0m'

# Regular Colors
Purple='[0;35m'

# Bold
BRed='[1;31m'

# Run the Commands
cd ..
echo -e "${BRed}Running cleanup...${Color_Off}"
rm -rf PIOFE-Unrolling
echo -e "${Purple}Complete!${Color_Off}"