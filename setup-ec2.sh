#!/bin/bash

sudo yum update -y

# Install Git
sudo yum -y install git

# Install Docker
sudo yum -y install docker
sudo systemctl start docker
sudo usermod -aG docker ec2-user

# Install Docker Compose
DOCKER_CONFIG=${DOCKER_CONFIG:-$HOME/.docker}
mkdir -p $DOCKER_CONFIG/cli-plugins
curl -SL https://github.com/docker/compose/releases/download/v2.27.0/docker-compose-linux-x86_64 -o $DOCKER_CONFIG/cli-plugins/docker-compose
chmod +x $DOCKER_CONFIG/cli-plugins/docker-compose

# Clone the repository
git clone https://github.com/itscreek/qual-backend.git
cd qual-backend

# Start the application
if [ ! -e .env]; then
    echo "Please configure env files. For more information, see Deployment section of README.md."
    exit 1
fi

if [ ! -e db.sqlite3]; then
    echo "Please configure database."
    exit 1
fi

docker compose -f docker-compose.prod.yml up --build -d