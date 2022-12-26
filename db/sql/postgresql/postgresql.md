# PostgreSQL

## Quick Start

Client Install:
```bash
# install dependencies
apt-get install postgresql libpq-dev postgresql-server-dev-all

# install libpqxx
git clone https://github.com/jtv/libpqxx.git
cd libpqxx
git checkout 7.6
mkdir build && cd build && cmake ..
make -j8
sudo make install
```

```bash
# This command attempts to find the library, REQUIRED argument is optional
find_package(PostgreSQL REQUIRED)

# Add include directories to your target. PRIVATE is useful with multi-target projects - see documentation of target_include_directories for more info
target_include_directories(MyTarget PRIVATE ${PostgreSQL_INCLUDE_DIRS})

# Add libraries to link your target againts. Again, PRIVATE is important for multi-target projects
target_link_libraries(MyTarget PRIVATE ${PostgreSQL_LIBRARIES})
```

### Postgres Server Run as Docker

* Prepare Docker

```bash
# docker engine
# Update the apt package index and install packages to allow apt to use a repository over HTTPS:
sudo apt-get update
sudo apt-get install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

#Add Docker’s official GPG key:
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Use the following command to set up the repository:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Update the apt package index:
sudo apt-get update

# Install Docker Engine, containerd, and Docker Compose.
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo apt-get install docker docker-compose

# docker desktop installation:
# first purge then installation
rm -r $HOME/.docker/desktop
sudo rm /usr/local/bin/com.docker.cli
sudo apt purge docker-desktop
sudo apt-get update
# docker deb download from https://docs.docker.com/desktop/install/ubuntu/
sudo apt-get install ./docker-desktop-<version>-<arch>.deb 

# Add the current user so that docker run does not need `sudo`
sudo usermod -a -G docker $(whoami) 
```

```yml
# docker-compose.yml
version: '3.3'

services:

    db:
        image: postgres:9.6
        environment:
            POSTGRES_PASSWORD: postgres
            PGDATA: /opt/pgsql/data
        ports:
            - 54320:5432
        volumes:
            - /datadrive/pgdata:/opt/pgsql/data
        privileged: true
```