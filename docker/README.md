## Docker usage

### System environment shouldn't be a problem

## If you don't know docker yet.

#### install docker
`sudo ./install_docker.sh`

### install nvidia-docker
`sudo ./install_nvidia_docker.sh`

### config docker
- Create the docker group:
    `sudo groupadd docker`
- Add your user to the docker group: 
    `sudo usermod -aG docker $USER`
- Activate the changes to groups(or reboot):
    `newgrp docker`
    
## Now you have docker

### pull image

`docker pull narrowsnap/my-jupyter`

### create container

`./create_container.sh`

### use docker 

- for bash: `docker exec -it pytorch bash`
- for jupyter: `http://localhost:8889`
