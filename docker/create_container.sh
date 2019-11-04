#!/usr/bin/env bash

docker run --name pytorch --restart=always -d --shm-size 8G -e LC_ALL=C.UTF-8 -e LANG=C.UTF-8 -p 8889:8888 -v /home/$USER:/workspace --runtime=nvidia my-jupyter
