# Qual Backend

## Development
Set up the server using a docker container.
```
docker compose up
```
Then, you can access the server by [localhost:18000](http://localhost:18000/).

To terminate the container, execute a command below.
```
docker compose down
```

### Visual Studio Code
If you use Visual Studio Code, the Dev Containers is very useful for developing. Try Reopen in Container command:
![](https://code.visualstudio.com/assets/docs/devcontainers/create-dev-container/dev-containers-reopen.png)
The configuration of the dev container is located in the `.devcontainer` directory.

### Developing without containers
You can also run the project in your local environment(without containers).
```
pip install -r requirements.txt
python manage.py runserver
```
However, we recommend to use docker compose.

## Deployment
Todo

## References
- [Django Docs(jp)](https://docs.djangoproject.com/ja/5.0/)
- [Docker](https://www.docker.com/ja-jp/)
- [Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers)
