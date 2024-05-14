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

### Developing with Python venv
You can also run the project with Python venv.
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python manage.py runserver 127.0.0.1:18000
```
Make sure your Python version is 3.10 or later.

## Deployment
```
docker compose -f docker-compose.prod.yml up
```
You need `.env` file for deployment. This file configures django. 
```
DJANGO_SECRET_KEY=secretkeyfordjango
DJANGO_DEBUG=False
DJANGO_ALLOWED_HOSTS=localhost,www.your.domain.for.app.com
```
`SECRET_KEY` is important for security and it must be private. Be careful when handling `SECRET_KEY`.

You also need to configure `nginx/nginx.conf` to access the app using the server's domain name.
```
http {
    server {
        ...

        # Add the domain name 'www.your.domain.for.app.com'
        server_name localhost www.your.domain.for.app.com 

        ...
    }
}
```

## References
- [Django Docs(jp)](https://docs.djangoproject.com/ja/5.0/)
- [Docker](https://www.docker.com/ja-jp/)
- [Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers)
