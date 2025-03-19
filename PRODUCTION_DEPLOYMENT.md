# Production Deployment Guide

This guide describes how to deploy the Kesejahteraan application using Docker in a production environment.

## Prerequisites

- Docker and Docker Compose installed on your server
- Git installed on your server

## Deployment Steps

1. Clone the repository to your server:

   ```
   git clone <repository-url>
   cd kesejahteraan
   ```

2. Set up production environment variables:

   ```
   cp .env.prod .env
   ```

   Edit `.env` to set secure values for the production environment:

   - Set a strong `POSTGRES_PASSWORD`
   - Generate a secure random string for `SECRET_KEY`

3. Build and start the application:

   ```
   docker-compose up -d --build
   ```

4. Verify the application is running:

   ```
   docker-compose ps
   ```

   You should be able to access the application at http://your-server-ip:5000

## Maintenance

### Viewing logs

```
docker-compose logs -f
```

### Stopping the application

```
docker-compose down
```

### Updating the application

```
git pull
docker-compose down
docker-compose up -d --build
```

### Database backups

To create a backup of the PostgreSQL database:

```
docker-compose exec db pg_dump -U postgres prosperity > backup_$(date +%Y%m%d_%H%M%S).sql
```

To restore a backup:

```
cat backup_file.sql | docker-compose exec -T db psql -U postgres prosperity
```

## Security Considerations

1. In production, consider using a reverse proxy like Nginx to handle HTTPS termination.
2. Restrict access to your server's ports using a firewall.
3. Regularly update the Docker images and dependencies.
4. Never commit sensitive data like passwords or API keys to the repository.
