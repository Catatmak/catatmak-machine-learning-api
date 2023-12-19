# Catatmak Machine Learning API

Catatmak Machine Learning API is a backend service that loads machine learning models for categorization and insights. It enables consumption by the Catatmak API service and facilitates data presentation in mobile apps.

## Installation Guide

Follow these steps to set up and run the Catatmak Machine Learning API:

### 1. Clone the Repository

```bash
git clone https://github.com/Catatmak/catatmak-machine-learning-api
```

### 2. Install Dependencies

Navigate to the project directory and install the required dependencies using:

```bash
pip install -r requirements.txt
```

### 3. Run the API

Start the API service by running the following command:

```bash
python main.py
```

### 4. Congratulations!

The API is now successfully running and ready to be integrated with the Catatmak API service.

## API Documentation

Explore the Catatmak Machine Learning API endpoints using the [Postman Collection](https://documenter.getpostman.com/view/4289441/2s9YkgC5Db). The Postman Collection provides detailed information about available endpoints and how to interact with the API.

## Cloud Run Docker Builds

If you want to deploy the Catatmak Machine Learning API using Docker and Google Cloud Run, follow these steps:

1. Build the Docker image:

    ```bash
    docker buildx build --platform linux/amd64 -t catatmak-machine-learning:v1.2 .
    ```

2. Tag the Docker image:

    ```bash
    docker tag {image_id} gcr.io/catatmak/catatmak-machine-learning:v1.2
    ```

3. Push the Docker image to Google Container Registry:

    ```bash
    docker push gcr.io/catatmak/catatmak-machine-learning:v1.2
    ```

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute the code for your purposes.