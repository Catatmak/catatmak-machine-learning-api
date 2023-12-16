Catatmak Machine Learning API

docker buildx build --platform linux/amd64 -t catatmak-ml-api:v1.1 .
docker tag 95ce75ee726c gcr.io/catatmak/catatmak-ml-api:v1.1
docker push gcr.io/catatmak/catatmak-ml-api:v1.1