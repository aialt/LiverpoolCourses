# MajorSelectRAG

## Environmental requirements

- Operation System
    - MacOS (recommend)
- Docker
- Python: 3.11.3

## Quickstart

### 1. Clone the repository

```shell
git clone git@github.com:aialt/LiverpoolCourses.git
cd LiverpoolCourses
python3 -m venv test_env
```

### 2. Install dependencies

```shell
source ./test_env/bin/activate
cd rag_app/
pip install -r requirements.txt
```

### 3. Install Docker

> Install docker according to this doc [Docker](https://docs.docker.com/get-started/get-docker/)

### 4. Run PgVector ()


### 5. Set OpenAI Key


### 6.Run without `docker-compose`
1. Set OpenAI key 
```shell
export OPENAI_API_KEY="****"
```

2. Run the `container` of PostgreSQL + PgVector
```shell
docker run -d \
  -e POSTGRES_DB=ai \
  -e POSTGRES_USER=ai \
  -e POSTGRES_PASSWORD=ai \
  -e PGDATA=/var/lib/postgresql/data/pgdata \
  -v pgvolume:/var/lib/postgresql/data \
  -p 5532:5432 \
  --name pgvector \
  agnohq/pgvector:16
```

2. Run with local streamlit ui app
```shell
streamlit run ./run_app_streamlit_ui.py
```

### 7. Run with `docker-compose`
1. Setup `.env` file
```shell
cp .env.example .env
```

2. In `.env` file, set the OpenAI key
```
OPENAI_API_KEY=your-openai-key
```

3. Run the containers with `docker-compose`
```shell
docker-compose -f docker-compose.dev.yml up --build -d
```

### Access to Application
Open the browser and go to: `http://localhost:8501`