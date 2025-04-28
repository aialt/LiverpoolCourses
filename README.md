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
pip install -r requirements.txt
```

### 3. Install Docker

> Install docker according to this doc [Docker](https://docs.docker.com/get-started/get-docker/)

### 4. Run PgVector

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

### 5. Set OpenAI Key

```shell
export OPENAI_API_KEY="****"
```

### 6.Run

Run with local streamlit ui app
```shell
streamlit run ./run_app_streamlit_ui.py
```

