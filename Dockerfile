FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY agents/ ./agents/
COPY config/ ./config/
COPY run_fleet.py ./

EXPOSE 5556

CMD ["python", "-u", "run_fleet.py"]