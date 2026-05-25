FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps first (Docker cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn[standard] pydantic

# Copy your actual project code
COPY phase1_simulation.py .
COPY phase2_environment.py .
COPY phase3_training.py .
COPY utils.py .
COPY api/ ./api/
COPY models/ ./models/
COPY agent/ ./agent/

EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860", "--reload"]