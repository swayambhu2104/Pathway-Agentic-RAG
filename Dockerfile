# FROM pathwaycom/pathway:latest

# WORKDIR /app

# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

# # CMD [ "python", "./main.py" ]
# EXPOSE 8501

# CMD ["streamlit", "run", "./try.py", "--server.port", "8501", "--server.fileWatcherType", "none"]

FROM pathwaycom/pathway:latest

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Start the Streamlit app
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.fileWatcherType=none"]
