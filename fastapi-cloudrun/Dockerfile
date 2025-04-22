FROM python:3.10-slim

WORKDIR /app
COPY . /app

# 使用正确的依赖文件名 requirement.txt
RUN pip install --no-cache-dir -r requirement.txt

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]