@echo off
echo Pulling Llama 3.2 model...
curl -X POST http://localhost:11434/api/pull -d "{\"name\": \"llama3.2\"}"
echo Model pulled successfully!
pause
