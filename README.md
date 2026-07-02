# AI Assistant for Dyslexic Students

A friendly, dyslexia-friendly AI chatbot with multiple AI providers, image-to-text, math assistance, and more!


## Features
- 💬 Chat with local AI (Ollama) or cloud (OpenAI/Hugging Face)
- 🖼️ Image-to-text (OCR)
- 🎤 Voice input
- 🎮 Gamified learning
- 📐 Math assistant
- And many more accessibility features!


## Quick Start with Docker (Easiest!)

### Prerequisites
- Install Docker Desktop: https://www.docker.com/get-started/

### Run Everything in One Step!
1. Open terminal in this project folder
2. Run:
   ```bash
   docker-compose up -d
   ```
3. Wait a few minutes (Ollama will automatically pull the Llama 3.2 model)
4. Open your browser at: http://localhost:8501

That's it! 🎉


## Local Development (Without Docker)

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Set Up Ollama (Local AI)
- Download Ollama: https://ollama.com/download
- Run: `ollama run llama3.2`
- Keep Ollama running

### 3. Run the App
```bash
streamlit run app1.py
```


## Other Deployment Options

### Streamlit Community Cloud (FREE)
1. Push your code to GitHub
2. Go to https://share.streamlit.io/
3. Deploy directly from your repo
4. Add your secrets (OpenAI/Hugging Face keys) in the settings


## Project Files
- `app1.py`: Main application
- `requirements.txt`: Dependencies
- `Dockerfile`: For Docker builds
- `docker-compose.yml`: For full local deployment with Ollama
- `.gitignore`: Files to ignore in git


## Built-in AI Fallback Order
1. Local Ollama (best, free, no internet)
2. OpenAI (if you have API key with quota)
3. Hugging Face (if you have API key)
4. Built-in knowledge base (answers common questions)
5. Friendly fallback responses


## Adding API Keys
Create a file `.streamlit/secrets.toml` (don't commit this to git!)
```toml
OPENAI_API_KEY = "your-openai-key-here"
HUGGING_FACE_API_KEY = "your-hugging-face-key-here"
```


## License
MIT License
