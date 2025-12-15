# Local_chat_ollama

A secure local PDF chat application powered by Ollama. Chat with your PDF documents privately without sending your data to external servers.

## About

Local_chat_ollama is a privacy-focused application that allows you to upload PDF documents and interact with them using AI. By leveraging Ollama to run Large Language Models (LLMs) locally on your machine, your sensitive data never leaves your computer - ensuring complete privacy and security.

## Features

- **PDF Upload & Chat**: Upload PDF documents and ask questions about their content
- **Complete Privacy**: All processing happens locally - your data stays on your machine
- **Ollama Integration**: Connects to a local Ollama server for AI-powered responses
- **Secure by Design**: No data sent to external APIs or cloud services
- **Configurable Models**: Choose from various Ollama models based on your needs

## Prerequisites

- macOS / Linux / Windows with WSL
- Ollama installed and running locally. See https://ollama.com for installation and model setup.
- Node.js (>=16) or Python (>=3.8) - check the actual project files in the repo.

## Quickstart

1. Install and run Ollama on your machine:

   - Follow Ollama's docs to install and run the service.
   - Pull a model, for example: `ollama pull llama2` (or any other model you prefer).

2. Configure the project (if a config file is present):

   - Look for `.env`, `config.json`, or similar files in the repository root.
   - Set the `OLLAMA_HOST` and `OLLAMA_PORT` or any other connection variable the project expects.
   - Default Ollama connection: `http://localhost:11434`

3. Run the application:

   - If there's a `package.json`: `npm install` then `npm start` or `node ./path/to/server.js`.
   - If there's a Python example: `pip install -r requirements.txt` then `python app.py`.

4. Open the UI, upload your PDF files, and start chatting with your documents securely.

## Configuration

- **Models**: Configure which Ollama model to use via environment variables or project config.
- **Connection**: Point the client to the local Ollama host and port. Typical default: `http://localhost:11434`.
- **PDF Processing**: Configure PDF parsing and chunking settings as needed.

## Privacy & Security

This application is designed with privacy as a core principle:
- ✅ All PDF processing happens locally
- ✅ AI model runs on your machine via Ollama
- ✅ No data transmission to external servers
- ✅ Your sensitive documents remain secure

## Development

- Update code and keep configurations out of source control where appropriate.
- Add more features or improve the UI to enhance user experience.

## Troubleshooting

- If you can't connect, ensure Ollama is running and listening on the expected port.
- Check firewall rules or local proxies that might block connections.
- Verify that your PDF files are readable and not corrupted.

## Contributing

Contributions are welcome! Please open issues or PRs with clear descriptions. If you add features, include README updates and examples.

## License

If this project has no license yet, add one (for example, MIT) or contact the repository owner to set the appropriate license.

---

**Built with privacy in mind. Your data, your machine, your control.**