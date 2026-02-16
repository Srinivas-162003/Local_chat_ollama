const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const chooseBtn = document.getElementById('chooseBtn');
const fileListEl = document.getElementById('fileList');
const fileCountEl = document.getElementById('fileCount');
const chatWindow = document.getElementById('chatWindow');
const chatForm = document.getElementById('chatForm');
const questionInput = document.getElementById('questionInput');
const statusTag = document.getElementById('statusTag');
const debugInput = document.getElementById('debugInput');
const debugBtn = document.getElementById('debugBtn');
const debugOutput = document.getElementById('debugOutput');
const claraMode = document.getElementById('claraMode');
const claraSettings = document.getElementById('claraSettings');
const maxIterations = document.getElementById('maxIterations');
const maxHops = document.getElementById('maxHops');
const iterLabel = document.getElementById('iterLabel');
const hopsLabel = document.getElementById('hopsLabel');

let sending = false;

const sleep = (ms) => new Promise((res) => setTimeout(res, ms));

function addMessage(role, text, loading = false) {
  const bubble = document.createElement('div');
  bubble.className = `bubble ${role}`;
  if (loading) {
    bubble.innerHTML = '<div class="loader"><span></span><span></span><span></span></div>';
  } else {
    bubble.textContent = text;
  }
  chatWindow.appendChild(bubble);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return bubble;
}

async function fetchFiles() {
  try {
    const res = await fetch('/api/files');
    if (!res.ok) throw new Error('Failed to load files');
    const data = await res.json();
    renderFiles(data.files || []);
    return data.files || [];
  } catch (err) {
    renderFiles([]);
    console.error(err);
    return [];
  }
}

function renderFiles(files) {
  fileListEl.innerHTML = '';
  if (!files.length) {
    const empty = document.createElement('div');
    empty.className = 'small';
    empty.textContent = 'No documents yet. Upload to get started.';
    fileListEl.appendChild(empty);
  } else {
    files.forEach((file) => {
      const row = document.createElement('div');
      row.className = 'file-row';
      const info = document.createElement('div');
      const name = document.createElement('div');
      name.className = 'file-name';
      name.textContent = file.name;
      const meta = document.createElement('div');
      meta.className = 'file-meta';
      meta.textContent = `${file.chunks} chunks`;
      info.appendChild(name);
      info.appendChild(meta);
      const deleteBtn = document.createElement('button');
      deleteBtn.className = 'delete-btn';
      deleteBtn.textContent = '✕';
      deleteBtn.addEventListener('click', () => deleteFile(file.name));
      row.appendChild(info);
      row.appendChild(deleteBtn);
      fileListEl.appendChild(row);
    });
  }
  fileCountEl.textContent = `${files.length} file${files.length === 1 ? '' : 's'}`;
}

async function uploadFile(file) {
  const form = new FormData();
  form.append('file', file);
  statusTag.textContent = 'Uploading…';
  statusTag.style.color = '#0a131f';
  addMessage('bot', `📤 Uploading ${file.name}...`);

  try {
    const res = await fetch('/api/upload', { method: 'POST', body: form });
    if (!res.ok) {
      const error = await res.json().catch(() => ({}));
      throw new Error(error.detail || 'Upload failed');
    }
    addMessage('bot', `📄 ${file.name} uploaded. Processing with Ollama (this may take a moment)...`);
    statusTag.textContent = 'Processing…';
    const indexed = await waitForIndex(file.name);
    if (indexed) {
      await fetchFiles();
      statusTag.textContent = 'Ready';
      statusTag.style.color = '';
      addMessage('bot', `✅ Successfully indexed ${file.name}. Ask me anything about it!`);
    } else {
      statusTag.textContent = 'Processing…';
      addMessage('bot', `⏳ Processing is taking longer than expected. Trying manual trigger...`);
      try {
        await fetch(`/api/process-uploads?file_name=${encodeURIComponent(file.name)}`, { method: 'POST' });
        await sleep(3000);
        await fetchFiles();
        statusTag.textContent = 'Ready';
        addMessage('bot', `✅ ${file.name} processed!`);
      } catch (e) {
        statusTag.textContent = 'Error';
        statusTag.style.color = '#ffb4b4';
        addMessage('bot', `❌ Processing timeout. Make sure Ollama is running on port 11434. File is saved in uploads/ folder. Try refreshing the page.`);
      }
    }
  } catch (err) {
    statusTag.textContent = 'Error';
    statusTag.style.color = '#ffb4b4';
    addMessage('bot', `❌ Error: ${err.message}`);
  }
}

async function waitForIndex(fileName) {
  for (let i = 0; i < 25; i += 1) {
    await sleep(i === 0 ? 500 : 1500);
    const files = await fetchFiles();
    const found = files.find((f) => f.name === fileName && f.chunks > 0);
    if (found) return true;
    if (i % 5 === 4) {
      addMessage('bot', `⏳ Still processing ${fileName}... (${(i + 1) * 1.5}s)`);
    }
  }
  return false;
}

function setSending(state) {
  sending = state;
  questionInput.disabled = state;
  chatForm.querySelector('button').disabled = state;
  statusTag.textContent = state ? 'Thinking…' : 'Ready';
}

async function deleteFile(fileName) {
  if (!confirm(`Delete "${fileName}" from uploads and the knowledge base?`)) return;
  statusTag.textContent = 'Deleting…';
  try {
    const res = await fetch(`/api/files/${encodeURIComponent(fileName)}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Delete failed');
    addMessage('bot', `Deleted ${fileName} from uploads and the knowledge base.`);
    await fetchFiles();
    statusTag.textContent = 'Ready';
  } catch (err) {
    statusTag.textContent = 'Error';
    addMessage('bot', err.message || 'Delete error');
    setTimeout(() => {
      statusTag.textContent = 'Ready';
    }, 2000);
  }
}

async function handleQuestion(question) {
  const userBubble = addMessage('user', question);
  const botBubble = addMessage('bot', '', true);
  setSending(true);

  const useCLaRa = claraMode.checked;
  const endpoint = useCLaRa ? '/api/clara-query' : '/api/query';
  const body = useCLaRa 
    ? { 
        question, 
        max_iterations: parseInt(maxIterations.value),
        max_hops: parseInt(maxHops.value),
        detailed: true 
      }
    : { question };

  try {
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Query failed');
    }
    const data = await res.json();
    
    if (useCLaRa && data.reasoning_steps) {
      displayCLaRaResponse(botBubble, data);
    } else {
      botBubble.textContent = data.answer || 'No answer returned.';
    }
  } catch (err) {
    botBubble.textContent = err.message || 'Something went wrong.';
  } finally {
    setSending(false);
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }
}

function displayCLaRaResponse(bubble, data) {
  bubble.innerHTML = '';
  bubble.className = 'bubble bot clara-response';
  
  // Main answer
  const answerDiv = document.createElement('div');
  answerDiv.className = 'clara-answer';
  answerDiv.textContent = data.answer;
  bubble.appendChild(answerDiv);
  
  // Metadata
  const metaDiv = document.createElement('div');
  metaDiv.className = 'clara-meta';
  metaDiv.innerHTML = `
    <span class="badge-small">🔄 ${data.total_iterations} iterations</span>
    <span class="badge-small">🎯 ${(data.confidence * 100).toFixed(0)}% confidence</span>
    <span class="badge-small">🔗 ${data.reasoning_steps.length} reasoning steps</span>
  `;
  bubble.appendChild(metaDiv);
  
  // Reasoning steps (collapsible)
  if (data.reasoning_steps && data.reasoning_steps.length > 0) {
    const stepsToggle = document.createElement('details');
    stepsToggle.className = 'clara-steps';
    const summary = document.createElement('summary');
    summary.textContent = '🧠 View Reasoning Process';
    stepsToggle.appendChild(summary);
    
    data.reasoning_steps.forEach((step, idx) => {
      const stepDiv = document.createElement('div');
      stepDiv.className = 'reasoning-step';
      stepDiv.innerHTML = `
        <div class="step-header">
          <strong>Step ${step.step}</strong>
          <span class="confidence-badge">${(step.confidence * 100).toFixed(0)}% confident</span>
        </div>
        <div class="step-query"><em>Query: ${step.query}</em></div>
        <div class="step-answer">${step.answer}</div>
        <div class="step-sources">Sources: ${step.sources.join(', ')}</div>
      `;
      stepsToggle.appendChild(stepDiv);
    });
    
    bubble.appendChild(stepsToggle);
  }
  
  // Clarifications if any
  if (data.clarifications && data.clarifications.length > 0) {
    const clarifDiv = document.createElement('div');
    clarifDiv.className = 'clara-clarifications';
    clarifDiv.innerHTML = `<strong>💡 Suggested clarifications:</strong><br>${data.clarifications.join('<br>')}`;
    bubble.appendChild(clarifDiv);
  }
}

chooseBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => {
  const file = e.target.files?.[0];
  if (file) uploadFile(file);
  fileInput.value = '';
});

dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropzone.classList.add('active');
});

dropzone.addEventListener('dragleave', () => dropzone.classList.remove('active'));

dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('active');
  const file = e.dataTransfer.files?.[0];
  if (file) uploadFile(file);
});

chatForm.addEventListener('submit', (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question || sending) return;
  questionInput.value = '';
  handleQuestion(question);
});

debugBtn.addEventListener('click', async () => {
  const query = debugInput.value.trim();
  if (!query) return;
  debugOutput.textContent = 'Retrieving...';
  try {
    const res = await fetch('/api/debug-query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: query }),
    });
    if (!res.ok) throw new Error('Debug failed');
    const data = await res.json();
    let output = `Retrieved ${data.retrieved_count} documents:\n\n`;
    data.documents.forEach((doc, i) => {
      output += `📄 ${i + 1}. [${doc.source}]\n${doc.content}\n...\n\n`;
    });
    debugOutput.textContent = output;
  } catch (err) {
    debugOutput.textContent = `Error: ${err.message}`;
  }
});

// CLaRa mode toggle handlers
claraMode.addEventListener('change', (e) => {
  claraSettings.style.display = e.target.checked ? 'block' : 'none';
});

maxIterations.addEventListener('input', (e) => {
  iterLabel.textContent = e.target.value;
});

maxHops.addEventListener('input', (e) => {
  hopsLabel.textContent = e.target.value;
});

fetchFiles();
