# 🎯 Financial RAG System UI

A clean, minimal web interface for the Financial RAG System. Provides real-time query processing, live log streaming, and conversation history management.

## 🎨 Design Philosophy

- **Minimalist**: Black, white, and grey color palette
- **Clean Typography**: System fonts for UI, monospace for logs
- **Generous Spacing**: High contrast, readable layout
- **Real-time**: Live log streaming during query processing
- **Provenance**: Full source attribution and metadata

## 🚀 Quick Start

### 1. Prerequisites
Ensure your RAG pipeline is set up and running:

```bash
# In the main rag_pipeline directory
cd ../rag_pipeline

# Make sure Qdrant is running
docker run -p 6333:6333 -d qdrant/qdrant

# Ensure vector database is populated
python3 setup_and_chunk.py
```

### 2. Install UI Dependencies
```bash
# In the rag-ui directory
pip3 install -r requirements.txt
```

### 3. Start the UI Server
```bash
python3 app.py
```

The UI will be available at: **http://localhost:5001**

## 🔧 Features

### **Query Interface**
- Clean search input with placeholder examples
- Real-time query processing with live log streaming
- Answer display with thinking section (collapsible logs)

### **Performance Metrics**
- Total processing time
- Token usage tracking
- CPU and memory usage monitoring
- Real-time resource monitoring

### **Source Attribution**
- Top source chunks with metadata
- Chunk type indicators (table/paragraph/kv)
- Similarity and rerank scores
- Metadata tags (source, section, page numbers)

### **Conversation History**
- Saved query/answer pairs
- Browsable chat history
- Detailed view of past queries
- Automatic history persistence

## 📁 Project Structure

```
rag-ui/
├── index.html          # Main UI page
├── app.py             # Flask backend server
├── requirements.txt   # Python dependencies
├── styles/
│   └── main.css       # Minimal CSS styling
├── js/
│   └── main.js        # Frontend JavaScript
└── README.md          # This file
```

## 🎯 UI Components

### **Main Interface**
- **Query Form**: Input field with search and history buttons
- **Answer Card**: Large, readable response display
- **Thinking Section**: Collapsible live log streaming
- **Metrics Panel**: Performance and resource usage
- **Sources Panel**: Retrieved chunks with metadata

### **Conversation History**
- **History Panel**: Slides in from the right
- **Chat List**: Chronological list of past queries
- **Details Pane**: Full query/answer/sources view

## 🎨 Design System

### **Colors**
- **Primary**: #000000 (Black)
- **Background**: #FFFFFF (White)
- **Accent**: #F5F5F5 (Light Grey)
- **Borders**: #E5E7EB (Border Grey)
- **Text Secondary**: #6B7280 (Text Grey)

### **Typography**
- **UI Text**: System sans-serif fonts
- **Logs**: Monospace fonts (Monaco, Menlo, Ubuntu Mono)
- **Generous line spacing** for readability

### **Layout**
- **Card-based design** with subtle borders
- **Grid layout** for responsive design
- **Minimal shadows** (1px subtle borders only)

## 🔌 API Endpoints

### **POST /api/query**
Process a RAG query with streaming logs
```json
{
  "query": "Tell me about the EPS this year"
}
```

### **GET /api/history**
Retrieve conversation history (last 50 items)

### **POST /api/history**
Save a conversation to history

### **GET /api/status**
Get system status and health check

## 🚀 Usage Examples

### **Financial Queries**
```
"Tell me about the EPS this year"
"What are the revenue figures?"
"Show me information about iPhone sales"
"Explain the balance sheet changes"
"What are the cash flow figures?"
```

### **Expected Response Flow**
1. Submit query → UI shows "Thinking..." state
2. Live logs stream in collapsible section
3. Answer appears in main content area
4. Metrics update with timing/resource usage
5. Source chunks display with metadata
6. Conversation saved to history automatically

## 🔧 Development

### **Local Development**
```bash
# Start the Flask server in debug mode
python3 app.py

# The server will reload automatically on file changes
```

### **Customization**
- **Styling**: Modify `styles/main.css`
- **Frontend Logic**: Update `js/main.js`
- **Backend API**: Extend `app.py`

## 🔍 Troubleshooting

### **Common Issues**

**UI doesn't load**
- Check that Flask server is running on port 5001
- Verify no other services are using port 5001

**No query results**
- Ensure Qdrant is running: `docker ps | grep qdrant`
- Check vector database is populated: `cd ../rag_pipeline && python3 query.py "test"`

**Log streaming not working**
- Verify browser supports Server-Sent Events (SSE)
- Check browser console for JavaScript errors

**History not loading**
- Ensure `../rag_pipeline/conversation_log.json` exists
- Check file permissions for read/write access

## 📊 Performance

The UI is optimized for:
- **Real-time log streaming** without blocking
- **Responsive design** for desktop and mobile
- **Minimal resource usage** with efficient JavaScript
- **Fast rendering** with optimized CSS

## 🎯 Production Notes

For production deployment:
1. Move OpenAI API key to environment variable
2. Add proper error logging and monitoring  
3. Implement authentication if needed
4. Consider using a production WSGI server (gunicorn)
5. Add rate limiting for API endpoints

---

**Built with**: Flask, Vanilla JavaScript, CSS Grid, Server-Sent Events
