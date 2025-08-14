#!/usr/bin/env python3
"""
Flask backend for RAG UI
Connects the web interface to the existing RAG pipeline
"""

import os
import sys
import json
import logging
import time
from datetime import datetime
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
import threading
import io
from contextlib import redirect_stdout, redirect_stderr

# Add the rag_pipeline directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rag_pipeline'))

try:
    from src.llm_query import LLMQuerySystem
except ImportError as e:
    print(f"Error importing RAG modules: {e}")
    print("Make sure you're running from the correct directory and dependencies are installed")
    sys.exit(1)

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# OpenAI API key (move to environment variable in production)
API_KEY = "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"

# Global query system instance
query_system = None

def initialize_query_system():
    """Initialize the LLM Query System"""
    global query_system
    try:
        query_system = LLMQuerySystem(API_KEY)
        logger.info("RAG Query System initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize RAG Query System: {e}")
        return False

# Removed unused LogCapture class - using simplified logging approach

@app.route('/')
def index():
    """Serve the main UI"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('.', filename)

@app.route('/api/query', methods=['POST'])
def process_query():
    """Process a RAG query and stream logs"""
    if not query_system:
        return jsonify({"error": "RAG system not initialized"}), 500
    
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Query is required"}), 400
    
    query = data['query'].strip()
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400
    
    def generate_response():
        """Generator function for streaming response"""
        try:
            # Stream initial message
            yield f"data: {json.dumps({'type': 'log', 'message': 'Starting RAG query processing...'})}\n\n"
            yield f"data: {json.dumps({'type': 'log', 'message': f'Processing query: {query}'})}\n\n"
            
            # Process the query
            start_time = time.time()
            
            # Capture logs by temporarily redirecting logging
            import logging
            log_messages = []
            
            class StreamHandler(logging.Handler):
                def emit(self, record):
                    log_msg = self.format(record)
                    log_messages.append(log_msg)
            
            # Add temporary handler
            stream_handler = StreamHandler()
            stream_handler.setLevel(logging.INFO)
            root_logger = logging.getLogger()
            root_logger.addHandler(stream_handler)
            
            try:
                result = query_system.full_query_pipeline(query)
                end_time = time.time()
                
                # Stream all captured log messages
                for log_msg in log_messages:
                    yield f"data: {json.dumps({'type': 'log', 'message': log_msg})}\n\n"
                
                yield f"data: {json.dumps({'type': 'log', 'message': f'Query completed in {end_time - start_time:.2f} seconds'})}\n\n"
                
                # Send the final result
                yield f"data: {json.dumps({'type': 'result', 'data': result})}\n\n"
                
            finally:
                # Remove the temporary handler
                root_logger.removeHandler(stream_handler)
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return Response(
        generate_response(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS'
        }
    )

@app.route('/api/history', methods=['GET'])
def get_conversation_history():
    """Get conversation history from the RAG pipeline"""
    try:
        history_file = os.path.join('..', 'rag_pipeline', 'conversation_log.json')
        
        if not os.path.exists(history_file):
            return jsonify([])
        
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        # Return the last 50 conversations
        return jsonify(history[-50:] if len(history) > 50 else history)
        
    except Exception as e:
        logger.error(f"Error loading conversation history: {e}")
        return jsonify({"error": "Failed to load conversation history"}), 500

@app.route('/api/history', methods=['POST'])
def save_conversation():
    """Save a conversation to history"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        history_file = os.path.join('..', 'rag_pipeline', 'conversation_log.json')
        
        # Load existing history
        history = []
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                history = []
        
        # Add new conversation
        history.append(data)
        
        # Save back to file
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        return jsonify({"success": True})
        
    except Exception as e:
        logger.error(f"Error saving conversation: {e}")
        return jsonify({"error": "Failed to save conversation"}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status"""
    status = {
        "rag_system_initialized": query_system is not None,
        "timestamp": datetime.now().isoformat()
    }
    
    if query_system:
        try:
            # Try to get some basic info about the system
            status["vector_store_available"] = hasattr(query_system, 'vector_store')
            status["embedder_available"] = hasattr(query_system, 'embedder')
            status["reranker_available"] = hasattr(query_system, 'reranker')
        except Exception as e:
            status["error"] = str(e)
    
    return jsonify(status)

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("="*60)
    print("🚀 Starting RAG UI Server")
    print("="*60)
    
    # Initialize the RAG system
    print("Initializing RAG Query System...")
    if initialize_query_system():
        print("✅ RAG system initialized successfully")
        print(f"🌐 Server starting at http://localhost:5002")
        print("📁 Serving UI from current directory")
        print("💡 Make sure Qdrant is running: docker run -p 6333:6333 -d qdrant/qdrant")
        print("="*60)
        
        app.run(
            host='0.0.0.0',
            port=5002,
            debug=True,
            threaded=True
        )
    else:
        print("❌ Failed to initialize RAG system")
        print("Please check:")
        print("1. Qdrant is running: docker run -p 6333:6333 -d qdrant/qdrant")
        print("2. Vector database has been populated: cd ../rag_pipeline && python3 setup_and_chunk.py")
        print("3. All dependencies are installed")
        sys.exit(1)
