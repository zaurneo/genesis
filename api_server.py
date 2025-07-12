#!/usr/bin/env python3
"""
FastAPI server with WebSocket support for Genesis Multi-Agent Stock Analysis System.
This server acts as a bridge between the React UI and the existing Genesis system.
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
import os
import sys
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# Import Genesis components
from supervisor.supervisor import create_supervisor
from agents import stock_data_agent, stock_analyzer_agent, stock_reporter_agent
from models import model_gpt_4o_mini
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from prompts import SUPERVISOR_PROMPT
from tools.logs.logging_helpers import setup_logging, log_info, log_success, log_error

# Data models
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class AgentUpdate(BaseModel):
    agent: str
    content: str
    status: str  # 'processing', 'complete', 'error'
    timestamp: str
    visualization_data: Optional[Dict[str, Any]] = None
    plotly_data: Optional[Dict[str, Any]] = None

# Global state
active_sessions: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize app on startup, cleanup on shutdown."""
    setup_logging(level="INFO")
    log_info("Genesis API Server starting...")
    yield
    log_info("Genesis API Server shutting down...")

# Create FastAPI app
app = FastAPI(
    title="Genesis Multi-Agent Stock Analysis API",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenesisSession:
    """Manages a Genesis analysis session."""
    
    def __init__(self, session_id: str, websocket: WebSocket):
        self.session_id = session_id
        self.websocket = websocket
        self.checkpointer = InMemorySaver()
        self.store = InMemoryStore()
        self.config = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": 150
        }
        self.workflow = None
        self.graph = None
        
    async def initialize(self):
        """Initialize the workflow."""
        self.workflow = create_supervisor(
            [stock_data_agent, stock_analyzer_agent, stock_reporter_agent],
            model=model_gpt_4o_mini,
            output_mode="full_history",
            prompt=SUPERVISOR_PROMPT
        )
        self.graph = self.workflow.compile(
            checkpointer=self.checkpointer,
            store=self.store
        )
        
    async def send_update(self, agent: str, content: str, status: str, 
                         visualization_data: Optional[Dict] = None,
                         plotly_data: Optional[Dict] = None):
        """Send an update to the connected client."""
        update = AgentUpdate(
            agent=agent,
            content=content,
            status=status,
            timestamp=datetime.now().isoformat(),
            visualization_data=visualization_data,
            plotly_data=plotly_data
        )
        await self.websocket.send_json(update.dict())
        
    async def process_query(self, query: str):
        """Process a user query through the Genesis workflow."""
        try:
            # Send initial acknowledgment
            await self.send_update(
                "Supervisor",
                f"Processing your request: '{query}'",
                "processing"
            )
            
            # Initialize the conversation
            inputs = {
                "messages": [
                    {
                        "role": "user",
                        "content": query
                    }
                ]
            }
            
            # Process through the workflow
# Process through the workflow
            async for chunk in self.graph.astream(inputs, config=self.config, stream_mode="updates"):
                for node_name, messages in chunk.items():
                    agent_name = self._get_friendly_agent_name(node_name)
                    
                    if isinstance(messages, dict) and "messages" in messages:
                        for msg in messages["messages"]:
                            content = str(msg.content) if hasattr(msg, 'content') else str(msg)
                            
                            # Check for visualization data in the message
                            viz_data = None
                            plotly_data = None
                            
                            # Extract any Plotly data from the message
                            if hasattr(msg, 'additional_kwargs'):
                                viz_data = msg.additional_kwargs.get('visualization_data')
                                plotly_data = msg.additional_kwargs.get('plotly_data')
                            
                            # Special handling for Reporter Agent final outputs
                            if agent_name == "Reporter Agent" and any(keyword in content.lower() for keyword in ['final report', 'executive summary', 'key findings', 'recommendations']):
                                # This is likely a final report
                                await self.send_update(
                                    agent_name,
                                    content,
                                    "complete",
                                    viz_data,
                                    plotly_data
                                )
                                
                                # Also check for any saved charts mentioned in the report
                                if "chart saved:" in content.lower() or "visualization saved:" in content.lower():
                                    # Extract chart data if available
                                    chart_data = await self._extract_chart_data(content)
                                    if chart_data:
                                        await self.send_update(
                                            agent_name,
                                            "Chart visualization ready",
                                            "complete",
                                            None,
                                            chart_data
                                        )
                            else:
                                # Regular update for other agents or non-final content
                                await self.send_update(
                                    agent_name,
                                    content,
                                    "processing",
                                    viz_data,
                                    plotly_data
                                )
                            
                            # Extract any Plotly data from the message
                            if hasattr(msg, 'additional_kwargs'):
                                viz_data = msg.additional_kwargs.get('visualization_data')
                                plotly_data = msg.additional_kwargs.get('plotly_data')
                            
                            # Send update
                            await self.send_update(
                                agent_name,
                                content,
                                "processing",
                                viz_data,
                                plotly_data
                            )
                            
                            # Check if we have chart files mentioned
                            if "CHART SAVED:" in content or "visualize_" in content:
                                # Extract chart data if available
                                chart_data = await self._extract_chart_data(content)
                                if chart_data:
                                    await self.send_update(
                                        agent_name,
                                        "Chart visualization ready",
                                        "complete",
                                        None,
                                        chart_data
                                    )
            
            # Send completion message
            await self.send_update(
                "Supervisor",
                "Analysis complete! All agents have finished their tasks.",
                "complete"
            )
            
        except Exception as e:
            log_error(f"Error processing query: {str(e)}")
            await self.send_update(
                "Supervisor",
                f"Error processing request: {str(e)}",
                "error"
            )
    
    def _get_friendly_agent_name(self, node_name: str) -> str:
        """Convert node names to friendly agent names."""
        mapping = {
            "supervisor": "Supervisor",
            "stock_data_agent": "Data Agent",
            "stock_analyzer_agent": "Analyzer Agent",
            "stock_reporter_agent": "Reporter Agent"  # Make sure this matches exactly
        }
        return mapping.get(node_name, node_name.replace('_', ' ').title())
    
    async def _extract_chart_data(self, content: str) -> Optional[Dict]:
        """Extract Plotly chart data from saved files or content."""
        try:
            # Look for saved chart files in the content
            if "Location:" in content or "CHART SAVED:" in content.upper():
                lines = content.split('\n')
                for line in lines:
                    if ("Location:" in line or "saved:" in line.lower()) and ".html" in line:
                        # Extract file path
                        import re
                        file_match = re.search(r'output/[\w\-_]+\.html', line)
                        if file_match:
                            file_path = file_match.group(0)
                        else:
                            # Try to find any .html file mentioned
                            html_match = re.search(r'[\w\-_]+\.html', line)
                            if html_match:
                                file_path = os.path.join("output", html_match.group(0))
                            else:
                                continue
                        
                        if os.path.exists(file_path):
                            # Read the HTML file and extract Plotly data
                            with open(file_path, 'r') as f:
                                html_content = f.read()
                            
                            # Simple extraction of Plotly data from HTML
                            import re
                            data_match = re.search(r'Plotly\.newPlot\([^,]+,\s*(\[.*?\])', html_content, re.DOTALL)
                            layout_match = re.search(r'Plotly\.newPlot\([^,]+,\s*\[.*?\],\s*(\{.*?\})', html_content, re.DOTALL)
                            
                            if data_match and layout_match:
                                return {
                                    "data": json.loads(data_match.group(1)),
                                    "layout": json.loads(layout_match.group(1))
                                }
        except Exception as e:
            log_error(f"Error extracting chart data: {str(e)}")
        
        return None

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication with the UI."""
    await websocket.accept()
    session_id = str(uuid.uuid4())
    session = GenesisSession(session_id, websocket)
    active_sessions[session_id] = session
    
    try:
        await session.initialize()
        log_info(f"New WebSocket connection established: {session_id}")
        
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            query = data.get("query", "")
            
            if query:
                # Process the query asynchronously
                await session.process_query(query)
                
    except WebSocketDisconnect:
        log_info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        log_error(f"WebSocket error: {str(e)}")
    finally:
        # Cleanup
        if session_id in active_sessions:
            del active_sessions[session_id]

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Genesis Multi-Agent Stock Analysis"}

@app.get("/api/sessions")
async def get_active_sessions():
    """Get list of active sessions."""
    return {
        "active_sessions": len(active_sessions),
        "sessions": list(active_sessions.keys())
    }

@app.get("/api/files/{symbol}")
async def get_symbol_files(symbol: str):
    """Get list of generated files for a symbol."""
    output_dir = Path("output")
    if not output_dir.exists():
        return {"files": []}
    
    files = []
    for file in output_dir.iterdir():
        if symbol.upper() in file.name.upper():
            files.append({
                "name": file.name,
                "size": file.stat().st_size,
                "modified": datetime.fromtimestamp(file.stat().st_mtime).isoformat(),
                "type": file.suffix
            })
    
    return {"symbol": symbol, "files": files}

@app.get("/api/chart/{filename}")
async def get_chart_data(filename: str):
    """Get Plotly chart data from a saved HTML file."""
    file_path = Path("output") / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Chart file not found")
    
    try:
        with open(file_path, 'r') as f:
            html_content = f.read()
        
        # Extract Plotly data
        import re
        data_match = re.search(r'Plotly\.newPlot\([^,]+,\s*(\[.*?\])', html_content, re.DOTALL)
        layout_match = re.search(r'Plotly\.newPlot\([^,]+,\s*\[.*?\],\s*(\{.*?\})', html_content, re.DOTALL)
        
        if data_match and layout_match:
            return {
                "data": json.loads(data_match.group(1)),
                "layout": json.loads(layout_match.group(1))
            }
        else:
            raise HTTPException(status_code=400, detail="Could not extract Plotly data")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )