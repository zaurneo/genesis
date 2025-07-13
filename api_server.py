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
                            if agent_name == "Reporter Agent":
                                # Check if this is a final report
                                is_final_report = any(keyword in content.lower() for keyword in [
                                    'final report', 'executive summary', 'key findings', 
                                    'recommendations', 'conclusion', 'investment recommendation',
                                    'analysis complete', 'report generated'
                                ])
                                
                                if is_final_report:
                                    # Send as complete for final reports
                                    await self.send_update(
                                        agent_name,
                                        content,
                                        "complete",
                                        viz_data,
                                        plotly_data
                                    )
                                    
                                    # Check for any saved charts mentioned in the report
                                    if "CHART SAVED:" in content or "visualize_" in content:
                                        # Extract chart info
                                        chart_info = await self._extract_chart_data(content)
                                        if chart_info:
                                            # If we successfully extracted Plotly data, send it
                                            if 'data' in chart_info and 'layout' in chart_info:
                                                await self.send_update(
                                                    agent_name,
                                                    f"Chart visualization ready: {chart_info.get('file', 'chart.html')}",
                                                    "complete",
                                                    None,
                                                    chart_info  # This contains data and layout
                                                )
                                            else:
                                                # Otherwise send as visualization_data
                                                await self.send_update(
                                                    agent_name,
                                                    f"Visualization created: {chart_info.get('file', 'chart.html')}",
                                                    "complete",
                                                    {"chart_available": True, "chart_info": chart_info},
                                                    None
                                                )
                                # Skip intermediate Reporter Agent messages
                                # Only send if it's a final output
                            elif agent_name != "Reporter Agent":
                                # Send updates for other agents
                                await self.send_update(
                                    agent_name,
                                    content,
                                    "processing",
                                    viz_data,
                                    plotly_data
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
            if "CHART SAVED:" in content.upper() or "Location:" in content:
                lines = content.split('\n')
                for line in lines:
                    if ".html" in line:
                        # Extract file path
                        import re
                        # Look for various patterns of file paths
                        patterns = [
                            r'output/[\w\-_]+\.html',
                            r'Location:\s*(.+\.html)',
                            r'saved:\s*(.+\.html)',
                            r'(visualize_[\w\-_]+\.html)'
                        ]
                        
                        file_path = None
                        for pattern in patterns:
                            match = re.search(pattern, line, re.IGNORECASE)
                            if match:
                                if match.group(0).startswith('output/'):
                                    file_path = match.group(0)
                                else:
                                    file_path = os.path.join("output", match.group(1) if len(match.groups()) > 0 else match.group(0))
                                break
                        
                        if file_path and os.path.exists(file_path):
                            # Try to extract Plotly data from the HTML file
                            try:
                                with open(file_path, 'r') as f:
                                    html_content = f.read()
                                
                                # Extract Plotly data more robustly
                                # Find the Plotly.newPlot call
                                if 'Plotly.newPlot(' in html_content:
                                    try:
                                        # Find the start of the Plotly.newPlot call
                                        start_idx = html_content.find('Plotly.newPlot(')
                                        if start_idx == -1:
                                            raise ValueError("Could not find Plotly.newPlot")
                                        
                                        # Move past 'Plotly.newPlot('
                                        content_start = start_idx + len('Plotly.newPlot(')
                                        
                                        # Extract the entire call by counting parentheses
                                        paren_count = 1
                                        idx = content_start
                                        while paren_count > 0 and idx < len(html_content):
                                            if html_content[idx] == '(':
                                                paren_count += 1
                                            elif html_content[idx] == ')':
                                                paren_count -= 1
                                            idx += 1
                                        
                                        # Extract the parameters
                                        params_str = html_content[content_start:idx-1]
                                        
                                        # Split by the first comma after the div ID
                                        # Find the div ID end
                                        div_end = params_str.find('",') + 1
                                        if div_end == 0:
                                            div_end = params_str.find("',") + 1
                                        
                                        # Extract data and layout parts
                                        remaining = params_str[div_end:].strip()
                                        if remaining.startswith(','):
                                            remaining = remaining[1:].strip()
                                        
                                        # Find the end of the data array
                                        bracket_count = 0
                                        data_end = 0
                                        for i, char in enumerate(remaining):
                                            if char == '[':
                                                bracket_count += 1
                                            elif char == ']':
                                                bracket_count -= 1
                                                if bracket_count == 0:
                                                    data_end = i + 1
                                                    break
                                        
                                        data_str = remaining[:data_end]
                                        
                                        # Find layout - it's after the data array and a comma
                                        layout_start = data_end
                                        while layout_start < len(remaining) and remaining[layout_start] in ' ,':
                                            layout_start += 1
                                        
                                        # Find the end of the layout object
                                        brace_count = 0
                                        layout_end = layout_start
                                        for i in range(layout_start, len(remaining)):
                                            if remaining[i] == '{':
                                                brace_count += 1
                                            elif remaining[i] == '}':
                                                brace_count -= 1
                                                if brace_count == 0:
                                                    layout_end = i + 1
                                                    break
                                        
                                        layout_str = remaining[layout_start:layout_end]
                                        
                                        # Parse the JSON
                                        chart_data = json.loads(data_str)
                                        chart_layout = json.loads(layout_str)
                                        
                                        return {
                                            "data": chart_data,
                                            "layout": chart_layout,
                                            "file": os.path.basename(file_path)
                                        }
                                    except Exception as e:
                                        log_error(f"Error extracting Plotly data: {str(e)}")
                                        # Try simpler regex as fallback
                                        import re
                                        plotly_match = re.search(
                                            r'Plotly\.newPlot\([^,]+,\s*(\[[^\]]+\]),\s*(\{[^}]+\})',
                                            html_content
                                        )
                                        if plotly_match:
                                            try:
                                                chart_data = json.loads(plotly_match.group(1))
                                                chart_layout = json.loads(plotly_match.group(2))
                                                return {
                                                    "data": chart_data,
                                                    "layout": chart_layout,
                                                    "file": os.path.basename(file_path)
                                                }
                                            except:
                                                pass
                            except Exception as parse_error:
                                log_error(f"Error parsing Plotly data from {file_path}: {str(parse_error)}")
                            
                            # Fallback to just returning file reference
                            return {
                                "type": "chart_reference",
                                "file": os.path.basename(file_path),
                                "message": f"Chart saved: {os.path.basename(file_path)}"
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
        
        # Extract Plotly data using the same logic as _extract_chart_data
        if 'Plotly.newPlot(' in html_content:
            try:
                # Find the start of the Plotly.newPlot call
                start_idx = html_content.find('Plotly.newPlot(')
                if start_idx == -1:
                    raise ValueError("Could not find Plotly.newPlot")
                
                # Move past 'Plotly.newPlot('
                content_start = start_idx + len('Plotly.newPlot(')
                
                # Extract the entire call by counting parentheses
                paren_count = 1
                idx = content_start
                while paren_count > 0 and idx < len(html_content):
                    if html_content[idx] == '(':
                        paren_count += 1
                    elif html_content[idx] == ')':
                        paren_count -= 1
                    idx += 1
                
                # Extract the parameters
                params_str = html_content[content_start:idx-1]
                
                # Split by the first comma after the div ID
                # Find the div ID end
                div_end = params_str.find('",') + 1
                if div_end == 0:
                    div_end = params_str.find("',") + 1
                
                # Extract data and layout parts
                remaining = params_str[div_end:].strip()
                if remaining.startswith(','):
                    remaining = remaining[1:].strip()
                
                # Find the end of the data array
                bracket_count = 0
                data_end = 0
                for i, char in enumerate(remaining):
                    if char == '[':
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            data_end = i + 1
                            break
                
                data_str = remaining[:data_end]
                
                # Find layout - it's after the data array and a comma
                layout_start = data_end
                while layout_start < len(remaining) and remaining[layout_start] in ' ,':
                    layout_start += 1
                
                # Find the end of the layout object
                brace_count = 0
                layout_end = layout_start
                for i in range(layout_start, len(remaining)):
                    if remaining[i] == '{':
                        brace_count += 1
                    elif remaining[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            layout_end = i + 1
                            break
                
                layout_str = remaining[layout_start:layout_end]
                
                # Parse the JSON
                return {
                    "data": json.loads(data_str),
                    "layout": json.loads(layout_str)
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Could not extract Plotly data: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="No Plotly.newPlot found in HTML")
            
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