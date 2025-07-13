import React, { useState, useRef, useEffect } from 'react';
import { Brain, Send, Bot, TrendingUp, BarChart3, FileText, Loader2, Activity, Database, AlertCircle, Eye, EyeOff } from 'lucide-react';
import * as Plotly from 'plotly.js-dist';
import './App.css';


// Configuration
const API_CONFIG = {
  WS_URL: 'ws://localhost:8000/ws',
  API_BASE: 'http://localhost:8000',
  USE_MOCK: false  // Set to true to use mock data without backend
};

const GenesisReportViewer = () => {
  const [query, setQuery] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeAgent, setActiveAgent] = useState(null);
  const [wsConnection, setWsConnection] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [reportContent, setReportContent] = useState('');
  const [visualizations, setVisualizations] = useState([]);
  const [showDebugMessages, setShowDebugMessages] = useState(false);
  const [debugMessages, setDebugMessages] = useState([]);
  
  const plotlyRefs = useRef({});

  const connectWebSocket = () => {
    if (!API_CONFIG.USE_MOCK) {
      try {
        const ws = new WebSocket(API_CONFIG.WS_URL);
        
        ws.onopen = () => {
          console.log('WebSocket connected');
          setConnectionStatus('connected');
          setWsConnection(ws);
        };

        ws.onmessage = (event) => {
          const agentUpdate = JSON.parse(event.data);
          handleAgentUpdate(agentUpdate);
        };

        ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          setConnectionStatus('error');
        };

        ws.onclose = () => {
          console.log('WebSocket disconnected');
          setConnectionStatus('disconnected');
          setWsConnection(null);
          // Try to reconnect after 3 seconds
          setTimeout(connectWebSocket, 3000);
        };

        return ws;
      } catch (error) {
        console.error('Failed to connect WebSocket:', error);
        setConnectionStatus('error');
      }
    }
  };

  useEffect(() => {
    const ws = connectWebSocket();
    return () => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, []);

  useEffect(() => {
    // Render Plotly charts when visualizations update
    visualizations.forEach((viz, index) => {
      const ref = plotlyRefs.current[`viz-${index}`];
      if (ref && viz.plotlyData) {
        try {
          Plotly.newPlot(
            ref,
            viz.plotlyData.data,
            viz.plotlyData.layout,
            { responsive: true }
          );
        } catch (e) {
          console.error('Error rendering Plotly chart:', e);
        }
      }
    });
  }, [visualizations]);

  const handleAgentUpdate = (agentUpdate) => {
    // Store all messages for debug view
    const debugMessage = {
      id: Date.now(),
      agent: agentUpdate.agent,
      content: agentUpdate.content,
      status: agentUpdate.status,
      timestamp: new Date(agentUpdate.timestamp).toLocaleTimeString()
    };
    setDebugMessages(prev => [...prev, debugMessage]);

    // Update active agent indicator
    if (agentUpdate.status === 'processing') {
      setActiveAgent(agentUpdate.agent);
    } else if (agentUpdate.status === 'complete' || agentUpdate.status === 'error') {
      if (activeAgent === agentUpdate.agent) {
        setActiveAgent(null);
      }
    }

    // Only process final outputs from Reporter Agent
    if (agentUpdate.agent === 'Reporter Agent' || agentUpdate.agent === 'Stock Reporter Agent') {
      // Only process if status is 'complete' - these are final outputs
      if (agentUpdate.status === 'complete') {
        const content = agentUpdate.content.toLowerCase();
        
        // Check if this is a final report
        const isFinalReport = 
          content.includes('report') || 
          content.includes('analysis') || 
          content.includes('summary') ||
          content.includes('recommendation') ||
          content.includes('conclusion') ||
          content.includes('findings') ||
          content.includes('executive summary');

        if (isFinalReport && !content.includes('chart visualization ready')) {
          // This is the final report - set it (don't append)
          setReportContent(agentUpdate.content);
        }

        // Check for visualizations
        if (agentUpdate.plotly_data) {
          const newViz = {
            id: Date.now(),
            title: extractChartTitle(agentUpdate.content),
            plotlyData: agentUpdate.plotly_data,
            content: agentUpdate.content
          };
          setVisualizations(prev => [...prev, newViz]);
        }
      }
    }

    // Check if all agents completed
    if (agentUpdate.agent === 'Supervisor' && agentUpdate.content.includes('Analysis complete')) {
      setIsProcessing(false);
    }
  };

  const extractChartTitle = (content) => {
    // Try to extract chart type from content
    if (content.includes('backtesting')) return 'Backtesting Results';
    if (content.includes('comparison')) return 'Model Comparison';
    if (content.includes('performance')) return 'Performance Analysis';
    if (content.includes('price')) return 'Price Chart';
    if (content.includes('volume')) return 'Volume Analysis';
    return 'Stock Analysis Chart';
  };

  const handleAnalyze = async () => {
    if (!query.trim() || isProcessing) return;

    // Clear previous results
    setReportContent('');
    setVisualizations([]);
    setDebugMessages([]);
    setIsProcessing(true);

    if (API_CONFIG.USE_MOCK) {
      // Mock mode
      setTimeout(() => {
        setReportContent(generateMockReport(query));
        setVisualizations(generateMockVisualizations());
        setIsProcessing(false);
      }, 2000);
    } else {
      // Send to WebSocket
      if (wsConnection && wsConnection.readyState === WebSocket.OPEN) {
        wsConnection.send(JSON.stringify({ query }));
      } else {
        // Connection error
        setReportContent('⚠️ Connection error. Please check if the Genesis backend is running on ' + API_CONFIG.WS_URL);
        setIsProcessing(false);
      }
    }
  };

  const generateMockReport = (query) => {
    return `# Stock Analysis Report

## Executive Summary
Based on the analysis of ${query}, here are the key findings...

## Technical Analysis
- Current Price: $150.25
- 50-day SMA: $148.50
- RSI: 62.3 (Neutral)
- MACD: Bullish crossover detected

## Model Performance
- XGBoost Model: 72.5% accuracy
- Random Forest: 68.3% accuracy
- Ensemble Prediction: Moderate bullish signal

## Recommendations
Based on the multi-agent analysis, the stock shows positive momentum with strong technical indicators...`;
  };

  const generateMockVisualizations = () => {
    return [
      {
        id: 1,
        title: 'Price Chart',
        plotlyData: {
          data: [{
            x: Array.from({length: 30}, (_, i) => new Date(2024, 0, i + 1)),
            y: Array.from({length: 30}, (_, i) => 140 + Math.random() * 20),
            type: 'scatter',
            mode: 'lines',
            name: 'Stock Price'
          }],
          layout: {
            title: 'Stock Price History',
            xaxis: { title: 'Date' },
            yaxis: { title: 'Price ($)' }
          }
        }
      }
    ];
  };

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex flex-col">
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800">
        <div className="px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center">
              <Brain className="h-8 w-8 text-purple-500 mr-3" />
              <div>
                <h1 className="text-xl font-bold bg-gradient-to-r from-purple-400 to-blue-500 bg-clip-text text-transparent">
                  Genesis AI Report Viewer
                </h1>
                <p className="text-xs text-gray-400">Multi-Agent Stock Analysis System</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              {activeAgent && (
                <div className="flex items-center bg-purple-900/20 border border-purple-700 rounded-lg px-3 py-1">
                  <Activity className="h-4 w-4 text-purple-400 animate-pulse mr-2" />
                  <span className="text-sm text-purple-300">{activeAgent} Active</span>
                </div>
              )}
              <button
                onClick={() => setShowDebugMessages(!showDebugMessages)}
                className="flex items-center space-x-2 px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors"
                title="Toggle debug messages"
              >
                {showDebugMessages ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                <span className="text-sm">Debug</span>
              </button>
              <div className="flex items-center space-x-2 text-sm text-gray-400">
                <div className="flex items-center">
                  <div className={`w-2 h-2 rounded-full mr-2 ${
                    connectionStatus === 'connected' ? 'bg-green-500 animate-pulse' : 
                    connectionStatus === 'error' ? 'bg-red-500' : 
                    'bg-gray-500'
                  }`}></div>
                  <span>
                    {connectionStatus === 'connected' ? 'Connected' : 
                     connectionStatus === 'error' ? 'Connection Error' : 
                     'Connecting...'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Query Input */}
      <div className="bg-gray-900 border-b border-gray-800 px-4 py-4">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center space-x-4">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleAnalyze()}
              placeholder="Enter stock symbol and analysis request (e.g., 'Analyze AAPL with ML models')"
              className="flex-1 bg-gray-800 text-gray-100 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-purple-500 border border-gray-700"
              disabled={isProcessing}
            />
            <button
              onClick={handleAnalyze}
              disabled={!query.trim() || isProcessing}
              className="px-6 py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-700 disabled:text-gray-500 rounded-lg font-medium transition-colors flex items-center space-x-2"
            >
              {isProcessing ? (
                <>
                  <Loader2 className="h-5 w-5 animate-spin" />
                  <span>Analyzing...</span>
                </>
              ) : (
                <>
                  <TrendingUp className="h-5 w-5" />
                  <span>Analyze</span>
                </>
              )}
            </button>
          </div>
          <div className="flex items-center mt-2 text-xs text-gray-500">
            <Database className="h-3 w-3 mr-1" />
            <span>Yahoo Finance</span>
            <span className="mx-2">•</span>
            <Brain className="h-3 w-3 mr-1" />
            <span>GPT-4 Enhanced</span>
            <span className="mx-2">•</span>
            <TrendingUp className="h-3 w-3 mr-1" />
            <span>XGBoost & Random Forest</span>
          </div>
        </div>
      </div>

      {/* Connection Error Alert */}
      {!API_CONFIG.USE_MOCK && connectionStatus === 'error' && (
        <div className="bg-red-900/20 border border-red-700 px-4 py-3">
          <div className="max-w-7xl mx-auto flex items-center">
            <AlertCircle className="h-5 w-5 text-red-400 mr-2 flex-shrink-0" />
            <div>
              <p className="text-sm text-red-300">
                <strong>Connection Error:</strong> Cannot connect to Genesis backend at {API_CONFIG.WS_URL}
              </p>
              <p className="text-xs text-red-400 mt-1">
                Make sure the backend is running: <code className="bg-red-900/50 px-1 rounded">python api_server.py</code>
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Main Content Area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel - Report Content */}
        <div className="flex-1 flex flex-col bg-gray-900 border-r border-gray-800">
          <div className="bg-gray-800 px-4 py-2 border-b border-gray-700">
            <h2 className="text-lg font-semibold flex items-center">
              <FileText className="h-5 w-5 mr-2 text-purple-400" />
              Analysis Report
            </h2>
          </div>
          <div className="flex-1 overflow-y-auto p-6">
            {reportContent ? (
              <div className="prose prose-invert max-w-none">
                <div className="bg-gray-800 rounded-lg p-6 whitespace-pre-wrap">
                  {reportContent}
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-center h-full text-gray-500">
                <div className="text-center">
                  <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No report generated yet</p>
                  <p className="text-sm mt-2">Enter a query above to start analysis</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Panel - Visualizations */}
        <div className="flex-1 flex flex-col bg-gray-950">
          <div className="bg-gray-800 px-4 py-2 border-b border-gray-700">
            <h2 className="text-lg font-semibold flex items-center">
              <BarChart3 className="h-5 w-5 mr-2 text-purple-400" />
              Visualizations
            </h2>
          </div>
          <div className="flex-1 overflow-y-auto p-6">
            {visualizations.length > 0 ? (
              <div className="space-y-6">
                {visualizations.map((viz, index) => (
                  <div key={viz.id} className="bg-gray-900 rounded-lg p-4 border border-gray-800">
                    <h3 className="text-md font-medium mb-3 text-gray-200">{viz.title}</h3>
                    <div 
                      ref={el => plotlyRefs.current[`viz-${index}`] = el}
                      className="w-full h-96 bg-gray-950 rounded"
                    />
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex items-center justify-center h-full text-gray-500">
                <div className="text-center">
                  <BarChart3 className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No visualizations generated yet</p>
                  <p className="text-sm mt-2">Charts will appear here after analysis</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Debug Panel (Hidden by default) */}
      {showDebugMessages && (
        <div className="fixed bottom-0 left-0 right-0 h-64 bg-gray-900 border-t border-gray-800 overflow-y-auto p-4 z-50">
          <div className="flex justify-between items-center mb-2">
            <h3 className="text-sm font-semibold text-gray-400">Debug Messages</h3>
            <button
              onClick={() => setShowDebugMessages(false)}
              className="text-gray-500 hover:text-gray-300"
            >
              ✕
            </button>
          </div>
          <div className="space-y-2 text-xs">
            {debugMessages.slice(-50).map((msg) => (
              <div key={msg.id} className="flex">
                <span className="text-gray-500 mr-2">{msg.timestamp}</span>
                <span className="text-purple-400 mr-2">[{msg.agent}]</span>
                <span className="text-gray-300 flex-1">{msg.content.substring(0, 100)}...</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default GenesisReportViewer;