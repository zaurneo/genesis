import React, { useState, useRef, useEffect } from 'react';
import { Brain, Send, Bot, User, TrendingUp, BarChart3, FileText, Loader2, ChevronRight, Activity, Sparkles, Database, Play, X, Maximize2, Minimize2, AlertCircle } from 'lucide-react';
import * as Plotly from 'plotly';

// Configuration
const API_CONFIG = {
  WS_URL: 'ws://localhost:8000/ws',
  API_BASE: 'http://localhost:8000',
  USE_MOCK: false  // Set to true to use mock data without backend
};

const GenesisAIApp = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'ai',
      agent: 'Supervisor',
      content: "Welcome to Genesis AI! I'm your trading analysis supervisor. I coordinate with our specialized agents:\n\n• **Stock Data Agent** - Fetches market data and technical indicators\n• **Stock Analyzer Agent** - Trains ML models and performs backtesting\n• **Stock Reporter Agent** - Creates visualizations and reports\n\nTry asking: 'Analyze Apple stock with ML models' or 'Compare XGBoost vs Random Forest for Tesla'",
      timestamp: new Date().toLocaleTimeString(),
      status: 'complete'
    }
  ]);
  const [input, setInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeAgent, setActiveAgent] = useState(null);
  const [showVisualization, setShowVisualization] = useState(false);
  const [expandedView, setExpandedView] = useState(false);
  const [currentPlotlyData, setCurrentPlotlyData] = useState(null);
  const [wsConnection, setWsConnection] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const messagesEndRef = useRef(null);
  const plotlyRefs = useRef({});

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
    
    // Render any Plotly charts in new messages
    messages.forEach(message => {
      if (message.plotlyData && plotlyRefs.current[message.id]) {
        try {
          Plotly.newPlot(
            plotlyRefs.current[message.id],
            message.plotlyData.data,
            message.plotlyData.layout,
            { responsive: true }
          );
        } catch (e) {
          console.error('Error rendering Plotly chart:', e);
        }
      }
    });
  }, [messages]);

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

  const handleAgentUpdate = (agentUpdate) => {
    const newMessage = {
      id: Date.now(),
      type: 'ai',
      agent: agentUpdate.agent,
      content: agentUpdate.content,
      status: agentUpdate.status,
      timestamp: new Date(agentUpdate.timestamp).toLocaleTimeString(),
      hasVisualization: agentUpdate.visualization_data ? true : false,
      plotlyData: agentUpdate.plotly_data || null
    };

    setMessages(prev => [...prev, newMessage]);

    // Update active agent indicator
    if (agentUpdate.status === 'processing') {
      setActiveAgent(agentUpdate.agent);
    } else if (agentUpdate.status === 'complete' || agentUpdate.status === 'error') {
      if (activeAgent === agentUpdate.agent) {
        setActiveAgent(null);
      }
      if (agentUpdate.status === 'complete' && messages.length > 0) {
        setIsProcessing(false);
      }
    }
  };

  const handleSendMessage = async () => {
    if (!input.trim() || isProcessing) return;

    const userMessage = {
      id: messages.length + 1,
      type: 'user',
      content: input,
      timestamp: new Date().toLocaleTimeString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsProcessing(true);

    if (API_CONFIG.USE_MOCK) {
      // Mock mode
      setTimeout(() => {
        const aiResponse = generateMockResponse(input);
        setMessages(prev => [...prev, aiResponse]);
        setIsProcessing(false);
      }, 2000);
    } else {
      // Send to WebSocket
      if (wsConnection && wsConnection.readyState === WebSocket.OPEN) {
        wsConnection.send(JSON.stringify({ query: input }));
      } else {
        // Connection error
        setMessages(prev => [...prev, {
          id: Date.now(),
          type: 'ai',
          agent: 'System',
          content: "⚠️ Connection error. Please check if the Genesis backend is running on " + API_CONFIG.WS_URL,
          timestamp: new Date().toLocaleTimeString(),
          status: 'error'
        }]);
        setIsProcessing(false);
      }
    }
  };

  const generateMockResponse = (query) => {
    const lowerQuery = query.toLowerCase();
    
    if (lowerQuery.includes('analyze') && (lowerQuery.includes('apple') || lowerQuery.includes('aapl'))) {
      return {
        id: Date.now(),
        type: 'ai',
        agent: 'Supervisor',
        content: "I'll analyze Apple (AAPL) stock using our multi-agent system. Starting the analysis now...",
        timestamp: new Date().toLocaleTimeString(),
        status: 'complete'
      };
    }
    
    return {
      id: Date.now(),
      type: 'ai',
      agent: 'Supervisor',
      content: "I understand your request. Please be more specific about which stock you'd like to analyze and what type of analysis you need (e.g., 'Analyze TSLA with ML models' or 'Compare models for MSFT').",
      timestamp: new Date().toLocaleTimeString(),
      status: 'complete'
    };
  };

  const suggestedQueries = [
    "Analyze Apple stock with ML models",
    "Compare XGBoost vs Random Forest for Tesla",
    "Backtest trading strategies for Microsoft",
    "Generate comprehensive report for Amazon"
  ];

  const handleShowVisualization = (plotlyData) => {
    setCurrentPlotlyData(plotlyData);
    setShowVisualization(true);
  };

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex flex-col">
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center">
              <Brain className="h-8 w-8 text-purple-500 mr-3" />
              <div>
                <h1 className="text-xl font-bold bg-gradient-to-r from-purple-400 to-blue-500 bg-clip-text text-transparent">
                  Genesis AI
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

      {/* Main Chat Interface */}
      <div className="flex-1 flex overflow-hidden">
        {/* Chat Area */}
        <div className="flex-1 flex flex-col">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-4 py-6">
            {!API_CONFIG.USE_MOCK && connectionStatus === 'error' && (
              <div className="max-w-4xl mx-auto mb-4 p-4 bg-red-900/20 border border-red-700 rounded-lg">
                <div className="flex items-center">
                  <AlertCircle className="h-5 w-5 text-red-400 mr-2" />
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
            <div className="max-w-4xl mx-auto space-y-6">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div className={`flex items-start max-w-3xl ${message.type === 'user' ? 'flex-row-reverse' : ''}`}>
                    <div className={`flex-shrink-0 ${message.type === 'user' ? 'ml-3' : 'mr-3'}`}>
                      {message.type === 'user' ? (
                        <div className="w-10 h-10 bg-blue-600 rounded-full flex items-center justify-center">
                          <User className="h-5 w-5" />
                        </div>
                      ) : (
                        <div className="w-10 h-10 bg-gradient-to-br from-purple-600 to-purple-800 rounded-full flex items-center justify-center">
                          <Bot className="h-5 w-5" />
                        </div>
                      )}
                    </div>
                    <div className={`flex-1 ${message.type === 'user' ? 'text-right' : ''}`}>
                      {message.agent && (
                        <div className="flex items-center mb-1">
                          <span className="text-xs text-purple-400 font-medium">{message.agent}</span>
                          {message.status === 'processing' && (
                            <Loader2 className="h-3 w-3 ml-2 animate-spin text-purple-400" />
                          )}
                          {message.status === 'error' && (
                            <AlertCircle className="h-3 w-3 ml-2 text-red-400" />
                          )}
                        </div>
                      )}
                      <div
                        className={`inline-block px-4 py-2 rounded-lg ${
                          message.type === 'user'
                            ? 'bg-blue-600 text-white'
                            : message.status === 'error'
                            ? 'bg-red-900/20 border border-red-700'
                            : 'bg-gray-800 border border-gray-700'
                        }`}
                      >
                        <p className="whitespace-pre-wrap">{message.content}</p>
                        {message.plotlyData && (
                          <div className="mt-4">
                            <div 
                              ref={el => plotlyRefs.current[message.id] = el}
                              className="w-full h-96 rounded-lg overflow-hidden bg-gray-900"
                            />
                          </div>
                        )}
                        {message.hasVisualization && message.plotlyData && (
                          <button
                            onClick={() => handleShowVisualization(message.plotlyData)}
                            className="mt-3 flex items-center text-purple-400 hover:text-purple-300 transition-colors"
                          >
                            <BarChart3 className="h-4 w-4 mr-2" />
                            View Full Chart
                            <ChevronRight className="h-4 w-4 ml-1" />
                          </button>
                        )}
                      </div>
                      <p className="text-xs text-gray-500 mt-1">{message.timestamp}</p>
                    </div>
                  </div>
                </div>
              ))}
              {isProcessing && (
                <div className="flex justify-start">
                  <div className="flex items-start max-w-3xl">
                    <div className="w-10 h-10 bg-gradient-to-br from-purple-600 to-purple-800 rounded-full flex items-center justify-center mr-3">
                      <Bot className="h-5 w-5" />
                    </div>
                    <div className="bg-gray-800 border border-gray-700 rounded-lg px-4 py-2">
                      <div className="flex items-center">
                        <Loader2 className="h-4 w-4 animate-spin text-purple-400 mr-2" />
                        <span className="text-gray-400">Genesis AI is processing...</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* Suggested Queries */}
          {messages.length === 1 && (
            <div className="px-4 pb-4">
              <div className="max-w-4xl mx-auto">
                <p className="text-sm text-gray-400 mb-2">Try these queries:</p>
                <div className="grid grid-cols-2 gap-2">
                  {suggestedQueries.map((query, index) => (
                    <button
                      key={index}
                      onClick={() => setInput(query)}
                      className="text-left p-3 bg-gray-800 border border-gray-700 rounded-lg hover:border-purple-600 transition-colors group"
                    >
                      <div className="flex items-center">
                        <Sparkles className="h-4 w-4 text-purple-400 mr-2 group-hover:text-purple-300" />
                        <span className="text-sm text-gray-300 group-hover:text-gray-100">{query}</span>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Input Area */}
          <div className="border-t border-gray-800 px-4 py-4">
            <div className="max-w-4xl mx-auto">
              <div className="flex items-center space-x-4">
                <div className="flex-1 relative">
                  <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                    placeholder="Ask Genesis AI to analyze stocks, train models, or generate reports..."
                    className="w-full bg-gray-800 text-gray-100 rounded-lg pl-4 pr-12 py-3 focus:outline-none focus:ring-2 focus:ring-purple-500 border border-gray-700"
                    disabled={isProcessing}
                  />
                  <button
                    onClick={handleSendMessage}
                    disabled={!input.trim() || isProcessing}
                    className="absolute right-2 top-1/2 transform -translate-y-1/2 p-2 text-purple-400 hover:text-purple-300 disabled:text-gray-600 transition-colors"
                  >
                    <Send className="h-5 w-5" />
                  </button>
                </div>
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
        </div>

        {/* Side Panel - Agent Status */}
        <div className="w-80 bg-gray-900 border-l border-gray-800 p-6 hidden lg:block">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <Activity className="h-5 w-5 text-purple-400 mr-2" />
            Agent Status
          </h3>
          
          <div className="space-y-4">
            {/* Stock Data Agent */}
            <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium">Stock Data Agent</span>
                <div className={`w-2 h-2 rounded-full ${activeAgent === 'Data Agent' ? 'bg-yellow-500 animate-pulse' : 'bg-green-500'}`}></div>
              </div>
              <p className="text-sm text-gray-400">Fetches market data & indicators</p>
              <div className="mt-2 text-xs text-gray-500">
                • Yahoo Finance API<br />
                • 15+ technical indicators<br />
                • Real-time validation
              </div>
            </div>

            {/* Stock Analyzer Agent */}
            <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium">Stock Analyzer Agent</span>
                <div className={`w-2 h-2 rounded-full ${activeAgent === 'Analyzer Agent' ? 'bg-yellow-500 animate-pulse' : 'bg-green-500'}`}></div>
              </div>
              <p className="text-sm text-gray-400">ML models & backtesting</p>
              <div className="mt-2 text-xs text-gray-500">
                • XGBoost, Random Forest<br />
                • Multi-model comparison<br />
                • Strategy backtesting
              </div>
            </div>

            {/* Stock Reporter Agent */}
            <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium">Stock Reporter Agent</span>
                <div className={`w-2 h-2 rounded-full ${activeAgent === 'Reporter Agent' ? 'bg-yellow-500 animate-pulse' : 'bg-green-500'}`}></div>
              </div>
              <p className="text-sm text-gray-400">Charts & reports</p>
              <div className="mt-2 text-xs text-gray-500">
                • Interactive Plotly charts<br />
                • HTML reports<br />
                • Performance analytics
              </div>
            </div>
          </div>

          <div className="mt-6">
            <h4 className="text-sm font-medium text-gray-400 mb-3">Recent Activities</h4>
            <div className="space-y-2">
              {messages.slice(-3).reverse().map((msg, idx) => (
                msg.type === 'ai' && msg.agent !== 'System' && (
                  <div key={idx} className="flex items-center text-xs text-gray-500">
                    <Play className="h-3 w-3 mr-2" />
                    <span className="truncate">{msg.agent}: {msg.content.substring(0, 30)}...</span>
                  </div>
                )
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Visualization Modal */}
      {showVisualization && currentPlotlyData && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
          <div className={`bg-gray-900 rounded-xl border border-gray-700 ${expandedView ? 'w-full h-full' : 'w-full max-w-6xl max-h-[90vh]'} overflow-hidden`}>
            <div className="flex items-center justify-between p-4 border-b border-gray-700">
              <h3 className="text-lg font-semibold">Chart Visualization</h3>
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => setExpandedView(!expandedView)}
                  className="p-2 hover:bg-gray-800 rounded-lg transition-colors"
                >
                  {expandedView ? <Minimize2 className="h-5 w-5" /> : <Maximize2 className="h-5 w-5" />}
                </button>
                <button
                  onClick={() => {
                    setShowVisualization(false);
                    setCurrentPlotlyData(null);
                  }}
                  className="p-2 hover:bg-gray-800 rounded-lg transition-colors"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>
            </div>
            <div className="p-6 overflow-y-auto h-[calc(100%-73px)]">
              <div 
                ref={el => {
                  if (el && currentPlotlyData) {
                    try {
                      Plotly.newPlot(el, currentPlotlyData.data, currentPlotlyData.layout, { responsive: true });
                    } catch (e) {
                      console.error('Error rendering modal Plotly chart:', e);
                    }
                  }
                }}
                className="w-full h-[600px]"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default GenesisAIApp;