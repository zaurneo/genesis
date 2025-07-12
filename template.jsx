import React, { useState, useRef, useEffect } from 'react';
import { Brain, Send, Bot, User, TrendingUp, BarChart3, FileText, Loader2, ChevronRight, Activity, Sparkles, Database, Play, X, Maximize2, Minimize2 } from 'lucide-react';
import * as Plotly from 'plotly';

// Configuration - Update these to match your backend
const API_CONFIG = {
  WS_URL: 'ws://localhost:8000/ws',
  API_BASE: 'http://localhost:8000',
  USE_MOCK: true  // Set to false when you have your backend running
};

const GenesisAIApp = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'ai',
      agent: 'Supervisor',
      content: "Welcome to Genesis AI! I'm your trading analysis supervisor. I coordinate with our specialized agents:\n\n• **Stock Data Agent** - Fetches market data and technical indicators\n• **Stock Analyzer Agent** - Trains ML models and performs backtesting\n• **Stock Reporter Agent** - Creates visualizations and reports\n\nTry asking: 'Analyze Apple stock for the last 2 years' or 'Compare XGBoost vs Random Forest for Tesla'",
      timestamp: new Date().toLocaleTimeString(),
      status: 'complete'
    }
  ]);
  const [input, setInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeAgent, setActiveAgent] = useState(null);
  const [showVisualization, setShowVisualization] = useState(false);
  const [expandedView, setExpandedView] = useState(false);
  const [plotlyData, setPlotlyData] = useState(null);
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
        Plotly.newPlot(
          plotlyRefs.current[message.id],
          message.plotlyData.data,
          message.plotlyData.layout,
          { responsive: true }
        );
      }
    });
  }, [messages]);

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

    // If using mock data or backend not available
    if (API_CONFIG.USE_MOCK) {
      setTimeout(() => {
        const aiResponse = generateAIResponse(input);
        setMessages(prev => [...prev, aiResponse]);
        setIsProcessing(false);
      }, 2000);
      return;
    }

    // Connect to your Genesis backend via WebSocket
    try {
      const ws = new WebSocket(API_CONFIG.WS_URL);
      
      ws.onopen = () => {
        ws.send(JSON.stringify({ query: input }));
      };

      ws.onmessage = (event) => {
        const agentUpdate = JSON.parse(event.data);
        
        setMessages(prev => [...prev, {
          id: Date.now(),
          type: 'ai',
          agent: agentUpdate.agent,
          content: agentUpdate.content,
          status: agentUpdate.status,
          timestamp: new Date().toLocaleTimeString(),
          hasVisualization: agentUpdate.visualization_data ? true : false,
          plotlyData: agentUpdate.plotly_data || null  // Plotly figure data from backend
        }]);
        
        // Update active agent indicator
        if (agentUpdate.status === 'processing') {
          setActiveAgent(agentUpdate.agent);
        } else if (agentUpdate.status === 'complete') {
          setActiveAgent(null);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setIsProcessing(false);
        // Fallback to simulated response if backend not available
        const aiResponse = generateAIResponse(input);
        setMessages(prev => [...prev, aiResponse]);
      };

      ws.onclose = () => {
        setIsProcessing(false);
      };
    } catch (error) {
      // Fallback to mock data if WebSocket fails
      console.error('Connection error:', error);
      setTimeout(() => {
        const aiResponse = generateAIResponse(input);
        setMessages(prev => [...prev, aiResponse]);
        setIsProcessing(false);
      }, 2000);
    }
  };

  const generateAIResponse = (query) => {
    const lowerQuery = query.toLowerCase();
    
    if (lowerQuery.includes('analyze') && (lowerQuery.includes('apple') || lowerQuery.includes('aapl'))) {
      setActiveAgent('Data Agent');
      setTimeout(() => {
        setMessages(prev => [...prev, {
          id: prev.length + 1,
          type: 'ai',
          agent: 'Data Agent',
          content: "Fetching AAPL data (2 years, daily intervals)...\n✓ 504 records retrieved\n✓ Technical indicators applied (SMA, EMA, RSI, MACD)\n✓ Data quality: Excellent (0 missing values)",
          timestamp: new Date().toLocaleTimeString(),
          status: 'complete'
        }]);
        
        setTimeout(() => {
          setActiveAgent('Analyzer Agent');
          setMessages(prev => [...prev, {
            id: prev.length + 1,
            type: 'ai',
            agent: 'Analyzer Agent',
            content: "Training ML models on AAPL data:\n\n**XGBoost Model**\n• Parameters optimized for swing trading\n• Cross-validation R²: 0.847\n• Feature importance: Price momentum (34%), Volume ratio (22%)\n\n**Random Forest Model**\n• Ensemble of 200 trees, max depth 15\n• Cross-validation R²: 0.823\n• Directional accuracy: 67.3%\n\nBacktesting complete. XGBoost shows superior performance (Sharpe 1.42)",
            timestamp: new Date().toLocaleTimeString(),
            status: 'complete',
            hasVisualization: true
          }]);
          setActiveAgent(null);
        }, 3000);
      }, 1500);

      return {
        id: messages.length + 2,
        type: 'ai',
        agent: 'Supervisor',
        content: "I'll coordinate a comprehensive analysis of Apple (AAPL) stock. Let me engage our specialized agents...",
        timestamp: new Date().toLocaleTimeString(),
        status: 'processing'
      };
    }
    
    if (lowerQuery.includes('backtest') || lowerQuery.includes('show') && lowerQuery.includes('chart')) {
      // Generate sample Plotly data for backtesting visualization
      const samplePlotlyData = {
        data: [
          {
            x: Array.from({length: 100}, (_, i) => new Date(2023, 0, i + 1)),
            y: Array.from({length: 100}, (_, i) => 100 + Math.random() * 20 + i * 0.3),
            name: 'Portfolio Value',
            type: 'scatter',
            mode: 'lines',
            line: { color: '#8b5cf6', width: 2 }
          },
          {
            x: Array.from({length: 100}, (_, i) => new Date(2023, 0, i + 1)),
            y: Array.from({length: 100}, (_, i) => 100 + i * 0.25),
            name: 'Buy & Hold',
            type: 'scatter',
            mode: 'lines',
            line: { color: '#6b7280', width: 2, dash: 'dash' }
          }
        ],
        layout: {
          title: 'Backtesting Results: XGBoost Strategy vs Buy & Hold',
          xaxis: { title: 'Date' },
          yaxis: { title: 'Portfolio Value ($)' },
          hovermode: 'x unified',
          plot_bgcolor: '#1f2937',
          paper_bgcolor: '#111827',
          font: { color: '#e5e7eb' },
          legend: { x: 0, y: 1 }
        }
      };
      
      return {
        id: messages.length + 2,
        type: 'ai',
        agent: 'Reporter Agent',
        content: "Here's the backtesting visualization for your strategy:",
        timestamp: new Date().toLocaleTimeString(),
        status: 'complete',
        plotlyData: samplePlotlyData
      };
    }
    
    return {
      id: messages.length + 2,
      type: 'ai',
      agent: 'Supervisor',
      content: "I understand your request. Here are some examples of what I can help with:\n\n• Stock analysis with ML predictions\n• Multi-model performance comparison\n• Technical indicator analysis\n• Backtesting trading strategies\n• Generating professional reports\n\nWhich would you like to explore?",
      timestamp: new Date().toLocaleTimeString(),
      status: 'complete'
    };
  };

  const suggestedQueries = [
    "Analyze Tesla stock with ML models",
    "Show me backtesting visualization",
    "Compare all models for Microsoft",
    "Generate report for Amazon stock"
  ];

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
                  <div className="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse"></div>
                  <span>All Agents Online</span>
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
            {API_CONFIG.USE_MOCK && (
              <div className="max-w-4xl mx-auto mb-4 p-4 bg-yellow-900/20 border border-yellow-700 rounded-lg">
                <p className="text-sm text-yellow-300">
                  <strong>Mock Mode Active:</strong> To connect to your Genesis backend, set USE_MOCK to false in the API_CONFIG and ensure your backend is running on {API_CONFIG.WS_URL}
                </p>
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
                        </div>
                      )}
                      <div
                        className={`inline-block px-4 py-2 rounded-lg ${
                          message.type === 'user'
                            ? 'bg-blue-600 text-white'
                            : 'bg-gray-800 border border-gray-700'
                        }`}
                      >
                        <p className="whitespace-pre-wrap">{message.content}</p>
                        {message.plotlyData && (
                          <div className="mt-4">
                            <div 
                              ref={el => plotlyRefs.current[message.id] = el}
                              className="w-full h-96 rounded-lg overflow-hidden"
                            />
                          </div>
                        )}
                        {message.hasVisualization && (
                          <button
                            onClick={() => {
                              setShowVisualization(true);
                              setPlotlyData(message.plotlyData);
                            }}
                            className="mt-3 flex items-center text-purple-400 hover:text-purple-300 transition-colors"
                          >
                            <BarChart3 className="h-4 w-4 mr-2" />
                            View Performance Charts
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
                        <span className="text-gray-400">Genesis AI is thinking...</span>
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
                <span>Connected to Yahoo Finance</span>
                <span className="mx-2">•</span>
                <Brain className="h-3 w-3 mr-1" />
                <span>GPT-4 Enhanced</span>
                <span className="mx-2">•</span>
                <TrendingUp className="h-3 w-3 mr-1" />
                <span>6 ML Models Available</span>
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
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              </div>
              <p className="text-sm text-gray-400">Ready to fetch market data</p>
              <div className="mt-2 text-xs text-gray-500">
                • 15+ technical indicators<br />
                • Multi-timeframe support<br />
                • Real-time validation
              </div>
            </div>

            {/* Stock Analyzer Agent */}
            <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium">Stock Analyzer Agent</span>
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              </div>
              <p className="text-sm text-gray-400">ML models ready</p>
              <div className="mt-2 text-xs text-gray-500">
                • XGBoost, Random Forest<br />
                • Neural Networks, SVR<br />
                • 6 backtesting strategies
              </div>
            </div>

            {/* Stock Reporter Agent */}
            <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium">Stock Reporter Agent</span>
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              </div>
              <p className="text-sm text-gray-400">Visualization engine online</p>
              <div className="mt-2 text-xs text-gray-500">
                • Interactive Plotly charts<br />
                • Professional HTML reports<br />
                • Performance analytics
              </div>
            </div>
          </div>

          <div className="mt-6">
            <h4 className="text-sm font-medium text-gray-400 mb-3">Recent Activities</h4>
            <div className="space-y-2">
              <div className="flex items-center text-xs text-gray-500">
                <Play className="h-3 w-3 mr-2" />
                <span>Model training completed</span>
              </div>
              <div className="flex items-center text-xs text-gray-500">
                <FileText className="h-3 w-3 mr-2" />
                <span>Report generated</span>
              </div>
              <div className="flex items-center text-xs text-gray-500">
                <BarChart3 className="h-3 w-3 mr-2" />
                <span>Backtest analysis ready</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Visualization Modal */}
      {showVisualization && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
          <div className={`bg-gray-900 rounded-xl border border-gray-700 ${expandedView ? 'w-full h-full' : 'w-full max-w-6xl max-h-[90vh]'} overflow-hidden`}>
            <div className="flex items-center justify-between p-4 border-b border-gray-700">
              <h3 className="text-lg font-semibold">Model Performance Analysis</h3>
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
                    setPlotlyData(null);
                  }}
                  className="p-2 hover:bg-gray-800 rounded-lg transition-colors"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>
            </div>
            <div className="p-6 overflow-y-auto h-[calc(100%-73px)]">
              {/* Render actual Plotly chart if data available */}
              {plotlyData ? (
                <div 
                  ref={el => {
                    if (el && plotlyData) {
                      Plotly.newPlot(el, plotlyData.data, plotlyData.layout, { responsive: true });
                    }
                  }}
                  className="w-full h-[600px] mb-6"
                />
              ) : (
                /* Placeholder charts */
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                    <h4 className="text-base font-medium mb-4">Returns Comparison</h4>
                    <div className="h-64 bg-gray-700 rounded flex items-center justify-center">
                      <BarChart3 className="h-12 w-12 text-gray-600" />
                    </div>
                  </div>
                  <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                    <h4 className="text-base font-medium mb-4">Prediction Accuracy</h4>
                    <div className="h-64 bg-gray-700 rounded flex items-center justify-center">
                      <TrendingUp className="h-12 w-12 text-gray-600" />
                    </div>
                  </div>
                  <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                    <h4 className="text-base font-medium mb-4">Risk Metrics</h4>
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Sharpe Ratio</span>
                        <span className="font-medium">1.42</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Max Drawdown</span>
                        <span className="font-medium text-red-400">-8.3%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Win Rate</span>
                        <span className="font-medium text-green-400">67.3%</span>
                      </div>
                    </div>
                  </div>
                  <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                    <h4 className="text-base font-medium mb-4">Model Rankings</h4>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between p-2 bg-gray-700 rounded">
                        <span>XGBoost</span>
                        <span className="text-green-400">★★★★★</span>
                      </div>
                      <div className="flex items-center justify-between p-2 bg-gray-700 rounded">
                        <span>Random Forest</span>
                        <span className="text-green-400">★★★★☆</span>
                      </div>
                      <div className="flex items-center justify-between p-2 bg-gray-700 rounded">
                        <span>Neural Network</span>
                        <span className="text-yellow-400">★★★☆☆</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
              <div className="mt-6 flex justify-end">
                <button className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors flex items-center">
                  <FileText className="h-4 w-4 mr-2" />
                  Generate Full Report
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default GenesisAIApp;