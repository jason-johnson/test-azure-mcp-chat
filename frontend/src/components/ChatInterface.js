import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import { useMsal } from '@azure/msal-react';
import '../ChatInterface.css';
import { mcpScopes } from '../authConfig';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:7071/api';

const AGENT_MODES = {
  azure: { label: 'Azure Resources', endpoint: 'query/azure', icon: '☁️' },
  tickets: { label: 'Support Tickets', endpoint: 'query/tickets', icon: '🎫' },
  both: { label: 'Both', endpoint: 'query/both', icon: '🔍' },
};

const ChatInterface = () => {
  const { instance, accounts } = useMsal();

  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [agentMode, setAgentMode] = useState('both');
  const chatHistoryRef = useRef(null);

  useEffect(() => {
    if (chatHistoryRef.current) {
      chatHistoryRef.current.scrollTop = chatHistoryRef.current.scrollHeight;
    }
  }, [messages]);

  const acquireMcpToken = async () => {
    const account = accounts[0];
    if (!account) throw new Error('No authenticated account');

    try {
      const response = await instance.acquireTokenSilent({ scopes: mcpScopes, account });
      return response.accessToken;
    } catch {
      const response = await instance.acquireTokenPopup({ scopes: mcpScopes, account });
      return response.accessToken;
    }
  };

  const pollForResult = async (instanceId) => {
    const maxAttempts = 60;
    for (let i = 0; i < maxAttempts; i++) {
      await new Promise((r) => setTimeout(r, 3000));
      const res = await axios.get(`${API_URL}/query/status/${instanceId}`);
      const status = res.data;

      if (status.runtimeStatus === 'Completed') {
        return status.output;
      } else if (status.runtimeStatus === 'Failed') {
        throw new Error('Query failed');
      }
    }
    throw new Error('Query timed out');
  };

  const formatOutput = (output, mode) => {
    if (!output) return 'No results returned.';

    if (mode === 'both') {
      const parts = [];
      const azureResult = output.azure?.result || output.azure?.error;
      const ticketResult = output.tickets?.result || output.tickets?.error;

      if (azureResult) {
        parts.push(`## ☁️ Azure Resources\n\n${azureResult}`);
      }
      if (ticketResult) {
        parts.push(`## 🎫 Support Tickets\n\n${ticketResult}`);
      }
      return parts.join('\n\n---\n\n') || 'No results returned.';
    }

    return output.result || output.error || 'No results returned.';
  };

  const submitQuery = async () => {
    const text = query.trim();
    if (!text || loading) return;

    setQuery('');
    setLoading(true);
    setMessages((prev) => [...prev, { role: 'user', content: text }]);

    try {
      const token = await acquireMcpToken();
      const mode = AGENT_MODES[agentMode];

      const startRes = await axios.post(
        `${API_URL}/${mode.endpoint}`,
        { query: text, userAccessToken: token },
        { headers: { 'Content-Type': 'application/json' } }
      );

      const output = await pollForResult(startRes.data.id);
      const formatted = formatOutput(output, agentMode);
      setMessages((prev) => [...prev, { role: 'bot', content: formatted }]);
    } catch (error) {
      console.error('Query error:', error);
      const msg =
        error.message === 'No authenticated account'
          ? 'Please sign in to continue.'
          : `Error: ${error.message || 'Something went wrong. Please try again.'}`;
      setMessages((prev) => [...prev, { role: 'bot', content: msg }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submitQuery();
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  const userName = accounts[0]?.name || accounts[0]?.username || 'User';

  return (
    <div className="page-container">
      <div className="chat-title-container">
        <h1>Azure Support Assistant</h1>
        <div className="header-right">
          <span className="user-name">{userName}</span>
          <button onClick={clearChat} className="clear-btn">Clear Chat</button>
        </div>
      </div>

      <div className="chat-container">
        <div ref={chatHistoryRef} className="chat-history">
          {messages.length === 0 && (
            <div className="welcome-message">
              <h2>How can I help you?</h2>
              <p>Ask about Azure resources, search support tickets, or query both at once.</p>
              <div className="example-queries">
                <button onClick={() => setQuery('List all web apps in my subscription')}>
                  ☁️ List all web apps
                </button>
                <button onClick={() => setQuery('Show open high priority tickets')}>
                  🎫 Open high priority tickets
                </button>
                <button onClick={() => setQuery('Are there any tickets related to VMs not responding?')}>
                  🔍 VM issues + tickets
                </button>
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} className={`chat-message ${msg.role}`}>
              <ReactMarkdown>{msg.content}</ReactMarkdown>
            </div>
          ))}

          {loading && (
            <div className="loading-container">
              <div className="loading-spinner" />
              <div className="loading-message">
                {agentMode === 'both' ? 'Querying Azure resources and searching tickets...' :
                 agentMode === 'azure' ? 'Querying Azure resources...' :
                 'Searching support tickets...'}
              </div>
            </div>
          )}
        </div>

        <div className="query-input-area">
          <div className="agent-mode-selector">
            {Object.entries(AGENT_MODES).map(([key, mode]) => (
              <button
                key={key}
                className={`mode-btn ${agentMode === key ? 'active' : ''}`}
                onClick={() => setAgentMode(key)}
                disabled={loading}
              >
                {mode.icon} {mode.label}
              </button>
            ))}
          </div>
          <div className="chat-input">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about Azure resources or support tickets..."
              disabled={loading}
            />
            <button onClick={submitQuery} disabled={loading || !query.trim()}>
              {loading ? '...' : 'Send'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;