import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import '../ChatInterface.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:7071/api';

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [threadId, setThreadId] = useState(null);
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async (e) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || loading) return;

    setInput('');
    setMessages((prev) => [...prev, { role: 'user', content: text }]);
    setLoading(true);

    try {
      const url = threadId
        ? `${API_URL}/chat/${threadId}`
        : `${API_URL}/chat`;

      const res = await axios.post(url, { message: text });

      if (res.data.threadId) {
        setThreadId(res.data.threadId);
      }

      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: res.data.response },
      ]);
    } catch (err) {
      const errorMsg =
        err.response?.data?.error || err.message || 'Something went wrong';
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `**Error:** ${errorMsg}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const startNewChat = () => {
    setMessages([]);
    setThreadId(null);
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h1>Azure MCP Chat</h1>
        <button className="new-chat-btn" onClick={startNewChat}>
          New Chat
        </button>
      </div>

      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="empty-state">
            <p>Ask me anything about your Azure resources.</p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <div className="message-content">
              <ReactMarkdown>{msg.content}</ReactMarkdown>
            </div>
          </div>
        ))}
        {loading && (
          <div className="message assistant">
            <div className="message-content loading-dots">
              <span>.</span><span>.</span><span>.</span>
            </div>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>

      <form className="chat-input-form" onSubmit={sendMessage}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type a message..."
          disabled={loading}
          autoFocus
        />
        <button type="submit" disabled={loading || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  );
};

export default ChatInterface;
