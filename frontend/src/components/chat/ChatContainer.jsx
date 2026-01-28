import { useState, useRef, useEffect, useImperativeHandle, forwardRef } from 'react';
import ChatMessage from './ChatMessage';
import ChatInput from './ChatInput';
import LoadingSpinner from '../ui/LoadingSpinner';
import { chat } from '../../services/api';
import './ChatContainer.css';

const EXAMPLE_PROMPTS = [
    'Thuế thu nhập cá nhân ở Nhật tính như thế nào?',
    'Bảo hiểm y tế ở Nhật có những loại nào?',
    'Thu nhập từ tiền mã hóa có phải đóng thuế không?',
];

/**
 * Main chat container with message history and input
 * Exposes resetChat method via ref
 */
const ChatContainer = forwardRef(function ChatContainer(props, ref) {
    const [messages, setMessages] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const messagesEndRef = useRef(null);

    // Expose resetChat method to parent
    useImperativeHandle(ref, () => ({
        resetChat: () => {
            setMessages([]);
            setError(null);
        }
    }));

    // Auto-scroll to bottom on new messages
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, isLoading]);

    const handleSend = async (query) => {
        // Add user message
        const userMessage = { role: 'user', content: query };
        setMessages((prev) => [...prev, userMessage]);
        setIsLoading(true);
        setError(null);

        try {
            const response = await chat(query);

            // Add assistant message with sources
            const assistantMessage = {
                role: 'assistant',
                content: response.answer,
                sources: response.sources || [],
                processingTime: response.processing_time,
            };
            setMessages((prev) => [...prev, assistantMessage]);
        } catch (err) {
            console.error('Chat error:', err);
            setError(err.message || 'Có lỗi xảy ra. Vui lòng thử lại.');
            // Add error message
            setMessages((prev) => [
                ...prev,
                {
                    role: 'assistant',
                    content: '❌ Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi của bạn. Vui lòng thử lại.',
                    sources: [],
                },
            ]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleExampleClick = (prompt) => {
        handleSend(prompt);
    };

    return (
        <div className="chat-container">
            <div className="chat-container__messages">
                {messages.length === 0 ? (
                    <div className="chat-container__welcome">
                        <div className="chat-container__welcome-glow" />
                        <div className="chat-container__welcome-icon">
                            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M12 3v18" />
                                <path d="M3 7h18" />
                                <path d="M3 7l3 10h3" />
                                <path d="M15 17h3l3-10" />
                                <circle cx="6" cy="17" r="3" />
                                <circle cx="18" cy="17" r="3" />
                            </svg>
                        </div>
                        <h2 className="chat-container__welcome-title">
                            Xin chào! Tôi là Norman
                        </h2>
                        <p className="chat-container__welcome-text">
                            Trợ lý tra cứu pháp luật Nhật Bản. Hãy đặt câu hỏi bằng tiếng Việt,
                            tôi sẽ trả lời và trích dẫn các điều luật liên quan.
                        </p>
                        <div className="chat-container__examples">
                            <p className="chat-container__examples-label">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ marginRight: '6px', verticalAlign: 'middle' }}>
                                    <polygon points="5 3 19 12 5 21 5 3" />
                                </svg>
                                Thử hỏi
                            </p>
                            {EXAMPLE_PROMPTS.map((prompt, idx) => (
                                <button
                                    key={idx}
                                    className="chat-container__example-btn"
                                    onClick={() => handleExampleClick(prompt)}
                                >
                                    <span className="chat-container__example-icon">
                                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                            <circle cx="12" cy="12" r="10" />
                                            <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
                                            <path d="M12 17h.01" />
                                        </svg>
                                    </span>
                                    {prompt}
                                </button>
                            ))}
                        </div>
                    </div>
                ) : (
                    messages.map((msg, idx) => (
                        <ChatMessage key={idx} message={msg} />
                    ))
                )}

                {/* Loading indicator */}
                {isLoading && (
                    <div className="chat-container__loading">
                        <LoadingSpinner size="md" />
                        <span>Đang tìm kiếm và phân tích...</span>
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            <ChatInput onSend={handleSend} isLoading={isLoading} />
        </div>
    );
});

export default ChatContainer;
