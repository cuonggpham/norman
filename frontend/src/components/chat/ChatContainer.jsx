import { useState, useRef, useEffect } from 'react';
import ChatMessage from './ChatMessage';
import ChatInput from './ChatInput';
import LoadingSpinner from '../ui/LoadingSpinner';
import { chat } from '../../services/api';
import './ChatContainer.css';

const EXAMPLE_PROMPTS = [
    'Quy định về thời gian làm việc theo Luật Lao động Nhật Bản?',
    'Thời hạn thuê nhà theo pháp luật Nhật Bản là bao lâu?',
    'Điều kiện để thành lập công ty tại Nhật Bản?',
];

/**
 * Main chat container with message history and input
 */
export default function ChatContainer() {
    const [messages, setMessages] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const messagesEndRef = useRef(null);

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
                        <div className="chat-container__welcome-icon">⚖️</div>
                        <h2 className="chat-container__welcome-title">
                            Xin chào! Tôi là Norman
                        </h2>
                        <p className="chat-container__welcome-text">
                            Trợ lý tra cứu pháp luật Nhật Bản. Hãy đặt câu hỏi bằng tiếng Việt,
                            tôi sẽ trả lời và trích dẫn các điều luật liên quan.
                        </p>
                        <div className="chat-container__examples">
                            <p className="chat-container__examples-label">Thử hỏi:</p>
                            {EXAMPLE_PROMPTS.map((prompt, idx) => (
                                <button
                                    key={idx}
                                    className="chat-container__example-btn"
                                    onClick={() => handleExampleClick(prompt)}
                                >
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
}
