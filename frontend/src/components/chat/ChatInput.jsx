import { useState, useRef, useEffect } from 'react';
import './ChatInput.css';

/**
 * Chat input with auto-resize textarea
 */
export default function ChatInput({ onSend, isLoading = false, disabled = false }) {
    const [value, setValue] = useState('');
    const textareaRef = useRef(null);

    // Auto-resize textarea
    useEffect(() => {
        const textarea = textareaRef.current;
        if (textarea) {
            textarea.style.height = 'auto';
            textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
        }
    }, [value]);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (value.trim() && !isLoading && !disabled) {
            onSend(value.trim());
            setValue('');
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e);
        }
    };

    return (
        <form className="chat-input" onSubmit={handleSubmit}>
            <div className="chat-input__container">
                <textarea
                    ref={textareaRef}
                    className="chat-input__textarea"
                    value={value}
                    onChange={(e) => setValue(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Nhập câu hỏi về pháp luật Nhật Bản..."
                    disabled={isLoading || disabled}
                    rows={1}
                    aria-label="Câu hỏi"
                />
                <button
                    type="submit"
                    className="chat-input__button"
                    disabled={!value.trim() || isLoading || disabled}
                    aria-label="Gửi câu hỏi"
                >
                    {isLoading ? (
                        <span className="chat-input__loading" />
                    ) : (
                        <svg
                            width="20"
                            height="20"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                        >
                            <path d="M22 2L11 13" />
                            <path d="M22 2L15 22L11 13L2 9L22 2Z" />
                        </svg>
                    )}
                </button>
            </div>
            <p className="chat-input__hint">
                Enter để gửi • Shift+Enter để xuống dòng
            </p>
        </form>
    );
}
