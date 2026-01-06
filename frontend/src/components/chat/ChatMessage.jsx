import Markdown from 'react-markdown';
import SourceCard from './SourceCard';
import './ChatMessage.css';

/**
 * Renders a single chat message (user or assistant)
 */
export default function ChatMessage({ message }) {
    const { role, content, sources = [], processingTime } = message;
    const isUser = role === 'user';

    return (
        <div className={`chat-message chat-message--${role}`}>
            <div className="chat-message__avatar">
                {isUser ? '👤' : '⚖️'}
            </div>
            <div className="chat-message__body">
                <div className="chat-message__content">
                    {isUser ? (
                        <p>{content}</p>
                    ) : (
                        <Markdown
                            components={{
                                // Custom renderers for better styling
                                p: ({ children }) => <p className="chat-message__paragraph">{children}</p>,
                                strong: ({ children }) => <strong className="chat-message__strong">{children}</strong>,
                                ul: ({ children }) => <ul className="chat-message__list">{children}</ul>,
                                ol: ({ children }) => <ol className="chat-message__list chat-message__list--ordered">{children}</ol>,
                                li: ({ children }) => <li className="chat-message__list-item">{children}</li>,
                                code: ({ children, inline }) => (
                                    inline
                                        ? <code className="chat-message__code">{children}</code>
                                        : <pre className="chat-message__pre"><code>{children}</code></pre>
                                ),
                            }}
                        >
                            {content}
                        </Markdown>
                    )}
                </div>

                {/* Sources for assistant messages */}
                {!isUser && sources.length > 0 && (
                    <div className="chat-message__sources">
                        <h4 className="chat-message__sources-title">
                            📚 Nguồn tham khảo ({sources.length})
                        </h4>
                        <div className="chat-message__sources-list">
                            {sources.map((source, idx) => (
                                <SourceCard key={idx} source={source} index={idx} />
                            ))}
                        </div>
                    </div>
                )}

                {/* Processing time for assistant */}
                {!isUser && processingTime && (
                    <p className="chat-message__meta">
                        ⏱️ {processingTime.toFixed(2)}s
                    </p>
                )}
            </div>
        </div>
    );
}
