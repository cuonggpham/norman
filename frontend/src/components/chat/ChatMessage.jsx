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
                {isUser ? (
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="12" cy="8" r="5" />
                        <path d="M20 21a8 8 0 0 0-16 0" />
                    </svg>
                ) : (
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M12 3v18" />
                        <path d="M3 7h18" />
                        <path d="M3 7l3 10h3" />
                        <path d="M15 17h3l3-10" />
                        <circle cx="6" cy="17" r="3" />
                        <circle cx="18" cy="17" r="3" />
                    </svg>
                )}
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
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
                                <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
                            </svg>
                            Nguồn tham khảo ({sources.length})
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
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <circle cx="12" cy="12" r="10" />
                            <path d="M12 6v6l4 2" />
                        </svg>
                        {processingTime.toFixed(2)}s
                    </p>
                )}
            </div>
        </div>
    );
}
