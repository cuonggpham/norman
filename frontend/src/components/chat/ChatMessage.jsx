import { useState, useCallback, useMemo, Children, isValidElement } from 'react';
import Markdown from 'react-markdown';
import SourceCard from './SourceCard';
import './ChatMessage.css';

/**
 * Parse inline citations [1], [2], etc. from text and convert to clickable spans
 */
function CitationText({ text, onCitationClick, sourcesCount }) {
    const parts = useMemo(() => {
        if (typeof text !== 'string') return [{ type: 'text', content: text }];

        // Match [1], [2], etc. but not things like [第7条]
        const regex = /\[(\d+)\]/g;
        const result = [];
        let lastIndex = 0;
        let match;

        while ((match = regex.exec(text)) !== null) {
            // Add text before the match
            if (match.index > lastIndex) {
                result.push({
                    type: 'text',
                    content: text.slice(lastIndex, match.index)
                });
            }

            const num = parseInt(match[1], 10);
            // Only make it clickable if within sources range (1-10 typically)
            if (num >= 1 && num <= Math.max(sourcesCount, 10)) {
                result.push({
                    type: 'citation',
                    num: num,
                    content: match[0]
                });
            } else {
                result.push({
                    type: 'text',
                    content: match[0]
                });
            }

            lastIndex = regex.lastIndex;
        }

        // Add remaining text
        if (lastIndex < text.length) {
            result.push({
                type: 'text',
                content: text.slice(lastIndex)
            });
        }

        return result;
    }, [text, sourcesCount]);

    if (parts.length === 1 && parts[0].type === 'text') {
        return <>{parts[0].content}</>;
    }

    return (
        <>
            {parts.map((part, idx) =>
                part.type === 'citation' ? (
                    <span
                        key={idx}
                        className="chat-message__citation"
                        role="button"
                        tabIndex={0}
                        onClick={() => onCitationClick(part.num - 1)}
                        onKeyDown={(e) => e.key === 'Enter' && onCitationClick(part.num - 1)}
                        title={`Xem nguồn ${part.num}`}
                    >
                        {part.content}
                    </span>
                ) : (
                    <span key={idx}>{part.content}</span>
                )
            )}
        </>
    );
}

/**
 * Recursively process children to add citation parsing to text nodes
 */
function processChildren(children, onCitationClick, sourcesCount) {
    return Children.map(children, (child) => {
        // If it's a string, parse for citations
        if (typeof child === 'string') {
            return (
                <CitationText
                    text={child}
                    onCitationClick={onCitationClick}
                    sourcesCount={sourcesCount}
                />
            );
        }
        // If it's a valid React element with children, recurse
        if (isValidElement(child) && child.props.children) {
            // Don't recurse into code blocks
            if (child.type === 'code' || child.type === 'pre') {
                return child;
            }
            return {
                ...child,
                props: {
                    ...child.props,
                    children: processChildren(child.props.children, onCitationClick, sourcesCount)
                }
            };
        }
        return child;
    });
}

/**
 * Renders a single chat message (user or assistant)
 */
export default function ChatMessage({ message }) {
    const { role, content, sources = [], processingTime } = message;
    const isUser = role === 'user';
    const [activeSourceIndex, setActiveSourceIndex] = useState(null);

    const handleCitationClick = useCallback((index) => {
        setActiveSourceIndex(index);
    }, []);

    const handleSourceActivated = useCallback(() => {
        setActiveSourceIndex(null);
    }, []);

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
                                p: ({ children }) => (
                                    <p className="chat-message__paragraph">
                                        {processChildren(children, handleCitationClick, sources.length)}
                                    </p>
                                ),
                                strong: ({ children }) => (
                                    <strong className="chat-message__strong">
                                        {processChildren(children, handleCitationClick, sources.length)}
                                    </strong>
                                ),
                                ul: ({ children }) => <ul className="chat-message__list">{children}</ul>,
                                ol: ({ children }) => <ol className="chat-message__list chat-message__list--ordered">{children}</ol>,
                                li: ({ children }) => (
                                    <li className="chat-message__list-item">
                                        {processChildren(children, handleCitationClick, sources.length)}
                                    </li>
                                ),
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
                                <SourceCard
                                    key={idx}
                                    source={source}
                                    index={idx}
                                    isActive={activeSourceIndex === idx}
                                    onActivated={handleSourceActivated}
                                />
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
