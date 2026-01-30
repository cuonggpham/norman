import { useState, useEffect, useRef } from 'react';
import { translate } from '../../services/api';
import './SourceCard.css';

/**
 * Displays a legal source citation with expandable content
 * Shows structured hierarchy: Law → Chapter → Article → Paragraph
 * Supports active state for when user clicks inline citation [1], [2], etc.
 */
const TEXT_TRUNCATE_LENGTH = 500;

export default function SourceCard({ source, index, isActive = false, onActivated }) {
    const [isExpanded, setIsExpanded] = useState(false);
    const [isTextExpanded, setIsTextExpanded] = useState(false);
    const [translatedText, setTranslatedText] = useState(null);
    const [isTranslating, setIsTranslating] = useState(false);
    const [translateError, setTranslateError] = useState(null);
    const cardRef = useRef(null);

    const {
        law_title = 'Unknown Law',
        article = '',
        article_title = '',  // Fallback for old data
        text = '',
        score = 0,
        highlight_path = '',
        // Additional metadata
        law_id = '',
        chapter_title = '',
        article_caption = '',
        paragraph_num = '',
    } = source;

    // Use article or article_title (backward compatibility)
    const displayArticle = article || article_title;

    const relevancePercent = Math.round(score * 100);

    // Auto-expand and scroll when activated by citation click
    useEffect(() => {
        if (isActive) {
            setIsExpanded(true);
            if (cardRef.current) {
                cardRef.current.scrollIntoView({
                    behavior: 'smooth',
                    block: 'center'
                });
            }
            // Clear active state after animation
            const timer = setTimeout(() => {
                onActivated?.();
            }, 2000);
            return () => clearTimeout(timer);
        }
    }, [isActive, onActivated]);

    // Build hierarchy path for display
    const hierarchy = [];
    if (chapter_title) hierarchy.push({ label: 'Chương', value: chapter_title });
    if (displayArticle) hierarchy.push({ label: 'Điều', value: displayArticle });
    if (paragraph_num) hierarchy.push({ label: 'Khoản', value: paragraph_num });

    // Handle translation request
    const handleTranslate = async () => {
        if (translatedText || isTranslating) return;

        setIsTranslating(true);
        setTranslateError(null);

        try {
            const result = await translate(text);
            setTranslatedText(result.translated);
        } catch (error) {
            setTranslateError(error.message || 'Không thể dịch văn bản');
        } finally {
            setIsTranslating(false);
        }
    };

    return (
        <div
            className={`source-card ${isActive ? 'source-card--active' : ''}`}
            ref={cardRef}
            id={`source-${index}`}
        >
            {/* Header - Always visible */}
            <button
                className="source-card__header"
                onClick={() => setIsExpanded(!isExpanded)}
                aria-expanded={isExpanded}
            >
                <span className="source-card__index">{index + 1}</span>

                <div className="source-card__info">
                    <h4 className="source-card__title">{law_title}</h4>
                    {displayArticle && (
                        <span className="source-card__article">{displayArticle}</span>
                    )}
                </div>

                {/* Score Bar - Original Design */}
                <div className="source-card__score">
                    <div
                        className="source-card__score-bar"
                        style={{ width: `${relevancePercent}%` }}
                    />
                    <span className="source-card__score-text">{relevancePercent}%</span>
                </div>

                <svg
                    className={`source-card__chevron ${isExpanded ? 'source-card__chevron--open' : ''}`}
                    width="20"
                    height="20"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                >
                    <path d="M6 9l6 6 6-6" />
                </svg>
            </button>

            {/* Expanded Content */}
            {isExpanded && (
                <div className="source-card__content">
                    {/* Metadata Section */}
                    <div className="source-card__meta">
                        {/* Law ID if available */}
                        {law_id && (
                            <div className="source-card__meta-row">
                                <span className="source-card__meta-label">Mã luật</span>
                                <span className="source-card__meta-value source-card__law-id">{law_id}</span>
                            </div>
                        )}

                        {/* Article Caption if available */}
                        {article_caption && (
                            <div className="source-card__meta-row">
                                <span className="source-card__meta-label">Tiêu đề</span>
                                <span className="source-card__meta-value">{article_caption}</span>
                            </div>
                        )}
                    </div>

                    {/* Hierarchy Path */}
                    {hierarchy.length > 0 && (
                        <div className="source-card__hierarchy">
                            {hierarchy.map((item, idx) => (
                                <span key={idx} className="source-card__hierarchy-item">
                                    <span className="source-card__hierarchy-label">{item.label}</span>
                                    <span className="source-card__hierarchy-value">{item.value}</span>
                                    {idx < hierarchy.length - 1 && (
                                        <span className="source-card__hierarchy-separator">›</span>
                                    )}
                                </span>
                            ))}
                        </div>
                    )}

                    {/* Legacy highlight_path support */}
                    {!hierarchy.length && highlight_path && typeof highlight_path === 'object' && Object.keys(highlight_path).length > 0 && (
                        <p className="source-card__path">
                            {Object.values(highlight_path).filter(Boolean).join(' › ')}
                        </p>
                    )}
                    {!hierarchy.length && highlight_path && typeof highlight_path === 'string' && (
                        <p className="source-card__path">{highlight_path}</p>
                    )}

                    {/* Text Content */}
                    <div className="source-card__text-section">
                        <div className="source-card__text-header">
                            <span className="source-card__text-label">Nội dung gốc (日本語)</span>
                            <button
                                className={`source-card__translate-btn ${translatedText ? 'source-card__translate-btn--done' : ''}`}
                                onClick={handleTranslate}
                                disabled={isTranslating || translatedText}
                                title={translatedText ? 'Đã dịch' : 'Dịch sang tiếng Việt'}
                            >
                                {isTranslating ? (
                                    <>
                                        <svg className="source-card__translate-spinner" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                            <path d="M12 2v4m0 12v4m-7.07-3.93l2.83-2.83m8.48-8.48l2.83-2.83M2 12h4m12 0h4M4.93 4.93l2.83 2.83m8.48 8.48l2.83 2.83" />
                                        </svg>
                                        <span>Đang dịch...</span>
                                    </>
                                ) : translatedText ? (
                                    <>
                                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                            <path d="M20 6L9 17l-5-5" />
                                        </svg>
                                        <span>Đã dịch</span>
                                    </>
                                ) : (
                                    <>
                                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                            <path d="M12.87 15.07l-2.54-2.51.03-.03A17.52 17.52 0 0014.07 6H17V4h-7V2H8v2H1v2h11.17C11.5 7.92 10.44 9.75 9 11.35 8.07 10.32 7.3 9.19 6.69 8h-2c.73 1.63 1.73 3.17 2.98 4.56l-5.09 5.02L4 19l5-5 3.11 3.11.76-2.04M18.5 10h-2L12 22h2l1.12-3h4.75L21 22h2l-4.5-12m-2.62 7l1.62-4.33L19.12 17h-3.24z" />
                                        </svg>
                                        <span>Dịch</span>
                                    </>
                                )}
                            </button>
                        </div>
                        <div className="source-card__text-wrapper">
                            <p className="source-card__text">
                                {text.length > TEXT_TRUNCATE_LENGTH && !isTextExpanded
                                    ? text.slice(0, TEXT_TRUNCATE_LENGTH) + '...'
                                    : text}
                            </p>
                            {text.length > TEXT_TRUNCATE_LENGTH && (
                                <button
                                    className="source-card__expand-btn"
                                    onClick={() => setIsTextExpanded(!isTextExpanded)}
                                >
                                    {isTextExpanded ? (
                                        <>
                                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                                <path d="M18 15l-6-6-6 6" />
                                            </svg>
                                            <span>Thu gọn</span>
                                        </>
                                    ) : (
                                        <>
                                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                                <path d="M6 9l6 6 6-6" />
                                            </svg>
                                            <span>Xem thêm</span>
                                        </>
                                    )}
                                </button>
                            )}
                        </div>
                    </div>

                    {/* Translated Content */}
                    {translateError && (
                        <div className="source-card__translate-error">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <circle cx="12" cy="12" r="10" />
                                <path d="M12 8v4m0 4h.01" />
                            </svg>
                            <span>{translateError}</span>
                        </div>
                    )}

                    {translatedText && (
                        <div className="source-card__text-section source-card__translated-section">
                            <span className="source-card__text-label source-card__text-label--vietnamese">Bản dịch tiếng Việt</span>
                            <div className="source-card__text-wrapper source-card__text-wrapper--translated">
                                <p className="source-card__text source-card__text--translated">{translatedText}</p>
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

